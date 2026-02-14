#include <cudaTypedefs.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x.cuh"

using namespace cute;

namespace {

bool pplx_debug_enabled() {
  auto* env = std::getenv("VLLM_PPLX_DEBUG");
  return env != nullptr && std::atoi(env) != 0;
}

void maybe_log_sm100_grouped_moe_config(
    const char* branch_name, bool swap_ab, torch::Tensor const& out_tensors,
    torch::Tensor const& a_tensors, torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales, torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets, torch::Tensor const& problem_sizes,
    bool per_act_token, bool per_out_ch) {
  if (!pplx_debug_enabled()) {
    return;
  }

  static bool logged_m64 = false;
  static bool logged_n8192 = false;
  static bool logged_default = false;

  bool* logged = &logged_default;
  if (std::strcmp(branch_name, "M64") == 0) {
    logged = &logged_m64;
  } else if (std::strcmp(branch_name, "N8192") == 0) {
    logged = &logged_n8192;
  }
  if (*logged) {
    return;
  }
  *logged = true;

  int64_t const num_experts = expert_offsets.size(0);
  bool const inferred_per_act_token = a_scales.numel() != 1;
  bool const inferred_per_out_ch = b_scales.numel() != num_experts;
  int const m_idx = swap_ab ? 1 : 0;

  auto problem_sizes_cpu = problem_sizes.to(torch::kCPU);
  auto const* ps = problem_sizes_cpu.data_ptr<int32_t>();

  int32_t m_min = std::numeric_limits<int32_t>::max();
  int32_t m_max = 0;
  int32_t m_zero_count = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    int32_t m = ps[e * 3 + m_idx];
    m_min = std::min(m_min, m);
    m_max = std::max(m_max, m);
    if (m == 0) {
      ++m_zero_count;
    }
  }

  TORCH_WARN(
      "PPLX_SM100_MOE_MM_CONFIG branch=", branch_name, " swap_ab=", swap_ab,
      " out=(", out_tensors.size(0), ",", out_tensors.size(1), ")", " a=(",
      a_tensors.size(0), ",", a_tensors.size(1), ")", " b=(", b_tensors.size(0),
      ",", b_tensors.size(1), ",", b_tensors.size(2), ")",
      " per_act_token=", per_act_token, " per_out_ch=", per_out_ch,
      " inferred_per_act_token=", inferred_per_act_token,
      " inferred_per_out_ch=", inferred_per_out_ch,
      " a_scales_numel=", a_scales.numel(),
      " b_scales_numel=", b_scales.numel(), " experts=", num_experts,
      " m_min=", m_min, " m_max=", m_max, " m_zero_count=", m_zero_count);
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_config_M64 {
  // M in [1,64]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using TileShape = cute::Shape<cute::_128, cute::_16, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule,
                            true>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_config_N8192 {
  // N in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType>
void run_cutlass_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  using Cutlass3xGemmDefault = typename sm100_fp8_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmN8192 = typename sm100_fp8_config_N8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmM64 = typename sm100_fp8_config_M64<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);

  if (m <= 64) {
    maybe_log_sm100_grouped_moe_config(
        "M64", true, out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        expert_offsets, problem_sizes, per_act_token, per_out_ch);
    cutlass_group_gemm_caller<Cutlass3xGemmM64>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else if (n >= 8192) {
    maybe_log_sm100_grouped_moe_config(
        "N8192", false, out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        expert_offsets, problem_sizes, per_act_token, per_out_ch);
    cutlass_group_gemm_caller<Cutlass3xGemmN8192>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    maybe_log_sm100_grouped_moe_config(
        "DEFAULT", false, out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        expert_offsets, problem_sizes, per_act_token, per_out_ch);
    cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}
}  // namespace

void dispatch_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {
  if (out_tensors.dtype() == torch::kBFloat16) {
    run_cutlass_moe_mm_sm100<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    run_cutlass_moe_mm_sm100<cutlass::float_e4m3_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

void cutlass_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {
  dispatch_moe_mm_sm100(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                        expert_offsets, problem_sizes, a_strides, b_strides,
                        c_strides, per_act_token, per_out_ch);
}
