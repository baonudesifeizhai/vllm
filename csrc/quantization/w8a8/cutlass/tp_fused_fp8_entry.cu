#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <type_traits>

#include "custom_all_reduce.cuh"
#include "tp_fused_fp8_common.cuh"

namespace vllm {

using fptr_t = int64_t;
using tp_fused_fp8::Fp8;

namespace {

void check_fp8_inputs(torch::Tensor const& a, torch::Tensor const& b,
                      torch::Tensor const& scale_a,
                      torch::Tensor const& scale_b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tp fused fp8 inputs must be CUDA");
  TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(),
              "tp fused fp8 scales must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn,
              "expected fp8 e4m3 activations");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn,
              "expected fp8 e4m3 weights");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "tp fused fp8 expects 2D inputs");
  TORCH_CHECK(a.stride(1) == 1, "A must be row-major");
  TORCH_CHECK(b.stride(0) == 1, "B must be column-major");
  TORCH_CHECK(scale_a.numel() == 1 && scale_b.numel() == 1,
              "tp fused fp8 currently only supports per-tensor scales");
}

at::ScalarType check_out_dtype(at::ScalarType dtype) {
  TORCH_CHECK(
      dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16,
      "tp fused fp8 only supports fp16/bf16 outputs");
  return dtype;
}

int choose_ag_blocks(int total_rows, int n) {
  int total_tiles =
      tp_fused_fp8::ceil_div(total_rows, tp_fused_fp8::kTileRows) *
      tp_fused_fp8::ceil_div(n, tp_fused_fp8::kTileCols);
  int blocks = total_tiles < kMaxBlocks ? total_tiles : kMaxBlocks;
  return blocks > 0 ? blocks : 1;
}

template <typename OutT>
int choose_rs_blocks(int m, int n, int64_t reg_buffer_sz_bytes) {
  int total_tiles = tp_fused_fp8::ceil_div(m, tp_fused_fp8::kTileRows) *
                    tp_fused_fp8::ceil_div(n, tp_fused_fp8::kTileCols);
  int64_t bytes_per_block = tp_fused_fp8::mailbox_workspace_bytes<OutT>(1);
  TORCH_CHECK(bytes_per_block > 0, "invalid mailbox workspace size");
  int max_blocks_by_workspace =
      static_cast<int>(reg_buffer_sz_bytes / bytes_per_block);
  TORCH_CHECK(max_blocks_by_workspace > 0,
              "tp fused bmm+rs workspace too small: need at least ",
              bytes_per_block, " bytes");
  int blocks = total_tiles;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  if (blocks > max_blocks_by_workspace) {
    blocks = max_blocks_by_workspace;
  }
  return blocks > 0 ? blocks : 1;
}

float get_scale(torch::Tensor const& scale_a, torch::Tensor const& scale_b) {
  return scale_a.item<float>() * scale_b.item<float>();
}

void* prepare_registered_input(torch::Tensor const& input, fptr_t reg_buffer,
                               int64_t reg_buffer_sz_bytes,
                               cudaStream_t stream) {
  void* shared_ptr = reinterpret_cast<void*>(reg_buffer);
  if (shared_ptr == nullptr) {
    return input.data_ptr();
  }
  auto input_bytes = input.numel() * input.element_size();
  TORCH_CHECK(input_bytes <= reg_buffer_sz_bytes,
              "registered buffer is smaller than input shard");
  AT_CUDA_CHECK(cudaMemcpyAsync(shared_ptr, input.data_ptr(), input_bytes,
                                cudaMemcpyDeviceToDevice, stream));
  return shared_ptr;
}

template <typename OutT>
torch::Tensor launch_tp_fused_all_gather_bmm_impl(
    CustomAllreduce* fa, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b,
    fptr_t reg_buffer, int64_t reg_buffer_sz_bytes) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  void* shard_ptr =
      prepare_registered_input(a, reg_buffer, reg_buffer_sz_bytes, stream);
  auto info = fa->resolve_comm_launch(stream, shard_ptr);
  int m_local = static_cast<int>(a.size(0));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(a.size(1));
  int total_rows = m_local * info.world_size;
  auto out_dtype = std::is_same_v<OutT, half> ? at::ScalarType::Half
                                              : at::ScalarType::BFloat16;
  auto out = torch::empty({total_rows, n}, a.options().dtype(out_dtype));
  auto blocks = choose_ag_blocks(total_rows, n);
  tp_fused_fp8::launch_tp_fused_all_gather_bmm_fp8_sm100<OutT>(
      info, reinterpret_cast<const Fp8*>(shard_ptr),
      reinterpret_cast<const Fp8*>(b.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), m_local, n, k,
      get_scale(scale_a, scale_b), blocks, stream);
  return out;
}

template <typename OutT>
torch::Tensor launch_tp_fused_bmm_rs_impl(
    CustomAllreduce* fa, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b,
    fptr_t reg_buffer, int64_t reg_buffer_sz_bytes) {
  TORCH_CHECK(
      reg_buffer != 0,
      "tp_fused_bmm_fp8_reduce_scatter requires a registered workspace");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto info =
      fa->resolve_comm_launch(stream, reinterpret_cast<void*>(reg_buffer));
  int m = static_cast<int>(a.size(0));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(a.size(1));
  TORCH_CHECK(m >= info.world_size, "tp fused bmm+rs requires M >= world size");
  TORCH_CHECK(m % info.world_size == 0,
              "tp fused bmm+rs requires M divisible by world size");
  auto out_dtype = std::is_same_v<OutT, half> ? at::ScalarType::Half
                                              : at::ScalarType::BFloat16;
  auto out =
      torch::empty({m / info.world_size, n}, a.options().dtype(out_dtype));
  auto blocks = choose_rs_blocks<OutT>(m, n, reg_buffer_sz_bytes);
  tp_fused_fp8::launch_tp_fused_bmm_reduce_scatter_sm100<OutT>(
      info, reinterpret_cast<const Fp8*>(a.data_ptr()),
      reinterpret_cast<const Fp8*>(b.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), m, n, k,
      get_scale(scale_a, scale_b), blocks, stream);
  return out;
}

}  // namespace

torch::Tensor tp_fused_all_gather_bmm_fp8(torch::Tensor const& a,
                                          torch::Tensor const& b,
                                          torch::Tensor const& scale_a,
                                          torch::Tensor const& scale_b,
                                          at::ScalarType out_dtype, fptr_t _fa,
                                          fptr_t reg_buffer,
                                          int64_t reg_buffer_sz_bytes) {
  check_fp8_inputs(a, b, scale_a, scale_b);
  check_out_dtype(out_dtype);
  auto* fa = reinterpret_cast<CustomAllreduce*>(_fa);
  switch (out_dtype) {
    case at::ScalarType::Half:
      return launch_tp_fused_all_gather_bmm_impl<half>(
          fa, a, b, scale_a, scale_b, reg_buffer, reg_buffer_sz_bytes);
    case at::ScalarType::BFloat16:
      return launch_tp_fused_all_gather_bmm_impl<nv_bfloat16>(
          fa, a, b, scale_a, scale_b, reg_buffer, reg_buffer_sz_bytes);
    default:
      TORCH_CHECK(false, "unsupported out dtype for tp fused ag+bmm");
  }
  TORCH_CHECK(false, "unreachable tp fused ag+bmm dispatch");
}

torch::Tensor tp_fused_bmm_fp8_reduce_scatter(torch::Tensor const& a,
                                              torch::Tensor const& b,
                                              torch::Tensor const& scale_a,
                                              torch::Tensor const& scale_b,
                                              at::ScalarType out_dtype,
                                              fptr_t _fa, fptr_t reg_buffer,
                                              int64_t reg_buffer_sz_bytes) {
  check_fp8_inputs(a, b, scale_a, scale_b);
  check_out_dtype(out_dtype);
  auto* fa = reinterpret_cast<CustomAllreduce*>(_fa);
  switch (out_dtype) {
    case at::ScalarType::Half:
      return launch_tp_fused_bmm_rs_impl<half>(fa, a, b, scale_a, scale_b,
                                               reg_buffer, reg_buffer_sz_bytes);
    case at::ScalarType::BFloat16:
      return launch_tp_fused_bmm_rs_impl<nv_bfloat16>(
          fa, a, b, scale_a, scale_b, reg_buffer, reg_buffer_sz_bytes);
    default:
      TORCH_CHECK(false, "unsupported out dtype for tp fused bmm+rs");
  }
  TORCH_CHECK(false, "unreachable tp fused bmm+rs dispatch");
}

torch::Tensor fused_all_gather_bmm_fp8(torch::Tensor const& a,
                                       torch::Tensor const& b,
                                       torch::Tensor const& scale_a,
                                       torch::Tensor const& scale_b,
                                       at::ScalarType out_dtype, fptr_t _fa,
                                       fptr_t reg_buffer,
                                       int64_t reg_buffer_sz_bytes) {
  return tp_fused_all_gather_bmm_fp8(a, b, scale_a, scale_b, out_dtype, _fa,
                                     reg_buffer, reg_buffer_sz_bytes);
}

torch::Tensor fused_bmm_fp8_reduce_scatter(torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           torch::Tensor const& scale_a,
                                           torch::Tensor const& scale_b,
                                           at::ScalarType out_dtype, fptr_t _fa,
                                           fptr_t reg_buffer,
                                           int64_t reg_buffer_sz_bytes) {
  return tp_fused_bmm_fp8_reduce_scatter(a, b, scale_a, scale_b, out_dtype, _fa,
                                         reg_buffer, reg_buffer_sz_bytes);
}

}  // namespace vllm
