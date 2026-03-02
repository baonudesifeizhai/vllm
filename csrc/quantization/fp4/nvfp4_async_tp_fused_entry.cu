/*
 * Copyright (c) 2026, The vLLM project authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ops.h"

namespace {

int64_t normalize_dim(int64_t dim, int64_t ndim) {
  return dim >= 0 ? dim : dim + ndim;
}

std::vector<int64_t> compute_balanced_split_sizes(int64_t total,
                                                  int64_t world_size) {
  const int64_t base = total / world_size;
  const int64_t remainder = total % world_size;
  std::vector<int64_t> sizes(world_size, base);
  for (int64_t rank = 0; rank < remainder; ++rank) {
    sizes[rank] += 1;
  }
  return sizes;
}

torch::Tensor call_tensor_op(std::string const& op_name,
                             std::string const& overload,
                             std::vector<c10::IValue> stack) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(op_name, overload);
  op.callBoxed(&stack);
  TORCH_CHECK(stack.size() == 1 && stack[0].isTensor(), op_name,
              " must return a Tensor");
  return std::move(stack[0]).toTensor();
}

std::tuple<torch::Tensor, torch::Tensor> call_qutlass_fused_quantize_nv(
    torch::Tensor const& input, torch::Tensor const& hadamard_matrix,
    torch::Tensor const& output_q, torch::Tensor const& output_scale,
    torch::Tensor const& global_scale) {
  std::vector<c10::IValue> stack;
  stack.emplace_back(input);
  stack.emplace_back(hadamard_matrix);
  stack.emplace_back(output_q);
  stack.emplace_back(output_scale);
  stack.emplace_back(global_scale);

  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_qutlass_C::fusedQuantizeNv", "");
  op.callBoxed(&stack);
  TORCH_CHECK(stack.size() == 1 && stack[0].isTuple(),
              "_qutlass_C::fusedQuantizeNv must return a tuple");
  const auto elems = stack[0].toTuple()->elements();
  TORCH_CHECK(elems.size() == 2,
              "_qutlass_C::fusedQuantizeNv must return 2 tensors");
  return {elems[0].toTensor(), elems[1].toTensor()};
}

torch::Tensor reinterpret_as_fp8_e4m3(torch::Tensor const& input) {
  std::vector<c10::IValue> stack;
  stack.emplace_back(input);
  stack.emplace_back(static_cast<int64_t>(at::ScalarType::Float8_e4m3fn));
  return call_tensor_op("aten::view", "dtype", std::move(stack));
}

torch::Tensor to_blocked_flat(torch::Tensor const& input) {
  TORCH_CHECK(input.dim() == 2, "scale tensor must be 2D");

  auto in = input.contiguous();
  const int64_t rows = in.size(0);
  const int64_t cols = in.size(1);
  const int64_t n_row_blocks = (rows + 127) / 128;
  const int64_t n_col_blocks = (cols + 3) / 4;
  const int64_t padded_rows = n_row_blocks * 128;
  const int64_t padded_cols = n_col_blocks * 4;

  auto padded = in;
  if (padded_rows != rows || padded_cols != cols) {
    padded = torch::zeros({padded_rows, padded_cols}, in.options());
    padded.narrow(0, 0, rows).narrow(1, 0, cols).copy_(in);
  }

  auto blocks =
      padded.view({n_row_blocks, 128, n_col_blocks, 4}).permute({0, 2, 1, 3});
  auto rearranged =
      blocks.reshape({-1, 4, 32, 4}).transpose(1, 2).reshape({-1, 32, 16});
  return rearranged.flatten();
}

torch::Tensor to_blocked_scale_for_fp4_mm(torch::Tensor const& scale,
                                          int64_t k_half) {
  auto blocked = to_blocked_flat(scale);
  if (blocked.scalar_type() != at::ScalarType::Float8_e4m3fn) {
    blocked = reinterpret_as_fp8_e4m3(blocked);
  }
  const int64_t scale_cols = k_half / 8;
  return blocked.view({-1, scale_cols}).contiguous();
}

torch::Tensor call_vllm_all_gather(torch::Tensor const& tensor, int64_t dim,
                                   int64_t world_size,
                                   std::string const& group_name) {
  std::vector<c10::IValue> stack;
  stack.emplace_back(tensor);
  stack.emplace_back(dim);
  stack.emplace_back(world_size);
  stack.emplace_back(group_name);
  return call_tensor_op("vllm::all_gather", "", std::move(stack));
}

torch::Tensor call_vllm_all_reduce(torch::Tensor const& tensor,
                                   std::string const& group_name) {
  std::vector<c10::IValue> stack;
  stack.emplace_back(tensor);
  stack.emplace_back(group_name);
  return call_tensor_op("vllm::all_reduce", "", std::move(stack));
}

torch::Tensor call_vllm_all_gatherv(torch::Tensor const& tensor, int64_t dim,
                                    std::vector<int64_t> const& sizes,
                                    std::string const& group_name) {
  std::vector<c10::IValue> stack;
  stack.emplace_back(tensor);
  stack.emplace_back(dim);
  stack.emplace_back(sizes);
  stack.emplace_back(group_name);
  return call_tensor_op("vllm::all_gatherv", "", std::move(stack));
}

torch::Tensor call_vllm_reduce_scatter(torch::Tensor const& tensor, int64_t dim,
                                       int64_t world_size,
                                       std::string const& group_name) {
  std::vector<c10::IValue> stack;
  stack.emplace_back(tensor);
  stack.emplace_back(dim);
  stack.emplace_back(world_size);
  stack.emplace_back(group_name);
  return call_tensor_op("vllm::reduce_scatter", "", std::move(stack));
}

torch::Tensor call_vllm_reduce_scatterv(torch::Tensor const& tensor,
                                        int64_t dim,
                                        std::vector<int64_t> const& sizes,
                                        std::string const& group_name) {
  std::vector<c10::IValue> stack;
  stack.emplace_back(tensor);
  stack.emplace_back(dim);
  stack.emplace_back(sizes);
  stack.emplace_back(group_name);
  return call_tensor_op("vllm::reduce_scatterv", "", std::move(stack));
}

std::vector<int64_t> gather_dim0_sizes(torch::Tensor const& local_shard,
                                       int64_t world_size,
                                       std::string const& group_name) {
  auto local_tokens = torch::full({1}, local_shard.size(0),
                                  local_shard.options().dtype(torch::kInt64));
  auto total_tokens_tensor = call_vllm_all_reduce(local_tokens, group_name)
                                 .to(torch::kCPU, torch::kInt64);
  const int64_t total_tokens = total_tokens_tensor.item<int64_t>();
  return compute_balanced_split_sizes(total_tokens, world_size);
}

}  // namespace

torch::Tensor fused_nvf4_matmul_reduce_scatter(
    torch::Tensor const& A_q, torch::Tensor const& B_q,
    torch::Tensor const& A_scale, torch::Tensor const& B_scale,
    torch::Tensor const& alpha, const std::string& reduce_op,
    int64_t scatter_dim, int64_t world_size, const std::string& group_name) {
  (void)reduce_op;
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(A_q));

  auto A_scale_blocked = to_blocked_scale_for_fp4_mm(A_scale, A_q.size(1));
  auto B_scale_blocked = to_blocked_scale_for_fp4_mm(B_scale, A_q.size(1));

  auto mm_out = torch::empty({A_q.size(0), B_q.size(0)},
                             A_q.options().dtype(torch::kBFloat16));
  cutlass_scaled_fp4_mm(mm_out, A_q, B_q, A_scale_blocked, B_scale_blocked,
                        alpha);

  const int64_t dim = normalize_dim(scatter_dim, mm_out.dim());
  if (world_size > 1 && dim == 0 && mm_out.size(0) % world_size != 0) {
    auto sizes = compute_balanced_split_sizes(mm_out.size(0), world_size);
    return call_vllm_reduce_scatterv(mm_out, dim, sizes, group_name);
  }
  return call_vllm_reduce_scatter(mm_out, dim, world_size, group_name);
}

torch::Tensor fused_all_gather_quantize_nvf4_matmul(
    torch::Tensor const& A_shard, torch::Tensor const& hadamard_matrix,
    torch::Tensor const& act_global_scale, torch::Tensor const& B_q,
    torch::Tensor const& B_scale, torch::Tensor const& weight_global_scale,
    int64_t gather_dim, int64_t world_size, const std::string& group_name) {
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(A_shard));

  const int64_t dim = normalize_dim(gather_dim, A_shard.dim());
  auto gathered = A_shard;
  if (world_size > 1) {
    if (dim == 0) {
      auto sizes = gather_dim0_sizes(A_shard, world_size, group_name);
      const bool is_ragged =
          !std::all_of(sizes.begin() + 1, sizes.end(),
                       [&](int64_t s) { return s == sizes.front(); });
      gathered =
          is_ragged
              ? call_vllm_all_gatherv(A_shard, dim, sizes, group_name)
              : call_vllm_all_gather(A_shard, dim, world_size, group_name);
    } else {
      gathered = call_vllm_all_gather(A_shard, dim, world_size, group_name);
    }
  }

  const int64_t rows = gathered.size(0);
  const int64_t cols = gathered.size(1) / 16;
  const int64_t padded_rows = ((rows + 127) / 128) * 128;
  const int64_t padded_cols = ((cols + 3) / 4) * 4;

  auto A_q = torch::empty({gathered.size(0), gathered.size(1) / 2},
                          gathered.options().dtype(torch::kUInt8));
  auto A_scale =
      torch::empty({padded_rows, padded_cols},
                   gathered.options().dtype(at::ScalarType::Float8_e4m3fn));

  auto [A_q_out, A_scale_out] = call_qutlass_fused_quantize_nv(
      gathered, hadamard_matrix, A_q, A_scale, act_global_scale);

  auto A_scale_blocked =
      to_blocked_scale_for_fp4_mm(A_scale_out, A_q_out.size(1));
  auto B_scale_blocked = to_blocked_scale_for_fp4_mm(B_scale, A_q_out.size(1));
  auto alpha = torch::reciprocal(weight_global_scale * act_global_scale);

  auto mm_out = torch::empty({A_q_out.size(0), B_q.size(0)},
                             gathered.options().dtype(torch::kBFloat16));
  cutlass_scaled_fp4_mm(mm_out, A_q_out, B_q, A_scale_blocked, B_scale_blocked,
                        alpha);
  return mm_out;
}
