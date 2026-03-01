#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace {

constexpr int64_t kMaxPipelineChunks = 4;
constexpr int64_t kMinChunkRowsPerRank = 64;

int64_t round_down_to_multiple(int64_t value, int64_t multiple) {
  if (multiple <= 1) {
    return value;
  }
  return (value / multiple) * multiple;
}

int64_t select_chunk_rows(int64_t rows, int64_t granularity) {
  if (rows <= 0 || granularity <= 0) {
    return 0;
  }
  int64_t target_chunks =
      std::max<int64_t>(1, rows / (granularity * kMinChunkRowsPerRank));
  target_chunks = std::min<int64_t>(kMaxPipelineChunks, target_chunks);
  if (target_chunks <= 1) {
    return 0;
  }
  int64_t chunk_rows = round_down_to_multiple(
      (rows + target_chunks - 1) / target_chunks, granularity);
  if (chunk_rows <= 0 || chunk_rows >= rows) {
    return 0;
  }
  return chunk_rows;
}

at::Tensor call_tensor_op(const char* op_name, c10::Stack&& stack) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(op_name, "");
  op.callBoxed(&stack);
  TORCH_CHECK(stack.size() == 1, "Expected single return value from ", op_name);
  return std::move(stack[0]).toTensor();
}

at::Tensor call_vllm_bmm_fp8(const at::Tensor& A, const at::Tensor& B,
                             const at::Tensor& A_scale,
                             const at::Tensor& B_scale,
                             c10::ScalarType out_dtype,
                             const std::string& backend) {
  c10::Stack stack;
  stack.emplace_back(A);
  stack.emplace_back(B);
  stack.emplace_back(A_scale);
  stack.emplace_back(B_scale);
  stack.emplace_back(out_dtype);
  stack.emplace_back(backend);
  return call_tensor_op("vllm::bmm_fp8", std::move(stack));
}

at::Tensor call_vllm_reduce_scatter(const at::Tensor& tensor, int64_t dim,
                                    int64_t world_size,
                                    const std::string& group_name) {
  c10::Stack stack;
  stack.emplace_back(tensor);
  stack.emplace_back(dim);
  stack.emplace_back(world_size);
  stack.emplace_back(group_name);
  return call_tensor_op("vllm::reduce_scatter", std::move(stack));
}

at::Tensor call_vllm_all_gather(const at::Tensor& tensor, int64_t dim,
                                int64_t world_size,
                                const std::string& group_name) {
  c10::Stack stack;
  stack.emplace_back(tensor);
  stack.emplace_back(dim);
  stack.emplace_back(world_size);
  stack.emplace_back(group_name);
  return call_tensor_op("vllm::all_gather", std::move(stack));
}

at::Tensor bmm_fp8_reduce_scatter_serial(
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& A_scale,
    const at::Tensor& B_scale, c10::ScalarType out_dtype, int64_t scatter_dim,
    int64_t world_size, const std::string& group_name) {
  auto bmm_out = call_vllm_bmm_fp8(A.unsqueeze(0), B.unsqueeze(0), A_scale,
                                   B_scale, out_dtype, "auto");
  auto mm_out = bmm_out.view({A.size(0), B.size(1)});
  return call_vllm_reduce_scatter(mm_out, scatter_dim, world_size, group_name);
}

at::Tensor all_gather_bmm_fp8_serial(
    const at::Tensor& A_local, const at::Tensor& B, const at::Tensor& A_scale,
    const at::Tensor& B_scale, c10::ScalarType out_dtype, int64_t gather_dim,
    int64_t world_size, const std::string& group_name) {
  auto gathered =
      call_vllm_all_gather(A_local, gather_dim, world_size, group_name);
  auto bmm_out = call_vllm_bmm_fp8(gathered.unsqueeze(0), B.unsqueeze(0),
                                   A_scale, B_scale, out_dtype, "auto");
  return bmm_out.view({gathered.size(0), B.size(1)});
}

bool is_stream_capturing(const c10::cuda::CUDAStream& stream) {
  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &status));
  return status == cudaStreamCaptureStatusActive;
}

class EventRing {
 public:
  EventRing() {
    for (auto& event : events_) {
      AT_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
  }

  ~EventRing() {
    for (auto& event : events_) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
  }

  cudaEvent_t operator[](int idx) const { return events_[idx]; }

 private:
  std::array<cudaEvent_t, 2> events_{};
};

}  // namespace

at::Tensor fused_bmm_fp8_reduce_scatter_overlap(
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& A_scale,
    const at::Tensor& B_scale, c10::ScalarType out_dtype,
    const std::string& reduce_op, int64_t scatter_dim, int64_t world_size,
    const std::string& group_name) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(reduce_op == "sum",
              "Only sum reduce_op is supported, got: ", reduce_op);

  if (world_size <= 1) {
    auto bmm_out = call_vllm_bmm_fp8(A.unsqueeze(0), B.unsqueeze(0), A_scale,
                                     B_scale, out_dtype, "auto");
    return bmm_out.view({A.size(0), B.size(1)});
  }

  // Fast path assumptions for pipelined overlap.
  if (scatter_dim != 0 || A.dim() != 2 || B.dim() != 2 ||
      A.size(0) % world_size != 0) {
    return bmm_fp8_reduce_scatter_serial(A, B, A_scale, B_scale, out_dtype,
                                         scatter_dim, world_size, group_name);
  }

  auto device_index = A.get_device();
  auto caller_stream = at::cuda::getCurrentCUDAStream(device_index);
  if (is_stream_capturing(caller_stream)) {
    // Stream creation and event choreography are not capture-safe.
    return bmm_fp8_reduce_scatter_serial(A, B, A_scale, B_scale, out_dtype,
                                         scatter_dim, world_size, group_name);
  }

  const int64_t rows = A.size(0);
  const int64_t chunk_rows = select_chunk_rows(rows, world_size);
  if (chunk_rows <= 0) {
    return bmm_fp8_reduce_scatter_serial(A, B, A_scale, B_scale, out_dtype,
                                         scatter_dim, world_size, group_name);
  }

  auto compute_stream = caller_stream;
  auto comm_stream = c10::cuda::getStreamFromPool(false, device_index);
  EventRing gemm_done;
  EventRing comm_done;

  std::array<at::Tensor, 2> mm_slots;
  std::vector<at::Tensor> reduced_chunks;
  reduced_chunks.reserve((rows + chunk_rows - 1) / chunk_rows);

  for (int64_t start = 0, chunk_idx = 0; start < rows;
       start += chunk_rows, ++chunk_idx) {
    int slot = static_cast<int>(chunk_idx & 1);
    int64_t len = std::min(chunk_rows, rows - start);
    if (chunk_idx >= 2) {
      AT_CUDA_CHECK(
          cudaStreamWaitEvent(compute_stream.stream(), comm_done[slot], 0));
    }

    auto A_chunk = A.narrow(0, start, len);
    {
      c10::cuda::CUDAStreamGuard guard(compute_stream);
      auto bmm_out = call_vllm_bmm_fp8(A_chunk.unsqueeze(0), B.unsqueeze(0),
                                       A_scale, B_scale, out_dtype, "auto");
      mm_slots[slot] = bmm_out.view({len, B.size(1)});
    }
    AT_CUDA_CHECK(cudaEventRecord(gemm_done[slot], compute_stream.stream()));
    AT_CUDA_CHECK(
        cudaStreamWaitEvent(comm_stream.stream(), gemm_done[slot], 0));
    {
      c10::cuda::CUDAStreamGuard guard(comm_stream);
      reduced_chunks.emplace_back(call_vllm_reduce_scatter(
          mm_slots[slot], scatter_dim, world_size, group_name));
    }
    AT_CUDA_CHECK(cudaEventRecord(comm_done[slot], comm_stream.stream()));
  }

  int used_slots = std::min<int>(2, static_cast<int>(reduced_chunks.size()));
  for (int slot = 0; slot < used_slots; ++slot) {
    AT_CUDA_CHECK(
        cudaStreamWaitEvent(caller_stream.stream(), comm_done[slot], 0));
  }

  return at::cat(reduced_chunks, 0);
}

at::Tensor fused_all_gather_bmm_fp8_overlap(
    const at::Tensor& A_local, const at::Tensor& B, const at::Tensor& A_scale,
    const at::Tensor& B_scale, c10::ScalarType out_dtype, int64_t gather_dim,
    int64_t world_size, const std::string& group_name) {
  TORCH_CHECK(A_local.is_cuda() && B.is_cuda(),
              "A_local and B must be CUDA tensors");

  if (world_size <= 1) {
    auto bmm_out = call_vllm_bmm_fp8(A_local.unsqueeze(0), B.unsqueeze(0),
                                     A_scale, B_scale, out_dtype, "auto");
    return bmm_out.view({A_local.size(0), B.size(1)});
  }

  // Fast path assumptions for pipelined overlap.
  if (gather_dim != 0 || A_local.dim() != 2 || B.dim() != 2) {
    return all_gather_bmm_fp8_serial(A_local, B, A_scale, B_scale, out_dtype,
                                     gather_dim, world_size, group_name);
  }

  auto device_index = A_local.get_device();
  auto caller_stream = at::cuda::getCurrentCUDAStream(device_index);
  if (is_stream_capturing(caller_stream)) {
    return all_gather_bmm_fp8_serial(A_local, B, A_scale, B_scale, out_dtype,
                                     gather_dim, world_size, group_name);
  }

  const int64_t rows_local = A_local.size(0);
  const int64_t chunk_rows = select_chunk_rows(rows_local, 1);
  if (chunk_rows <= 0) {
    return all_gather_bmm_fp8_serial(A_local, B, A_scale, B_scale, out_dtype,
                                     gather_dim, world_size, group_name);
  }

  auto compute_stream = caller_stream;
  auto comm_stream = c10::cuda::getStreamFromPool(false, device_index);
  EventRing gather_done;
  EventRing mm_done;

  auto output = at::empty({rows_local * world_size, B.size(1)},
                          A_local.options().dtype(out_dtype));

  std::array<at::Tensor, 2> gathered_slots;
  std::array<at::Tensor, 2> mm_slots;

  for (int64_t start = 0, chunk_idx = 0; start < rows_local;
       start += chunk_rows, ++chunk_idx) {
    int slot = static_cast<int>(chunk_idx & 1);
    int64_t len = std::min(chunk_rows, rows_local - start);
    if (chunk_idx >= 2) {
      AT_CUDA_CHECK(
          cudaStreamWaitEvent(comm_stream.stream(), mm_done[slot], 0));
    }

    auto A_chunk = A_local.narrow(0, start, len);
    {
      c10::cuda::CUDAStreamGuard guard(comm_stream);
      gathered_slots[slot] =
          call_vllm_all_gather(A_chunk, gather_dim, world_size, group_name);
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[slot], comm_stream.stream()));
    AT_CUDA_CHECK(
        cudaStreamWaitEvent(compute_stream.stream(), gather_done[slot], 0));
    {
      c10::cuda::CUDAStreamGuard guard(compute_stream);
      auto bmm_out =
          call_vllm_bmm_fp8(gathered_slots[slot].unsqueeze(0), B.unsqueeze(0),
                            A_scale, B_scale, out_dtype, "auto");
      mm_slots[slot] = bmm_out.view({gathered_slots[slot].size(0), B.size(1)});
      for (int64_t rank = 0; rank < world_size; ++rank) {
        auto src = mm_slots[slot].narrow(0, rank * len, len);
        auto dst = output.narrow(0, rank * rows_local + start, len);
        dst.copy_(src);
      }
    }
    AT_CUDA_CHECK(cudaEventRecord(mm_done[slot], compute_stream.stream()));
  }

  int used_slots = std::min<int>(
      2, static_cast<int>((rows_local + chunk_rows - 1) / chunk_rows));
  for (int slot = 0; slot < used_slots; ++slot) {
    AT_CUDA_CHECK(
        cudaStreamWaitEvent(caller_stream.stream(), mm_done[slot], 0));
  }

  return output;
}
