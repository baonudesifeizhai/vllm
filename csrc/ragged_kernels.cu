#include <torch/all.h>

#include <c10/cuda/CUDAGuard.h>

// Compute out[out_row_start:out_row_start + a.size(0), :] = a @ b
// This op is intentionally simple and stream-friendly so upper-level
// communication/computation pipelining can invoke it per ragged chunk.
void ragged_mm_slice(torch::Tensor& out, torch::Tensor const& a,
                     torch::Tensor const& b, int64_t out_row_start) {
  TORCH_CHECK(out.is_cuda() && a.is_cuda() && b.is_cuda(),
              "ragged_mm_slice expects CUDA tensors");
  TORCH_CHECK(out.dim() == 2 && a.dim() == 2 && b.dim() == 2,
              "ragged_mm_slice expects 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "a.size(1) must equal b.size(0)");
  TORCH_CHECK(out.size(1) == b.size(1), "out.size(1) must equal b.size(1)");
  TORCH_CHECK(out_row_start >= 0, "out_row_start must be non-negative");
  TORCH_CHECK(out_row_start + a.size(0) <= out.size(0),
              "output slice is out of bounds");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));

  auto out_slice = out.narrow(0, out_row_start, a.size(0));
  auto mm = at::matmul(a, b);
  out_slice.copy_(mm);
}
