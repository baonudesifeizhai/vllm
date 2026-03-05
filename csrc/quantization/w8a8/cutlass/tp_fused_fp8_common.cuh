#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "cutlass/float8.h"
#include "custom_all_reduce.cuh"

namespace vllm {
namespace tp_fused_fp8 {

using Fp8 = cutlass::float_e4m3_t;

constexpr int kTileRows = 64;
constexpr int kTileCols = 64;
constexpr int kKChunk = 128;
constexpr int kThreads = 256;
constexpr int kPipelineSlots = kMaxTpSlots;

inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

template <typename T>
DINLINE float to_float(T val);

template <>
DINLINE float to_float<float>(float val) {
  return val;
}

template <>
DINLINE float to_float<half>(half val) {
  return __half2float(val);
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
DINLINE float to_float<nv_bfloat16>(nv_bfloat16 val) {
  return __bfloat162float(val);
}
#endif

template <>
DINLINE float to_float<Fp8>(Fp8 val) {
  return static_cast<float>(val);
}

template <typename T>
DINLINE T from_float(float val);

template <>
DINLINE float from_float<float>(float val) {
  return val;
}

template <>
DINLINE half from_float<half>(float val) {
  return __float2half_rn(val);
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
DINLINE nv_bfloat16 from_float<nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}
#endif

template <typename OutT>
constexpr int64_t mailbox_workspace_bytes(int blocks) {
  return static_cast<int64_t>(blocks) * kPipelineSlots * kTileRows * kTileCols *
         sizeof(OutT);
}

template <typename OutT>
DINLINE OutT* mailbox_slot_ptr(RankData const* rank_data, int producer_rank,
                               int block, int slot) {
  auto* base = reinterpret_cast<OutT*>(
      const_cast<void*>(rank_data->ptrs[producer_rank]));
  auto slot_elems =
      static_cast<int64_t>(kTileRows) * static_cast<int64_t>(kTileCols);
  return base +
         ((static_cast<int64_t>(block) * kPipelineSlots + slot) * slot_elems);
}

template <typename OutT>
DINLINE void compute_bmm_tile_partial(OutT* slot_ptr, const Fp8* a,
                                      const Fp8* b, int row_start,
                                      int col_start, int tile_rows,
                                      int tile_cols, int k, float scale) {
  auto tile_elems = kTileRows * kTileCols;
  for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x) {
    int tile_row = linear / kTileCols;
    int tile_col = linear % kTileCols;
    float acc = 0.0f;
    if (tile_row < tile_rows && tile_col < tile_cols) {
      int global_row = row_start + tile_row;
      int global_col = col_start + tile_col;
      for (int k_base = 0; k_base < k; k_base += kKChunk) {
        int k_end = k_base + kKChunk;
        if (k_end > k) {
          k_end = k;
        }
        for (int kk = k_base; kk < k_end; ++kk) {
          float a_val = to_float(a[global_row * k + kk]);
          float b_val = to_float(b[global_col * k + kk]);
          acc += a_val * b_val;
        }
      }
    }
    slot_ptr[linear] = from_float<OutT>(acc * scale);
  }
}

template <typename OutT>
DINLINE void reduce_mailbox_tile(OutT* out, RankData const* rank_data, int slot,
                                 int block, int row_start, int col_start,
                                 int tile_rows, int tile_cols, int shard_rows,
                                 int n, int owner_rank, int world_size) {
  auto tile_elems = kTileRows * kTileCols;
  for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x) {
    int tile_row = linear / kTileCols;
    int tile_col = linear % kTileCols;
    if (tile_row >= tile_rows || tile_col >= tile_cols) {
      continue;
    }
    float sum = 0.0f;
#pragma unroll
    for (int peer = 0; peer < 8; ++peer) {
      if (peer >= world_size) {
        break;
      }
      const OutT* peer_slot =
          mailbox_slot_ptr<OutT>(rank_data, peer, block, slot);
      sum += to_float(peer_slot[linear]);
    }
    int shard_row = row_start - owner_rank * shard_rows + tile_row;
    out[shard_row * n + col_start + tile_col] = from_float<OutT>(sum);
  }
}

template <typename OutT>
void launch_tp_fused_all_gather_bmm_fp8_sm100(const CommLaunchInfo& info,
                                              const Fp8* a_shard, const Fp8* b,
                                              OutT* out, int m_local, int n,
                                              int k, float scale, int blocks,
                                              cudaStream_t stream);

template <typename OutT>
void launch_tp_fused_bmm_reduce_scatter_sm100(const CommLaunchInfo& info,
                                              const Fp8* a, const Fp8* b,
                                              OutT* out, int m, int n, int k,
                                              float scale, int blocks,
                                              cudaStream_t stream);

}  // namespace tp_fused_fp8
}  // namespace vllm
