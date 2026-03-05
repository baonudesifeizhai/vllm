#include "tp_fused_fp8_common.cuh"

namespace vllm {
namespace tp_fused_fp8 {

template <typename OutT, int ngpus>
__global__ void __launch_bounds__(kThreads, 1)
    tp_fused_ag_bmm_sm100_kernel(RankData* rank_data, RankSignals sg,
                                 Signal* self_sg, const Fp8* b, OutT* out,
                                 int m_local, int n, int k, float scale,
                                 int rank) {
  int total_rows = m_local * ngpus;
  int row_tiles = ceil_div(total_rows, kTileRows);
  int col_tiles = ceil_div(n, kTileCols);
  int total_tiles = row_tiles * col_tiles;
  FlagType seq = self_sg->_flag[blockIdx.x] + 1;

  tp_signal_ready<ngpus>(sg, rank, 0, seq);
  tp_wait_ready<ngpus>(self_sg, 0, seq);

  const Fp8* peer_a[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i) {
    peer_a[i] = reinterpret_cast<const Fp8*>(rank_data->ptrs[i]);
  }

  for (int tile_index = blockIdx.x; tile_index < total_tiles;
       tile_index += gridDim.x) {
    int row_tile = tile_index / col_tiles;
    int col_tile = tile_index % col_tiles;
    int row_start = row_tile * kTileRows;
    int col_start = col_tile * kTileCols;
    int tile_rows = total_rows - row_start;
    int tile_cols = n - col_start;
    if (tile_rows > kTileRows) {
      tile_rows = kTileRows;
    }
    if (tile_cols > kTileCols) {
      tile_cols = kTileCols;
    }

    int tile_elems = kTileRows * kTileCols;
    for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x) {
      int tile_row = linear / kTileCols;
      int tile_col = linear % kTileCols;
      if (tile_row >= tile_rows || tile_col >= tile_cols) {
        continue;
      }
      int global_row = row_start + tile_row;
      int src_rank = global_row / m_local;
      int local_row = global_row - src_rank * m_local;
      int global_col = col_start + tile_col;
      float acc = 0.0f;
      for (int k_base = 0; k_base < k; k_base += kKChunk) {
        int k_end = k_base + kKChunk;
        if (k_end > k) {
          k_end = k;
        }
        for (int kk = k_base; kk < k_end; ++kk) {
          float a_val = to_float(peer_a[src_rank][local_row * k + kk]);
          float b_val = to_float(b[global_col * k + kk]);
          acc += a_val * b_val;
        }
      }
      out[global_row * n + global_col] = from_float<OutT>(acc * scale);
    }
  }

  if (threadIdx.x == 0) {
    self_sg->_flag[blockIdx.x] = seq;
  }
}

template <typename OutT>
void launch_tp_fused_all_gather_bmm_fp8_sm100(const CommLaunchInfo& info,
                                              const Fp8* a_shard, const Fp8* b,
                                              OutT* out, int m_local, int n,
                                              int k, float scale, int blocks,
                                              cudaStream_t stream) {
  (void)a_shard;
#define TP_AG_CASE(ngpus)                                                    \
  case ngpus:                                                                \
    tp_fused_ag_bmm_sm100_kernel<OutT, ngpus>                                \
        <<<blocks, kThreads, 0, stream>>>(info.rank_data, info.rank_signals, \
                                          info.self_signal, b, out, m_local, \
                                          n, k, scale, info.rank);           \
    break
  switch (info.world_size) {
    TP_AG_CASE(2);
    TP_AG_CASE(4);
    TP_AG_CASE(6);
    TP_AG_CASE(8);
    default:
      throw std::runtime_error(
          "tp_fused_all_gather_bmm_fp8 only supports world_size in (2,4,6,8)");
  }
#undef TP_AG_CASE
}

template void launch_tp_fused_all_gather_bmm_fp8_sm100<half>(
    const CommLaunchInfo&, const Fp8*, const Fp8*, half*, int, int, int, float,
    int, cudaStream_t);
template void launch_tp_fused_all_gather_bmm_fp8_sm100<nv_bfloat16>(
    const CommLaunchInfo&, const Fp8*, const Fp8*, nv_bfloat16*, int, int, int,
    float, int, cudaStream_t);

}  // namespace tp_fused_fp8
}  // namespace vllm
