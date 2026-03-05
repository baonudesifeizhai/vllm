#include "tp_fused_fp8_common.cuh"

namespace vllm {
namespace tp_fused_fp8 {

template <typename OutT, int ngpus>
__global__ void __launch_bounds__(kThreads, 1)
    tp_fused_bmm_rs_sm100_kernel(RankData* rank_data, RankSignals sg,
                                 Signal* self_sg, const Fp8* a, const Fp8* b,
                                 OutT* out, int m, int n, int k, float scale,
                                 int rank) {
  constexpr int slots = kPipelineSlots;
  int shard_rows = m / ngpus;
  int row_tiles = ceil_div(m, kTileRows);
  int col_tiles = ceil_div(n, kTileCols);
  int total_tiles = row_tiles * col_tiles;
  int iter = 0;
  FlagType base_seq = self_sg->_flag[blockIdx.x] + 1;

  for (int tile_index = blockIdx.x; tile_index < total_tiles;
       tile_index += gridDim.x, ++iter) {
    int slot = iter % slots;
    FlagType seq = base_seq + static_cast<FlagType>(iter / slots);
    if (iter >= slots) {
      int prev_tile = tile_index - gridDim.x * slots;
      int prev_row_tile = prev_tile / col_tiles;
      int prev_row_start = prev_row_tile * kTileRows;
      int prev_owner = prev_row_start / shard_rows;
      if (prev_owner >= ngpus) {
        prev_owner = ngpus - 1;
      }
      tp_wait_consumed(self_sg, slot, prev_owner, seq - 1);
    }

    int row_tile = tile_index / col_tiles;
    int col_tile = tile_index % col_tiles;
    int row_start = row_tile * kTileRows;
    int col_start = col_tile * kTileCols;
    int tile_rows = m - row_start;
    int tile_cols = n - col_start;
    if (tile_rows > kTileRows) {
      tile_rows = kTileRows;
    }
    if (tile_cols > kTileCols) {
      tile_cols = kTileCols;
    }
    int owner_rank = row_start / shard_rows;
    if (owner_rank >= ngpus) {
      owner_rank = ngpus - 1;
    }

    OutT* local_slot =
        mailbox_slot_ptr<OutT>(rank_data, rank, blockIdx.x, slot);
    compute_bmm_tile_partial(local_slot, a, b, row_start, col_start, tile_rows,
                             tile_cols, k, scale);
    __syncthreads();

    tp_signal_ready<ngpus>(sg, rank, slot, seq);

    if (rank == owner_rank) {
      tp_wait_ready<ngpus>(self_sg, slot, seq);
      reduce_mailbox_tile(out, rank_data, slot, blockIdx.x, row_start,
                          col_start, tile_rows, tile_cols, shard_rows, n,
                          owner_rank, ngpus);
      __syncthreads();
      tp_signal_consumed<ngpus>(sg, rank, slot, seq);
    }
  }

  int flush_iters = iter < slots ? iter : slots;
  for (int pending = iter - flush_iters; pending < iter; ++pending) {
    int tile_index = blockIdx.x + pending * gridDim.x;
    if (tile_index >= total_tiles) {
      continue;
    }
    int slot = pending % slots;
    FlagType seq = base_seq + static_cast<FlagType>(pending / slots);
    int row_tile = tile_index / col_tiles;
    int owner_rank = (row_tile * kTileRows) / shard_rows;
    if (owner_rank >= ngpus) {
      owner_rank = ngpus - 1;
    }
    tp_wait_consumed(self_sg, slot, owner_rank, seq);
  }

  if (threadIdx.x == 0 && iter > 0) {
    self_sg->_flag[blockIdx.x] =
        base_seq + static_cast<FlagType>((iter - 1) / slots);
  }
}

template <typename OutT>
void launch_tp_fused_bmm_reduce_scatter_sm100(const CommLaunchInfo& info,
                                              const Fp8* a, const Fp8* b,
                                              OutT* out, int m, int n, int k,
                                              float scale, int blocks,
                                              cudaStream_t stream) {
#define TP_RS_CASE(ngpus)                                                    \
  case ngpus:                                                                \
    tp_fused_bmm_rs_sm100_kernel<OutT, ngpus>                                \
        <<<blocks, kThreads, 0, stream>>>(info.rank_data, info.rank_signals, \
                                          info.self_signal, a, b, out, m, n, \
                                          k, scale, info.rank);              \
    break
  switch (info.world_size) {
    TP_RS_CASE(2);
    TP_RS_CASE(4);
    TP_RS_CASE(6);
    TP_RS_CASE(8);
    default:
      throw std::runtime_error(
          "tp_fused_bmm_fp8_reduce_scatter only supports world_size in "
          "(2,4,6,8)");
  }
#undef TP_RS_CASE
}

template void launch_tp_fused_bmm_reduce_scatter_sm100<half>(
    const CommLaunchInfo&, const Fp8*, const Fp8*, half*, int, int, int, float,
    int, cudaStream_t);
template void launch_tp_fused_bmm_reduce_scatter_sm100<nv_bfloat16>(
    const CommLaunchInfo&, const Fp8*, const Fp8*, nv_bfloat16*, int, int, int,
    float, int, cudaStream_t);

}  // namespace tp_fused_fp8
}  // namespace vllm
