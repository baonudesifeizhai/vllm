import torch
import torch.nn.functional as F

def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int) -> torch.Tensor:
    difference = (multiple - (x.shape[dim] % multiple)) % multiple
    if difference == 0:
        return x
    dim = dim if dim >= 0 else x.ndim + dim
    pad_list = []
    for i in range(x.ndim - 1, dim - 1, -1):
        if i == dim:
            pad_list.extend([0, difference])
        else:
            pad_list.extend([0, 0])
    return F.pad(x, pad_list, mode="constant", value=0)


def simulate_direct_build(
    *,
    num_reqs: int,
    seq_len: int,
    block_size: int,
    q_block_size: int,
    kv_block_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_cache_tokens = seq_len
    num_blocks = (seq_len + block_size - 1) // block_size

    block_table = torch.arange(num_blocks, device=device)
    block_table = block_table.repeat(num_reqs, 1)

    num_blocks_per_seq = torch.full((num_reqs,), num_blocks, dtype=torch.int32)

    page_to_block_ratio = kv_block_size // block_size
    assert page_to_block_ratio == 1, "This demo assumes page_size == block_size"

    kv_indices_all = []
    kv_num_blocks_all = []

    for req_idx in range(num_reqs):
        seq_blocks = block_table[req_idx, : num_blocks_per_seq[req_idx]]
        seq_used_pages = seq_blocks.unsqueeze(0).repeat(seq_len, 1)
        seq_used_pages_padded = pad_to_multiple(
            seq_used_pages, multiple=q_block_size, dim=0
        )
        num_q_blocks = seq_used_pages_padded.shape[0] // q_block_size
        seq_used_pages_reshaped = seq_used_pages_padded.reshape(num_q_blocks, -1)
        seq_used_pages_reshaped = seq_used_pages_reshaped // page_to_block_ratio

        max_blocks_per_seq = int(num_blocks_per_seq.max().item())
        kv_indices = torch.full(
            (num_q_blocks, max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=device,
        )
        kv_num_blocks = torch.zeros(num_q_blocks, dtype=torch.int32, device=device)

        for q_block_idx in range(num_q_blocks):
            row = seq_used_pages_reshaped[q_block_idx]
            valid = row[row >= 0]
            if valid.numel() == 0:
                continue
            unique_vals = torch.unique(valid, sorted=False)
            if unique_vals.numel() == 0:
                continue
            count = min(unique_vals.numel(), max_blocks_per_seq)
            kv_indices[q_block_idx, :count] = unique_vals[:count].to(torch.int32)
            kv_num_blocks[q_block_idx] = count

        kv_indices_all.append(kv_indices)
        kv_num_blocks_all.append(kv_num_blocks)

    kv_indices_cat = torch.cat(kv_indices_all, dim=0)
    kv_num_blocks_cat = torch.cat(kv_num_blocks_all, dim=0)

    print("=== direct build simulation ===")
    print(f"num_reqs: {num_reqs}")
    print(f"seq_len: {seq_len}")
    print(f"block_size: {block_size}")
    print(f"total_cache_tokens: {total_cache_tokens}")
    print(f"max_blocks_per_seq: {int(num_blocks_per_seq.max().item())}")
    print(f"kv_indices shape: {tuple(kv_indices_cat.shape)}")
    print(f"kv_num_blocks max: {int(kv_num_blocks_cat.max().item())}")


def main():
    num_reqs = 4
    block_size = 16
    q_block_size = 16
    kv_block_size = 16
    total_cache_tokens = 2_097_152

    simulate_direct_build(
        num_reqs=num_reqs,
        seq_len=total_cache_tokens,
        block_size=block_size,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )


if __name__ == "__main__":
    main()
