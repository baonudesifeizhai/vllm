# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton helpers for Gemma4 MTP."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gemma4_mtp_q_norm_rope_kernel(
    q_ptr,
    positions_ptr,
    weight_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    eps,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_DIM
    half = HEAD_DIM // 2
    first_half = offs < half
    pair_offs = tl.where(first_half, offs + half, offs - half)
    rope_offs = tl.where(first_half, offs, offs - half)

    q_base = (token_idx * NUM_HEADS + head_idx) * HEAD_DIM
    q = tl.load(q_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    q_pair = tl.load(q_ptr + q_base + pair_offs, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(q * q, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(weight_ptr + pair_offs, mask=mask, other=0.0).to(tl.float32)
    q = q * rrms * weight
    q_pair = q_pair * rrms * pair_weight

    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    rope_base = pos * HEAD_DIM
    cos = tl.load(cos_sin_cache_ptr + rope_base + rope_offs, mask=mask, other=1.0)
    sin = tl.load(
        cos_sin_cache_ptr + rope_base + half + rope_offs,
        mask=mask,
        other=0.0,
    )
    cos = cos.to(tl.float32)
    sin = sin.to(tl.float32)

    out = tl.where(first_half, q * cos - q_pair * sin, q * cos + q_pair * sin)
    tl.store(out_ptr + q_base + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def gemma4_mtp_q_norm_rope(
    q: torch.Tensor,
    positions: torch.Tensor,
    weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fuse per-head RMSNorm and neox RoPE for Gemma4 MTP Q projections."""
    assert q.ndim == 2
    assert q.shape[1] == num_heads * head_dim
    assert positions.ndim == 1 and positions.shape[0] == q.shape[0]
    assert weight.ndim == 1 and weight.shape[0] == head_dim
    assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == head_dim
    assert head_dim % 2 == 0
    assert q.is_contiguous()
    assert positions.is_contiguous()
    assert weight.is_contiguous()
    assert cos_sin_cache.is_contiguous()

    out = torch.empty_like(q)
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return out

    block_size = triton.next_power_of_2(head_dim)
    _gemma4_mtp_q_norm_rope_kernel[(num_tokens, num_heads)](
        q,
        positions,
        weight,
        cos_sin_cache,
        out,
        eps,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
    )
    return out
