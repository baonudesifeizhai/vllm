# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end benchmark for TRTLLM FP4 MoE quantization paths (issue #32057).

Following reviewer's suggestion: benchmark from an e2e perspective, comparing:
1. FlashInfer with swizzled layout (is_sf_swizzled_layout=True) - recommended approach
2. vLLM native quantization + convert_swizzled_to_linear - current implementation

This benchmark measures the COMPLETE MoE forward pass, including:
- Input quantization
- Router/expert selection
- Full MoE kernel execution (GEMM operations)
"""

import argparse
from typing import NamedTuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    convert_swizzled_to_linear,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton
from vllm.utils.flashinfer import flashinfer_fp4_quantize

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


class MockMoELayer(NamedTuple):
    """Mock MoE layer for benchmarking."""

    w13_weight: torch.Tensor
    # FP4 quantized weights [num_experts, 2*intermediate_size, hidden_size//2]
    w13_weight_scale: torch.Tensor
    # FP8 scales [num_experts, 2*intermediate_size, hidden_size//16]
    w2_weight: torch.Tensor
    # FP4 quantized weights [num_experts, hidden_size, intermediate_size//2]
    w2_weight_scale: torch.Tensor
    # FP8 scales [num_experts, hidden_size, intermediate_size//16]
    a1_gscale: torch.Tensor  # Global scale for input quantization
    g1_scale_c: torch.Tensor  # Output scale scalar
    g1_alphas: torch.Tensor  # Gate scale scalars
    g2_alphas: torch.Tensor  # Output2 scale scalars
    intermediate_size_per_partition: int
    local_num_experts: int
    ep_rank: int
    routing_method_type: int


def compute_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute global scale for FP4 quantization."""
    amax = torch.abs(tensor).max().to(torch.float32)
    return FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax


def create_mock_moe_layer(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> MockMoELayer:
    """Create a mock MoE layer with quantized weights for benchmarking."""
    # Create random weights
    w13 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=dtype,
        )
        / 10
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size), device=device, dtype=dtype
        )
        / 10
    )

    # Quantize weights to FP4
    w13_fp4_list = []
    w13_scale_list = []
    w2_fp4_list = []
    w2_scale_list = []

    for expert in range(num_experts):
        w13_gs = compute_global_scale(w13[expert])
        w2_gs = compute_global_scale(w2[expert])

        w13_fp4, w13_scale = ops.scaled_fp4_quant(w13[expert], w13_gs)
        w2_fp4, w2_scale = ops.scaled_fp4_quant(w2[expert], w2_gs)

        w13_fp4_list.append(w13_fp4)
        w13_scale_list.append(w13_scale)
        w2_fp4_list.append(w2_fp4)
        w2_scale_list.append(w2_scale)

    w13_weight = torch.stack(w13_fp4_list)
    w13_weight_scale = torch.stack(w13_scale_list)
    w2_weight = torch.stack(w2_fp4_list)
    w2_weight_scale = torch.stack(w2_scale_list)

    # Create global scales
    a1_gscale = torch.tensor(1.0, device=device, dtype=torch.float32)
    g1_scale_c = torch.ones(num_experts, device=device, dtype=torch.float32)
    g1_alphas = torch.ones(num_experts, device=device, dtype=torch.float32)
    g2_alphas = torch.ones(num_experts, device=device, dtype=torch.float32)

    return MockMoELayer(
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        a1_gscale=a1_gscale,
        g1_scale_c=g1_scale_c,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        intermediate_size_per_partition=intermediate_size,
        local_num_experts=num_experts,
        ep_rank=0,
        routing_method_type=1,  # Renormalize
    )


def benchmark_e2e_moe_forward(
    layer: MockMoELayer,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    method: str,  # "flashinfer_swizzled" or "vllm_linear"
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> tuple[float, float, float]:
    """
    Benchmark the COMPLETE MoE forward pass end-to-end.

    This includes:
    1. Input quantization (with different methods)
    2. Router/expert selection
    3. Full MoE kernel execution (GEMM operations)

    Args:
        layer: Mock MoE layer
        x: Input tensor [batch_size, hidden_size]
        router_logits: Router logits [batch_size, num_experts]
        top_k: Number of experts per token
        global_num_experts: Total number of experts
        method: Quantization method to use
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        (median_time_us, min_time_us, max_time_us)
    """
    import flashinfer

    if method == "flashinfer_swizzled":
        # FlashInfer with swizzled layout (reviewer's recommended approach)
        def moe_forward():
            # Quantize input using FlashInfer (directly outputs swizzled layout)
            hidden_states_fp4, hidden_states_scale = flashinfer_fp4_quantize(
                x, layer.a1_gscale, is_sf_swizzled_layout=True
            )
            hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn)

            # Complete MoE forward pass
            out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=router_logits.to(torch.bfloat16),
                routing_bias=None,
                hidden_states=hidden_states_fp4,
                hidden_states_scale=hidden_states_scale.flatten(),
                gemm1_weights=layer.w13_weight.data,
                gemm1_weights_scale=layer.w13_weight_scale.data.view(
                    torch.float8_e4m3fn
                ),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.w2_weight.data,
                gemm2_weights_scale=layer.w2_weight_scale.data.view(
                    torch.float8_e4m3fn
                ),
                gemm2_bias=None,
                output1_scale_scalar=layer.g1_scale_c.data,
                output1_scale_gate_scalar=layer.g1_alphas.data,
                output2_scale_scalar=layer.g2_alphas.data,
                num_experts=global_num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=layer.intermediate_size_per_partition,
                local_expert_offset=layer.ep_rank * layer.local_num_experts,
                local_num_experts=layer.local_num_experts,
                routed_scaling_factor=None,
                tile_tokens_dim=None,
                routing_method_type=layer.routing_method_type,
                do_finalize=True,
            )[0]
            return out

    elif method == "vllm_linear":
        # vLLM native + convert_swizzled_to_linear (current implementation)
        def moe_forward():
            # Quantize using vLLM native op (outputs swizzled scales)
            hidden_states_fp4, hidden_states_scale_swizzled = ops.scaled_fp4_quant(
                x, layer.a1_gscale
            )

            # Convert swizzled scales to linear layout (performance overhead)
            hidden_states_scale_linear = convert_swizzled_to_linear(
                hidden_states_scale_swizzled, x.shape[0], x.shape[1], block_size=16
            )

            # Complete MoE forward pass
            out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
                routing_logits=router_logits.to(torch.bfloat16),
                routing_bias=None,
                hidden_states=hidden_states_fp4,
                hidden_states_scale=hidden_states_scale_linear.view(
                    torch.float8_e4m3fn
                ).flatten(),
                gemm1_weights=layer.w13_weight.data,
                gemm1_weights_scale=layer.w13_weight_scale.data.view(
                    torch.float8_e4m3fn
                ),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=layer.w2_weight.data,
                gemm2_weights_scale=layer.w2_weight_scale.data.view(
                    torch.float8_e4m3fn
                ),
                gemm2_bias=None,
                output1_scale_scalar=layer.g1_scale_c.data,
                output1_scale_gate_scalar=layer.g1_alphas.data,
                output2_scale_scalar=layer.g2_alphas.data,
                num_experts=global_num_experts,
                top_k=top_k,
                n_group=0,
                topk_group=0,
                intermediate_size=layer.intermediate_size_per_partition,
                local_expert_offset=layer.ep_rank * layer.local_num_experts,
                local_num_experts=layer.local_num_experts,
                routed_scaling_factor=None,
                tile_tokens_dim=None,
                routing_method_type=layer.routing_method_type,
                do_finalize=True,
            )[0]
            return out

    else:
        raise ValueError(f"Unknown method: {method}")

    # Warmup
    for _ in range(num_warmup):
        moe_forward()
    torch.cuda.synchronize()

    # Benchmark with CUDA graph
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        moe_forward, quantiles=quantiles
    )

    return ms * 1000, min_ms * 1000, max_ms * 1000  # Convert to us


def benchmark_quantization_only(
    x: torch.Tensor,
    a1_gscale: torch.Tensor,
    method: str,
    num_warmup: int = 10,
) -> tuple[float, float, float]:
    """Benchmark only the quantization step (for comparison)."""
    if method == "flashinfer_swizzled":

        def quantize_fn():
            fp4, scale = flashinfer_fp4_quantize(
                x, a1_gscale, is_sf_swizzled_layout=True
            )
            return fp4, scale.view(torch.float8_e4m3fn)

    elif method == "vllm_linear":

        def quantize_fn():
            fp4, scale_swizzled = ops.scaled_fp4_quant(x, a1_gscale)
            scale_linear = convert_swizzled_to_linear(
                scale_swizzled, x.shape[0], x.shape[1], block_size=16
            )
            return fp4, scale_linear
    else:
        raise ValueError(f"Unknown method: {method}")

    # Warmup
    for _ in range(num_warmup):
        quantize_fn()
    torch.cuda.synchronize()

    # Benchmark
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        quantize_fn, quantiles=quantiles
    )

    return ms * 1000, min_ms * 1000, max_ms * 1000


def run_comprehensive_benchmark(
    batch_sizes: list[int],
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_experts: int = 8,
    top_k: int = 2,
    num_iterations: int = 100,
):
    """Run comprehensive end-to-end benchmark."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 100)
    print("END-TO-END BENCHMARK: FlashInfer Swizzled vs vLLM Linear (TRTLLM FP4 MoE)")
    print("=" * 100)
    print(f"Hidden size: {hidden_size}")
    print(f"Intermediate size: {intermediate_size}")
    print(f"Num experts: {num_experts}, Top-k: {top_k}")
    print(f"Iterations per test: {num_iterations}")
    print("=" * 100)

    # Create mock MoE layer (once, reuse for all batch sizes)
    print("\nCreating mock MoE layer...")
    layer = create_mock_moe_layer(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=dtype,
    )
    print("MoE layer created.")

    results = []

    for M in batch_sizes:
        print(f"\n{'=' * 100}")
        print(f"Benchmark: M={M}, K={hidden_size}")
        print(f"{'=' * 100}")

        # Create input and router logits
        x = torch.randn((M, hidden_size), device=device, dtype=dtype) / 10
        router_logits = torch.randn((M, num_experts), device=device, dtype=dtype)

        # Compute global scale
        a1_gscale = compute_global_scale(x)

        # 1. Benchmark quantization step only
        print("\n1. Quantization Step Only:")
        print("-" * 100)

        flashinfer_quant_us, _, _ = benchmark_quantization_only(
            x, a1_gscale, "flashinfer_swizzled"
        )
        vllm_quant_us, _, _ = benchmark_quantization_only(x, a1_gscale, "vllm_linear")

        print(f"  FlashInfer (swizzled): {flashinfer_quant_us:.2f} us")
        print(f"  vLLM + convert:        {vllm_quant_us:.2f} us")
        print(f"  Speedup:               {vllm_quant_us / flashinfer_quant_us:.2f}x")
        quant_overhead_pct = (
            (vllm_quant_us - flashinfer_quant_us) / flashinfer_quant_us * 100
        )
        print(f"  Overhead:              {quant_overhead_pct:.1f}%")

        # 2. Benchmark end-to-end MoE forward pass
        print("\n2. End-to-End MoE Forward Pass (COMPLETE):")
        print("-" * 100)

        flashinfer_e2e_us, flashinfer_min, flashinfer_max = benchmark_e2e_moe_forward(
            layer,
            x,
            router_logits,
            top_k,
            num_experts,
            "flashinfer_swizzled",
            num_iterations=num_iterations,
        )
        vllm_e2e_us, vllm_min, vllm_max = benchmark_e2e_moe_forward(
            layer,
            x,
            router_logits,
            top_k,
            num_experts,
            "vllm_linear",
            num_iterations=num_iterations,
        )

        print(
            f"  FlashInfer (swizzled): {flashinfer_e2e_us:.2f} us "
            f"[Min: {flashinfer_min:.2f}, Max: {flashinfer_max:.2f}]"
        )
        print(
            f"  vLLM + convert:        {vllm_e2e_us:.2f} us "
            f"[Min: {vllm_min:.2f}, Max: {vllm_max:.2f}]"
        )
        speedup = vllm_e2e_us / flashinfer_e2e_us
        print(f"  Speedup:               {speedup:.2f}x")
        overhead_pct = (vllm_e2e_us - flashinfer_e2e_us) / flashinfer_e2e_us * 100
        print(f"  Overhead:              {overhead_pct:.1f}%")

        # Breakdown: conversion overhead
        conversion_overhead = vllm_e2e_us - flashinfer_e2e_us
        conversion_overhead_pct = (
            conversion_overhead / vllm_e2e_us * 100 if vllm_e2e_us > 0 else 0
        )
        print(
            f"\n  Conversion overhead in e2e: "
            f"{conversion_overhead:.2f} us ({conversion_overhead_pct:.1f}%)"
        )

        results.append(
            {
                "M": M,
                "K": hidden_size,
                "flashinfer_quant_us": flashinfer_quant_us,
                "vllm_quant_us": vllm_quant_us,
                "flashinfer_e2e_us": flashinfer_e2e_us,
                "vllm_e2e_us": vllm_e2e_us,
                "conversion_overhead_us": conversion_overhead,
            }
        )

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY - End-to-End Performance")
    print("=" * 100)
    header = (
        f"{'Size':<12} {'FlashInfer(us)':<18} "
        f"{'vLLM+Conv(us)':<18} {'Ratio':<10} {'Overhead%':<12}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        ratio = r["vllm_e2e_us"] / r["flashinfer_e2e_us"]
        overhead_pct = (
            (r["vllm_e2e_us"] - r["flashinfer_e2e_us"]) / r["flashinfer_e2e_us"] * 100
        )
        print(
            f"{r['M']}x{r['K']:<6} "
            f"{r['flashinfer_e2e_us']:<18.2f} "
            f"{r['vllm_e2e_us']:<18.2f} "
            f"{ratio:<10.2f}x "
            f"{overhead_pct:<12.1f}%"
        )
    print("=" * 100)

    # Quantization-only summary
    print("\n" + "=" * 100)
    print("SUMMARY - Quantization Step Only")
    print("=" * 100)
    quant_header = (
        f"{'Size':<12} {'FlashInfer(us)':<18} {'vLLM+Conv(us)':<18} {'Ratio':<10}"
    )
    print(quant_header)
    print("-" * 100)
    for r in results:
        ratio = r["vllm_quant_us"] / r["flashinfer_quant_us"]
        print(
            f"{r['M']}x{r['K']:<6} "
            f"{r['flashinfer_quant_us']:<18.2f} "
            f"{r['vllm_quant_us']:<18.2f} "
            f"{ratio:<10.2f}x"
        )
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end benchmark for TRTLLM FP4 MoE quantization paths (issue #32057)"
        )
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 128, 1024, 4096],  # noqa: E501
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=11008,
        help="Intermediate dimension size",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k experts per token",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    args = parser.parse_args()

    run_comprehensive_benchmark(
        batch_sizes=args.batch_sizes,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_iterations=args.iterations,
    )
