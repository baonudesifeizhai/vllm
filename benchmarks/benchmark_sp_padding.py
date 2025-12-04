#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark Sequence Parallelism padding overhead.

Usage:
    python benchmarks/benchmark_sp_padding.py \
        --model meta-llama/Meta-Llama-3-70B-Instruct-FP8 \
        --tensor-parallel-size 4 \
        --test-token-counts "1024,1025" \
        --output-len 1 \
        --batch-size 1
"""

import argparse
import contextlib
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.profiler

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def analyze_profiler_results(prof: torch.profiler.profile, num_tokens: int) -> dict:
    """Analyze profiler results and extract kernel statistics."""
    events = prof.key_averages()

    # Debug: print total events captured
    print(f"    Debug: Total events captured: {len(events)}")

    # Filter CUDA kernels only - check both device_type and cuda_time
    cuda_events = [
        e
        for e in events
        if (
            hasattr(e, "device_type")
            and e.device_type == torch.profiler.DeviceType.CUDA
        )
        or (hasattr(e, "cuda_time_total") and e.cuda_time_total > 0)
    ]

    print(f"    Debug: CUDA events found: {len(cuda_events)}")

    # Sort by CUDA time
    cuda_events_sorted = sorted(
        cuda_events, key=lambda e: e.cuda_time_total, reverse=True
    )

    # Get top kernels
    top_kernels = []
    total_cuda_time = sum(e.cuda_time_total for e in cuda_events)

    for i, event in enumerate(cuda_events_sorted[:20]):  # Top 20
        kernel_info = {
            "rank": i + 1,
            "name": event.key,
            "cuda_time_us": event.cuda_time_total,
            "cuda_time_ms": event.cuda_time_total / 1000.0,
            "percentage": (event.cuda_time_total / total_cuda_time * 100)
            if total_cuda_time > 0
            else 0,
            "count": event.count,
            "avg_time_us": event.cuda_time_total / event.count
            if event.count > 0
            else 0,
        }
        top_kernels.append(kernel_info)

    # Categorize kernels
    attention_time = sum(
        e.cuda_time_total
        for e in cuda_events
        if any(
            keyword in e.key.lower()
            for keyword in [
                "attention",
                "flash",
                "fmha",
                "sdpa",
            ]
        )
    )

    matmul_time = sum(
        e.cuda_time_total
        for e in cuda_events
        if any(
            keyword in e.key.lower()
            for keyword in [
                "gemm",
                "matmul",
                "cublas",
                "cutlass",
            ]
        )
    )

    layernorm_time = sum(
        e.cuda_time_total
        for e in cuda_events
        if any(keyword in e.key.lower() for keyword in ["norm", "rms"])
    )

    activation_time = sum(
        e.cuda_time_total
        for e in cuda_events
        if any(
            keyword in e.key.lower()
            for keyword in [
                "gelu",
                "silu",
                "relu",
                "activation",
            ]
        )
    )

    return {
        "total_cuda_time_ms": total_cuda_time / 1000.0,
        "top_kernels": top_kernels,
        "category_breakdown": {
            "attention_ms": attention_time / 1000.0,
            "attention_pct": (attention_time / total_cuda_time * 100)
            if total_cuda_time > 0
            else 0,
            "matmul_ms": matmul_time / 1000.0,
            "matmul_pct": (matmul_time / total_cuda_time * 100)
            if total_cuda_time > 0
            else 0,
            "layernorm_ms": layernorm_time / 1000.0,
            "layernorm_pct": (layernorm_time / total_cuda_time * 100)
            if total_cuda_time > 0
            else 0,
            "activation_ms": activation_time / 1000.0,
            "activation_pct": (activation_time / total_cuda_time * 100)
            if total_cuda_time > 0
            else 0,
        },
    }


def run_benchmark(
    llm: LLM,
    num_tokens: int,
    batch_size: int,
    output_len: int,
    warmup_iters: int,
    measure_iters: int,
    seed: int = 42,
    enable_profiling: bool = False,
    profile_dir: str = "profiling_results",
):
    """Run benchmark for a specific token count."""

    # Generate dummy prompts
    set_random_seeds(seed)
    tokens_per_prompt = num_tokens // batch_size
    remainder = num_tokens % batch_size

    prompts = []
    for i in range(batch_size):
        prompt_len = tokens_per_prompt + (1 if i < remainder else 0)
        prompt_token_ids = np.random.randint(100, 32000, size=prompt_len).tolist()
        prompts.append({"prompt_token_ids": prompt_token_ids})

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
        detokenize=False,
    )

    # Warmup (includes compile and graph capture)
    for _ in range(warmup_iters):
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measurement
    latencies_cuda = []
    latencies_wall = []
    profiler_result = None

    for iter_idx in range(measure_iters):
        # Only profile in last iteration (avoid recording compile time)
        should_profile = (
            enable_profiling
            and iter_idx == measure_iters - 1
            and torch.cuda.is_available()
        )

        if should_profile:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)

            # Use context manager, run generate once
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
            ) as prof:
                # Time and generate
                start_wall = time.perf_counter()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
                end_event.record()
                torch.cuda.synchronize()
                end_wall = time.perf_counter()

                # Profiler step is important
                prof.step()

            latency_cuda_ms = start_event.elapsed_time(end_event)
            latency_wall_ms = (end_wall - start_wall) * 1000
            latencies_cuda.append(latency_cuda_ms)
            latencies_wall.append(latency_wall_ms)

            # Export trace
            trace_file = f"{profile_dir}/trace_{num_tokens}tokens.json"
            try:
                prof.export_chrome_trace(trace_file)
                print(f"    Trace exported to: {trace_file}")
            except Exception as e:
                print(f"    Warning: Failed to export trace: {e}")

            # Parse kernel stats
            try:
                profiler_result = analyze_profiler_results(prof, num_tokens)
            except Exception as e:
                print(f"    Warning: Failed to analyze profiler: {e}")
                profiler_result = None

        else:
            # Normal timing, no profiler
            if torch.cuda.is_available():
                start_wall = time.perf_counter()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
                end_event.record()
                torch.cuda.synchronize()
                end_wall = time.perf_counter()

                latency_cuda_ms = start_event.elapsed_time(end_event)
                latency_wall_ms = (end_wall - start_wall) * 1000
                latencies_cuda.append(latency_cuda_ms)
                latencies_wall.append(latency_wall_ms)
            else:
                start = time.perf_counter()
                llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies_cuda.append(latency_ms)
                latencies_wall.append(latency_ms)

    # Use wall time for main analysis (easier to see compile + runtime diff)
    latencies = np.array(latencies_wall)
    latencies_cuda_arr = np.array(latencies_cuda)

    # Remove outliers
    if len(latencies) > 3:
        mean = np.mean(latencies)
        std = np.std(latencies)
        if std > 0:
            filtered = latencies[np.abs(latencies - mean) <= 3 * std]
        else:
            filtered = latencies
    else:
        filtered = latencies

    mean_val = float(np.mean(filtered))
    std_val = float(np.std(filtered))
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0.0

    # CUDA-only metrics
    cuda_metrics = {}
    if len(latencies_cuda_arr) > 0:
        if len(latencies_cuda_arr) > 3:
            c_mean = np.mean(latencies_cuda_arr)
            c_std = np.std(latencies_cuda_arr)
            if c_std > 0:
                filtered_cuda = latencies_cuda_arr[
                    np.abs(latencies_cuda_arr - c_mean) <= 3 * c_std
                ]
            else:
                filtered_cuda = latencies_cuda_arr
        else:
            filtered_cuda = latencies_cuda_arr

        cuda_mean = float(np.mean(filtered_cuda))
        cuda_std = float(np.std(filtered_cuda))

        cuda_metrics = {
            "cuda_mean_ms": cuda_mean,
            "cuda_std_ms": cuda_std,
            "compilation_overhead_ms": float(mean_val - cuda_mean),
        }

    result = {
        "mean_ms": mean_val,
        "median_ms": float(np.median(filtered)),
        "std_ms": std_val,
        "min_ms": float(np.min(filtered)),
        "max_ms": float(np.max(filtered)),
        "p95_ms": float(np.percentile(filtered, 95)),
        "p99_ms": float(np.percentile(filtered, 99)),
        "cv_percent": float(cv),
        **cuda_metrics,
    }

    if profiler_result:
        result["profiler"] = profiler_result

    return result


def main(args: argparse.Namespace):
    """Main function."""

    print("=" * 80)
    print("Sequence Parallelism Padding Benchmark")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tensor_parallel_size}")
    print(f"Output Length: {args.output_len}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 80)

    # Parse token counts
    test_token_counts = [int(x.strip()) for x in args.test_token_counts.split(",")]
    print(f"Testing: {test_token_counts}")
    print("=" * 80)

    # Initialize LLM
    engine_args = EngineArgs.from_cli_args(args)

    # Force enable Sequence Parallelism if TP > 1
    from vllm.config.compilation import PassConfig

    if args.tensor_parallel_size > 1:
        print(f"Force enabling Sequence Parallelism (TP={args.tensor_parallel_size})")
        if engine_args.compilation_config is None:
            engine_args.compilation_config = CompilationConfig()
        if engine_args.compilation_config.pass_config is None:
            engine_args.compilation_config.pass_config = PassConfig()
        engine_args.compilation_config.pass_config.enable_sp = True
        print("=" * 80)

    # FIX: Disable CUDAGraph on EngineArgs object before asdict
    if args.disable_cudagraph_for_profiling and args.enable_profiling:
        print("Note: Disabling CUDAGraph for accurate kernel profiling")
        if engine_args.compilation_config is None:
            engine_args.compilation_config = CompilationConfig(
                cudagraph_mode=CUDAGraphMode.NONE
            )
        else:
            engine_args.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    llm_kwargs = dataclasses.asdict(engine_args)
    llm = LLM(**llm_kwargs)

    # Check SP status from vllm_config
    sp_enabled = False
    compilation_config = None
    cudagraph_mode = None

    with contextlib.suppress(AttributeError):
        if hasattr(llm.llm_engine, "vllm_config"):
            compilation_config = llm.llm_engine.vllm_config.compilation_config
            if compilation_config and compilation_config.pass_config:
                sp_enabled = compilation_config.pass_config.enable_sp or False
            if compilation_config:
                cudagraph_mode = compilation_config.cudagraph_mode

    print(f"Sequence Parallelism: {'ENABLED' if sp_enabled else 'DISABLED'}")
    print("=" * 80)

    # Show CUDAGraph status
    status = "DISABLED" if cudagraph_mode == CUDAGraphMode.NONE else "ENABLED"
    print(f"CUDAGraph: {status}")
    print("=" * 80)

    # Run benchmarks
    results = {}
    tp_size = args.tensor_parallel_size

    for num_tokens in test_token_counts:
        print(f"\nTesting {num_tokens} tokens...")

        is_multiple = (num_tokens % tp_size) == 0
        padding = (tp_size - (num_tokens % tp_size)) % tp_size
        padded = num_tokens + padding

        print(f"  Multiple of {tp_size}: {is_multiple}")
        print(f"  Padding: {padding} tokens → {padded} total")

        bench_result = run_benchmark(
            llm=llm,
            num_tokens=num_tokens,
            batch_size=args.batch_size,
            output_len=args.output_len,
            warmup_iters=args.num_iters_warmup,
            measure_iters=args.num_iters,
            seed=args.seed,
            enable_profiling=args.enable_profiling,
            profile_dir=args.profile_dir,
        )

        results[num_tokens] = {
            "is_multiple": is_multiple,
            "padding": padding,
            **bench_result,
        }

        print(f"  Wall clock mean: {bench_result['mean_ms']:.3f} ms")
        if "cuda_mean_ms" in bench_result:
            cuda_mean = bench_result["cuda_mean_ms"]
            comp_overhead = bench_result["compilation_overhead_ms"]
            print(f"  CUDA kernel mean: {cuda_mean:.3f} ms")
            print(f"  Compilation overhead: {comp_overhead:.3f} ms")
        std_ms = bench_result["std_ms"]
        cv_pct = bench_result["cv_percent"]
        print(f"  Std: {std_ms:.3f} ms")
        print(f"  CV: {cv_pct:.2f}%")

        # Print profiler results if available
        if "profiler" in bench_result:
            prof = bench_result["profiler"]
            print("\n  Kernel Profiling Results:")
            print(f"    Total CUDA time: {prof['total_cuda_time_ms']:.3f} ms")
            print("    Category breakdown:")
            breakdown = prof["category_breakdown"]
            attn_ms = breakdown["attention_ms"]
            attn_pct = breakdown["attention_pct"]
            print(f"      Attention: {attn_ms:.3f} ms ({attn_pct:.1f}%)")
            mm_ms = breakdown["matmul_ms"]
            mm_pct = breakdown["matmul_pct"]
            print(f"      MatMul:    {mm_ms:.3f} ms ({mm_pct:.1f}%)")
            ln_ms = breakdown["layernorm_ms"]
            ln_pct = breakdown["layernorm_pct"]
            print(f"      LayerNorm: {ln_ms:.3f} ms ({ln_pct:.1f}%)")
            act_ms = breakdown["activation_ms"]
            act_pct = breakdown["activation_pct"]
            print(f"      Activation:{act_ms:.3f} ms ({act_pct:.1f}%)")
            print("    Top 5 kernels:")
            for kernel in prof["top_kernels"][:5]:
                print(f"      {kernel['rank']}. {kernel['name'][:60]}")
                k_time = kernel["cuda_time_ms"]
                k_pct = kernel["percentage"]
                k_cnt = kernel["count"]
                print(f"         {k_time:.3f} ms ({k_pct:.1f}%), count={k_cnt}")

    # Analysis
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    headers = (
        f"{'Tokens':<10} {'Mean (ms)':<12} {'Std (ms)':<12} "
        f"{'CV (%)':<10} {'Multiple':<10} {'Padding':<10}"
    )
    print(headers)
    print("-" * 80)

    for num_tokens in test_token_counts:
        r = results[num_tokens]
        multiple_str = "Yes" if r["is_multiple"] else "No"
        row = (
            f"{num_tokens:<10} {r['mean_ms']:<12.3f} "
            f"{r['std_ms']:<12.3f} {r['cv_percent']:<10.2f} "
            f"{multiple_str:<10} {r['padding']:<10}"
        )
        print(row)

    # Compare if we have both multiple and non-multiple
    multiples = [r for r in results.values() if r["is_multiple"]]
    non_multiples = [r for r in results.values() if not r["is_multiple"]]

    if multiples and non_multiples:
        avg_mult = np.mean([r["mean_ms"] for r in multiples])
        avg_non_mult = np.mean([r["mean_ms"] for r in non_multiples])
        overhead = ((avg_non_mult - avg_mult) / avg_mult * 100) if avg_mult > 0 else 0

        print("\n" + "-" * 80)
        print(f"Average (multiples): {avg_mult:.3f} ms")
        print(f"Average (non-multiples): {avg_non_mult:.3f} ms")
        print(f"Padding overhead: {overhead:+.2f}%")

    # Save results
    if args.output_json:
        output = {
            "config": {
                "model": args.model,
                "tp_size": args.tensor_parallel_size,
                "sp_enabled": sp_enabled,
                "batch_size": args.batch_size,
                "output_len": args.output_len,
                "cudagraph_mode": str(cudagraph_mode),
            },
            "results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    print("=" * 80)


if __name__ == "__main__":
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="Benchmark SP padding overhead")

    parser.add_argument(
        "--test-token-counts",
        type=str,
        required=True,
        help="Comma-separated token counts to test (e.g., '1024,1025')",
    )
    parser.add_argument("--output-len", type=int, default=1, help="Output length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-iters-warmup", type=int, default=10, help="Warmup iterations"
    )
    parser.add_argument(
        "--num-iters", type=int, default=30, help="Measurement iterations"
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="Output JSON file"
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable PyTorch profiler to trace kernel execution",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="profiling_results",
        help="Directory to save profiling traces",
    )
    parser.add_argument(
        "--disable-cudagraph-for-profiling",
        action="store_true",
        help="Disable CUDA Graph when profiling to capture individual kernels",
    )

    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False, seed=42)

    args = parser.parse_args()
    main(args)
