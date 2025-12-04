#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark Sequence Parallelism padding overhead.

Tests issue #29136: compare padding overhead with reduce_scatterv uneven work.

Usage:
    python benchmarks/benchmark_sp_padding.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --tensor-parallel-size 4 \
        --test-token-counts "1024,1025" \
        --num-iters 10
"""

import time

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, PassConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def run_benchmark(
    llm: LLM,
    num_tokens: int,
    num_iters: int,
    warmup_iters: int = 3,
) -> dict:
    """Run benchmark for a specific token count."""

    # Generate dummy prompt
    prompt_token_ids = list(range(num_tokens))
    prompts = [{"prompt_token_ids": prompt_token_ids}]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )

    # Warmup - more aggressive for stability
    for _ in range(warmup_iters):
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Additional sync before measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Clear any residual state
        torch.cuda.empty_cache()

    # Measurement
    latencies = []
    for _ in range(num_iters):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
        else:
            start = time.perf_counter()
            llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            latency_ms = (time.perf_counter() - start) * 1000

        latencies.append(latency_ms)

    latencies = np.array(latencies)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def main():
    parser = FlexibleArgumentParser(description="Benchmark SP padding overhead")

    # Add benchmark-specific arguments first
    parser.add_argument(
        "--test-token-counts",
        type=str,
        required=True,
        help="Comma-separated token counts (e.g., '1024,1025')",
    )
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=3)

    # Add EngineArgs (includes --model, --tensor-parallel-size, etc.)
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    print("=" * 80)
    print("Sequence Parallelism Padding Benchmark - Issue #29136")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tensor_parallel_size}")
    print("=" * 80)

    # Parse token counts
    test_token_counts = [int(x.strip()) for x in args.test_token_counts.split(",")]
    print(f"Testing: {test_token_counts}")
    print("=" * 80)

    # Initialize LLM with SP enabled
    engine_args = EngineArgs.from_cli_args(args)

    # Force enable Sequence Parallelism if TP > 1
    if args.tensor_parallel_size > 1:
        print(f"Enabling Sequence Parallelism (TP={args.tensor_parallel_size})")
        if engine_args.compilation_config is None:
            engine_args.compilation_config = CompilationConfig()
        if engine_args.compilation_config.pass_config is None:
            engine_args.compilation_config.pass_config = PassConfig()
        engine_args.compilation_config.pass_config.enable_sp = True
        print("=" * 80)

    llm = LLM(**vars(engine_args))

    # Run benchmarks
    results = {}
    tp_size = args.tensor_parallel_size

    # Interleave tests to reduce caching effects
    # Test each size multiple times in a mixed order
    test_order = []
    for _ in range(3):  # Repeat 3 times
        test_order.extend(test_token_counts)

    print("\nRunning tests in interleaved order (3 rounds)...")
    print("=" * 80)

    raw_results = {tokens: [] for tokens in test_token_counts}

    for num_tokens in test_order:
        print(f"\n[Round] Testing {num_tokens} tokens...")

        result = run_benchmark(
            llm=llm,
            num_tokens=num_tokens,
            num_iters=args.num_iters,
            warmup_iters=args.warmup_iters,
        )

        raw_results[num_tokens].append(result["mean_ms"])
        print(f"  Result: {result['mean_ms']:.2f} ms")

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS")
    print("=" * 80)

    for num_tokens in test_token_counts:
        is_multiple = (num_tokens % tp_size) == 0
        padding = (tp_size - (num_tokens % tp_size)) % tp_size

        times = np.array(raw_results[num_tokens])

        results[num_tokens] = {
            "is_multiple": is_multiple,
            "padding": padding,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }

        mean = results[num_tokens]["mean_ms"]
        std = results[num_tokens]["std_ms"]
        print(f"{num_tokens} tokens: {mean:.2f} ± {std:.2f} ms")

    # Analysis
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    headers = (
        f"{'Tokens':<10} {'Mean (ms)':<12} {'Std (ms)':<12} "
        f"{'Multiple':<10} {'Padding':<10}"
    )
    print(headers)
    print("-" * 80)

    for num_tokens in test_token_counts:
        r = results[num_tokens]
        multiple_str = "Yes" if r["is_multiple"] else "No"
        print(
            f"{num_tokens:<10} {r['mean_ms']:<12.2f} {r['std_ms']:<12.2f} "
            f"{multiple_str:<10} {r['padding']:<10}"
        )

    # Compare overhead
    multiples = [r for r in results.values() if r["is_multiple"]]
    non_multiples = [r for r in results.values() if not r["is_multiple"]]

    if multiples and non_multiples:
        avg_mult = np.mean([r["mean_ms"] for r in multiples])
        avg_non_mult = np.mean([r["mean_ms"] for r in non_multiples])
        overhead = ((avg_non_mult - avg_mult) / avg_mult * 100) if avg_mult > 0 else 0

        print("\n" + "-" * 80)
        print(f"Average (divisible):     {avg_mult:.2f} ms")
        print(f"Average (non-divisible): {avg_non_mult:.2f} ms")
        print(f"Padding overhead:        {overhead:+.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
