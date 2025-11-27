#!/usr/bin/env python3
"""
Benchmark SP padding impact using vllm bench latency (standard tool).

Test cases as requested in #27700 feedback:
- llama3-70B-FP8, 1024 vs 1025
- llama3-70B, 1024 vs 1025
- llama3-70B, 512 vs 513

Configuration:
- input-len: 1024 vs 1025 (or 512 vs 513)
- output-len: 1
- batch-size: 1
- tensor-parallel-size: 8 (for 8 A6000 GPUs)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_vllm_bench_latency(
    model: str,
    input_len: int,
    output_len: int = 1,
    batch_size: int = 1,
    tensor_parallel_size: int = 8,
    enable_sp: bool = True,
    num_iters: int = 30,
    num_iters_warmup: int = 10,
    enforce_eager: bool = False,
    load_format: str = "auto",
    output_json: Optional[str] = None,
    **extra_args
) -> Optional[Dict]:
    """Run vllm bench latency and parse results."""
    cmd = [
        "vllm",
        "bench",
        "latency",
        "--model", model,
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--batch-size", str(batch_size),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--num-iters", str(num_iters),
        "--num-iters-warmup", str(num_iters_warmup),
        "--load-format", load_format,
    ]
    
    if enable_sp:
        # Enable sequence parallelism via compilation_config
        compilation_config = {
            "mode": 3,  # VLLM_COMPILE
            "pass_config": {
                "enable_sequence_parallelism": True,
            }
        }
        cmd.extend(["--compilation-config", json.dumps(compilation_config)])
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    if output_json:
        cmd.extend(["--output-json", output_json])
    
    # Add extra arguments
    for key, value in extra_args.items():
        if value is not None:
            arg_name = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(arg_name)
            else:
                cmd.extend([arg_name, str(value)])
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    try:
        # Run with real-time output to stdout/stderr
        # Also capture output for parsing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output lines for parsing
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Real-time output
            sys.stdout.flush()
            output_lines.append(line)
        
        process.wait()
        returncode = process.returncode
        stdout = ''.join(output_lines)
        stderr = ''  # Combined with stdout
        
        if returncode != 0:
            print(f"\nERROR: Command failed with return code {returncode}")
            return None
        
        # Parse output for avg latency
        avg_latency = None
        percentiles = {}
        
        for line in output_lines:
            if "Avg latency:" in line:
                try:
                    avg_latency = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
            elif "% percentile latency:" in line:
                try:
                    parts = line.split(":")
                    pct = int(parts[0].strip().replace("%", ""))
                    latency = float(parts[1].strip().split()[0])
                    percentiles[pct] = latency
                except:
                    pass
        
        # Try to load JSON if provided
        if output_json and Path(output_json).exists():
            try:
                with open(output_json, 'r') as f:
                    json_data = json.load(f)
                    if json_data and "avg_latency" in json_data:
                        avg_latency = json_data["avg_latency"]
                    if json_data and "percentiles" in json_data:
                        percentiles = json_data["percentiles"]
            except Exception as e:
                print(f"Warning: Could not load JSON results: {e}")
        
        return {
            "avg_latency": avg_latency,
            "percentiles": percentiles,
            "stdout": stdout,
            "stderr": stderr,
        }
    except Exception as e:
        print(f"ERROR: Exception while running command: {e}")
        return None


def run_test_pair(
    model: str,
    input_len_base: int,
    tensor_parallel_size: int = 8,
    enable_sp: bool = True,
    num_iters: int = 30,
    num_iters_warmup: int = 10,
    enforce_eager: bool = False,
    load_format: str = "auto",
    output_dir: str = "bench_results",
    **extra_args
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Run a test pair: base vs base+1 token counts."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    model_safe = model.replace("/", "_").replace("-", "_")
    
    # Test base (e.g., 1024)
    base_json = output_dir_path / f"{model_safe}_len{input_len_base}.json"
    result_base = run_vllm_bench_latency(
        model=model,
        input_len=input_len_base,
        tensor_parallel_size=tensor_parallel_size,
        enable_sp=enable_sp,
        num_iters=num_iters,
        num_iters_warmup=num_iters_warmup,
        enforce_eager=enforce_eager,
        load_format=load_format,
        output_json=str(base_json),
        **extra_args
    )
    
    # Test base+1 (e.g., 1025)
    base_plus_one_json = output_dir_path / f"{model_safe}_len{input_len_base + 1}.json"
    result_base_plus_one = run_vllm_bench_latency(
        model=model,
        input_len=input_len_base + 1,
        tensor_parallel_size=tensor_parallel_size,
        enable_sp=enable_sp,
        num_iters=num_iters,
        num_iters_warmup=num_iters_warmup,
        enforce_eager=enforce_eager,
        load_format=load_format,
        output_json=str(base_plus_one_json),
        **extra_args
    )
    
    return result_base, result_base_plus_one


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SP padding impact using vllm bench latency"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to benchmark (if not using --run-all)"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        nargs="+",
        help="Base input lengths to test (will test base and base+1)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Tensor parallel size (default: 8 for 8 A6000 GPUs)"
    )
    parser.add_argument(
        "--enable-sp",
        action="store_true",
        default=True,
        help="Enable sequence parallelism (default: True)"
    )
    parser.add_argument(
        "--disable-sp",
        dest="enable_sp",
        action="store_false",
        help="Disable sequence parallelism"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=30,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile (enforce eager mode)"
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        help="Model load format (auto, dummy, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bench_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all test cases from #27700 feedback"
    )
    
    args = parser.parse_args()
    
    if args.run_all:
        # Test cases from #27700 feedback
        test_cases = [
            ("RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8", 1024),
            ("meta-llama/Llama-3.1-70B-Instruct", 1024),
            ("meta-llama/Llama-3.1-70B-Instruct", 512),
        ]
        
        print("\n" + "="*80)
        print("Running all test cases from #27700 feedback")
        print("="*80)
        print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
        print(f"Sequence Parallelism: {'Enabled' if args.enable_sp else 'Disabled'}")
        print(f"Iterations: {args.num_iters} (warmup: {args.num_iters_warmup})")
        print("="*80)
        
        all_results = []
        
        for model, input_len in test_cases:
            print(f"\n{'#'*80}")
            print(f"Testing: {model}")
            print(f"Input length: {input_len} vs {input_len+1}")
            print(f"{'#'*80}")
            
            result_base, result_base_plus_one = run_test_pair(
                model=model,
                input_len_base=input_len,
                tensor_parallel_size=args.tensor_parallel_size,
                enable_sp=args.enable_sp,
                num_iters=args.num_iters,
                num_iters_warmup=args.num_iters_warmup,
                enforce_eager=args.enforce_eager,
                load_format=args.load_format,
                output_dir=args.output_dir,
            )
            
            if result_base and result_base_plus_one:
                base_latency = result_base.get("avg_latency")
                plus_one_latency = result_base_plus_one.get("avg_latency")
                
                if base_latency and plus_one_latency:
                    overhead = ((plus_one_latency - base_latency) / base_latency) * 100
                    all_results.append({
                        "model": model,
                        "input_len": input_len,
                        "base_latency": base_latency,
                        "plus_one_latency": plus_one_latency,
                        "overhead_pct": overhead,
                        "base_p50": result_base.get("percentiles", {}).get(50),
                        "plus_one_p50": result_base_plus_one.get("percentiles", {}).get(50),
                    })
                    
                    print(f"\n{'='*80}")
                    print(f"Results for {model}")
                    print(f"{'='*80}")
                    print(f"  Base ({input_len} tokens):     {base_latency:.4f}s")
                    print(f"  Base+1 ({input_len+1} tokens): {plus_one_latency:.4f}s")
                    print(f"  Overhead: {overhead:+.2f}%")
                    if result_base.get("percentiles", {}).get(50):
                        print(f"  P50 Base:     {result_base['percentiles'][50]:.4f}s")
                        print(f"  P50 Base+1:   {result_base_plus_one['percentiles'][50]:.4f}s")
                    print(f"{'='*80}")
                else:
                    print(f"WARNING: Could not parse latency from results")
            else:
                print(f"ERROR: Failed to run benchmark for {model}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Model':<50} {'Len':<8} {'Base(s)':<10} {'Base+1(s)':<12} {'Overhead':<10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['model']:<50} {r['input_len']:<8} {r['base_latency']:<10.4f} "
                  f"{r['plus_one_latency']:<12.4f} {r['overhead_pct']:+7.2f}%")
        print("="*80)
        
        # Save summary JSON
        summary_file = Path(args.output_dir) / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")
        
    else:
        # Run custom test case
        if not args.model or not args.input_len:
            parser.error("--model and --input-len are required when not using --run-all")
        
        for input_len in args.input_len:
            print(f"\n{'#'*80}")
            print(f"Testing: {args.model}, input_len={input_len} vs {input_len+1}")
            print(f"{'#'*80}")
            
            result_base, result_base_plus_one = run_test_pair(
                model=args.model,
                input_len_base=input_len,
                tensor_parallel_size=args.tensor_parallel_size,
                enable_sp=args.enable_sp,
                num_iters=args.num_iters,
                num_iters_warmup=args.num_iters_warmup,
                enforce_eager=args.enforce_eager,
                load_format=args.load_format,
                output_dir=args.output_dir,
            )
            
            if result_base and result_base_plus_one:
                base_latency = result_base.get("avg_latency")
                plus_one_latency = result_base_plus_one.get("avg_latency")
                
                if base_latency and plus_one_latency:
                    overhead = ((plus_one_latency - base_latency) / base_latency) * 100
                    print(f"\nResults:")
                    print(f"  Base ({input_len} tokens):     {base_latency:.4f}s")
                    print(f"  Base+1 ({input_len+1} tokens): {plus_one_latency:.4f}s")
                    print(f"  Overhead: {overhead:+.2f}%")


if __name__ == "__main__":
    main()

