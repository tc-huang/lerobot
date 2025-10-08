#!/usr/bin/env python3
"""
Benchmark script to demonstrate torch.compile improvements for ACT policy.

This script shows the performance difference between baseline and compiled ACT models.

Usage:
    python examples/benchmark_act_compile.py [--device cuda] [--batch-size 8] [--num-runs 100]
"""
import argparse
import time
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT


def create_benchmark_config():
    """Create ACT config suitable for benchmarking."""
    config = ACTConfig()
    config.action_feature = PolicyFeature(shape=[14], type=FeatureType.STATE)
    config.robot_state_feature = PolicyFeature(shape=[14], type=FeatureType.STATE)
    config.image_features = [
        PolicyFeature(shape=[3, 96, 96], type=FeatureType.IMAGE)
    ]
    config.chunk_size = 100
    config.n_action_steps = 100
    config.vision_backbone = "resnet18"
    config.use_vae = False  # Benchmark inference mode
    return config


def create_dummy_batch(config, batch_size, device):
    """Create a dummy batch for benchmarking."""
    batch = {}
    
    if config.robot_state_feature:
        batch['observation.state'] = torch.randn(
            batch_size, config.robot_state_feature.shape[0], device=device
        )
    
    if config.image_features:
        batch['observation.images'] = []
        for img_feat in config.image_features:
            h, w = img_feat.shape[1], img_feat.shape[2]
            batch['observation.images'].append(
                torch.randn(batch_size, 3, h, w, device=device)
            )
    
    return batch


def benchmark_model(model, batch, num_warmup=10, num_runs=50):
    """Benchmark inference time of a model."""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(batch)
    
    # Synchronize for accurate timing on CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(batch)
    
    # Synchronize again
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.time()
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark ACT with torch.compile")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run benchmark on')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--num-runs', type=int, default=50,
                        help='Number of runs for benchmarking')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = args.device
    print(f"Running benchmark on: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Compile mode: {args.compile_mode}")
    print()
    
    # Create model and batch
    config = create_benchmark_config()
    model = ACT(config).to(device)
    model.eval()
    
    batch = create_dummy_batch(config, args.batch_size, device)
    
    # Baseline benchmark
    print("="*80)
    print("BASELINE (no torch.compile)")
    print("="*80)
    
    baseline_time = benchmark_model(model, batch, num_runs=args.num_runs)
    print(f"Average inference time: {baseline_time:.2f}ms")
    print(f"Throughput: {args.batch_size * 1000 / baseline_time:.2f} samples/sec")
    print()
    
    # Compiled benchmark
    print("="*80)
    print("COMPILED (with torch.compile)")
    print("="*80)
    
    compiled_model = torch.compile(model, mode=args.compile_mode)
    
    # Note: First run will be slower due to compilation
    print("Compiling model (first run)...")
    with torch.no_grad():
        _ = compiled_model(batch)
    print("Compilation complete!")
    print()
    
    compiled_time = benchmark_model(compiled_model, batch, num_runs=args.num_runs)
    print(f"Average inference time: {compiled_time:.2f}ms")
    print(f"Throughput: {args.batch_size * 1000 / compiled_time:.2f} samples/sec")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    speedup = baseline_time / compiled_time
    print(f"Baseline:      {baseline_time:.2f}ms")
    print(f"Compiled:      {compiled_time:.2f}ms")
    print(f"Speedup:       {speedup:.2f}x")
    print()
    
    if speedup > 1.1:
        print(f"✓ Compilation provides {speedup:.2f}x speedup!")
    elif speedup > 1.0:
        print(f"✓ Minor speedup of {speedup:.2f}x")
    else:
        print(f"⚠ Compiled version is {1/speedup:.2f}x slower (may need more warmup)")


if __name__ == '__main__':
    main()
