#!/usr/bin/env python3
"""
Local benchmarking script for vLLM inference with different configurations.
Tests various batch sizes and quantization methods to find optimal settings.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Optional imports for different inference engines
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def load_test_data(dataset_name: str = "codefactory4791/amazon_test", 
                   split: str = "test", 
                   num_samples: int = 1000,
                   balance_lengths: bool = True) -> tuple:
    """
    Load test dataset and return prompts and labels.
    Optionally balance the dataset to include short, medium, and long sequences.
    """
    print(f"Loading dataset: {dataset_name} (split={split})")
    
    # Only download the test split
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    
    # Rename columns if needed
    if 'query' in df.columns and 'label' in df.columns:
        df = df.rename(columns={'query': 'text', 'label': 'labels'})
    
    # Add text length column
    df['text_length'] = df['text'].str.len()
    
    if balance_lengths and num_samples:
        # Use adaptive thresholds based on data distribution (percentiles)
        # This ensures we always get 3 balanced categories
        p33 = df['text_length'].quantile(0.33)
        p67 = df['text_length'].quantile(0.67)
        
        # Categorize by length using adaptive thresholds
        df['length_category'] = 'medium'
        df.loc[df['text_length'] < p33, 'length_category'] = 'short'
        df.loc[df['text_length'] > p67, 'length_category'] = 'long'
        
        # Check available samples in each category
        short_available = len(df[df['length_category'] == 'short'])
        medium_available = len(df[df['length_category'] == 'medium'])
        long_available = len(df[df['length_category'] == 'long'])
        
        print(f"  Adaptive length thresholds (based on percentiles):")
        print(f"    Short: < {p33:.0f} chars ({short_available} samples)")
        print(f"    Medium: {p33:.0f}-{p67:.0f} chars ({medium_available} samples)")
        print(f"    Long: > {p67:.0f} chars ({long_available} samples)")
        
        # Calculate samples per category (roughly 1/3 each)
        samples_per_category = num_samples // 3
        remainder = num_samples % 3
        
        # Sample from each category
        dfs = []
        for i, category in enumerate(['short', 'medium', 'long']):
            cat_df = df[df['length_category'] == category]
            n_samples = samples_per_category + (1 if i < remainder else 0)
            
            if len(cat_df) >= n_samples:
                sampled = cat_df.sample(n=n_samples, random_state=42)
            else:
                # If not enough samples, take all available
                sampled = cat_df
                if len(cat_df) > 0:
                    print(f"  Warning: Only {len(cat_df)} {category} samples available (requested {n_samples})")
            
            if len(sampled) > 0:
                dfs.append(sampled)
        
        # Combine and shuffle
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Print final distribution
            print(f"\n  Final sampled distribution ({len(df)} total):")
            for category in ['short', 'medium', 'long']:
                cat_data = df[df['length_category'] == category]
                count = len(cat_data)
                if count > 0:
                    avg_len = cat_data['text_length'].mean()
                    min_len = cat_data['text_length'].min()
                    max_len = cat_data['text_length'].max()
                    print(f"    {category.capitalize()}: {count} samples")
                    print(f"      Length range: {min_len:.0f}-{max_len:.0f} chars (avg: {avg_len:.0f})")
                else:
                    print(f"    {category.capitalize()}: 0 samples")
        else:
            print(f"  Warning: No samples found in any length category")
    
    elif num_samples and num_samples < len(df):
        # Simple random sampling without balancing
        df = df.sample(n=num_samples, random_state=42)
        print(f"  Sampled {num_samples} instances for testing")
        avg_length = df['text_length'].mean()
        print(f"    Average length: {avg_length:.0f} characters")
    
    # Extract data
    prompts = df['text'].tolist()
    
    # Build label mappings
    labels_unique = sorted(df['labels'].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(labels_unique)}
    id2label = {idx: lbl for idx, lbl in enumerate(labels_unique)}
    df['label_id'] = df['labels'].map(label2id)
    true_labels = df['label_id'].tolist()
    
    print(f"  Loaded {len(prompts)} prompts with {len(labels_unique)} classes")
    
    return prompts, true_labels, id2label


def initialize_model(model_id: str, quantization: str = "none", 
                    inference_engine: str = "transformers",
                    use_optimizations: bool = True) -> tuple:
    """
    Initialize model and tokenizer with specified configuration.
    
    Args:
        model_id: HuggingFace model ID
        quantization: Quantization method (none, bitsandbytes, awq, gptq)
        inference_engine: Inference engine (transformers, onnx)
        use_optimizations: Apply optimizations (BetterTransformer, torch.compile)
    
    Returns:
        tuple: (model, tokenizer, device, engine_type)
    """
    print(f"\nInitializing model...")
    print(f"  Model: {model_id}")
    print(f"  Quantization: {quantization}")
    print(f"  Inference Engine: {inference_engine}")
    print(f"  Optimizations: {use_optimizations}")
    
    start = time.perf_counter()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ONNX Runtime Engine
        if inference_engine == "onnx":
            if not ONNX_AVAILABLE:
                print("  Error: ONNX Runtime not available. Install with: pip install optimum[onnxruntime-gpu]")
                sys.exit(1)
            
            print(f"  Loading model with ONNX Runtime...")
            
            # Check if ONNX model already exists locally
            onnx_path = Path(f"./models_onnx/{quantization}")
            
            if onnx_path.exists():
                print(f"  Loading from cached ONNX model: {onnx_path}")
                model = ORTModelForSequenceClassification.from_pretrained(
                    str(onnx_path),
                    provider="CUDAExecutionProvider"
                )
            else:
                print(f"  Converting model to ONNX format...")
                model = ORTModelForSequenceClassification.from_pretrained(
                    model_id,
                    export=True,
                    provider="CUDAExecutionProvider"
                )
                # Save for future use
                onnx_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(onnx_path))
                tokenizer.save_pretrained(str(onnx_path))
                print(f"  ONNX model cached to: {onnx_path}")
            
            engine_type = "onnx"
        
        # Standard Transformers Engine
        else:
            if quantization == "bitsandbytes":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            elif quantization == "awq":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    device_map="auto"
                )
            elif quantization == "gptq":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    device_map="auto"
                )
            else:
                # No quantization - simple loading
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                model = model.to(device)
            
            # Ensure model knows the pad token
            model.config.pad_token_id = tokenizer.pad_token_id
            model.eval()
            
            # Apply optimizations for transformers engine
            if use_optimizations and quantization == "none" and inference_engine == "transformers":
                # Apply BetterTransformer
                try:
                    model = model.to_bettertransformer()
                    print(f"  Applied BetterTransformer optimization")
                except Exception as e:
                    print(f"  Warning: Could not apply BetterTransformer: {e}")
                
                # Apply torch.compile if available
                if hasattr(torch, "compile"):
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        print(f"  Applied torch.compile optimization")
                    except Exception as e:
                        print(f"  Warning: Could not apply torch.compile: {e}")
            
            engine_type = "transformers"
        
        load_time = time.perf_counter() - start
        print(f"  Model loaded in {load_time:.2f}s")
        print(f"  Device: {device}")
        print(f"  Engine: {engine_type}")
        
        return model, tokenizer, device, engine_type
        
    except Exception as e:
        print(f"  Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def benchmark_batch_size(model, tokenizer, device, prompts: List[str], batch_size: int, 
                         engine_type: str = "transformers") -> Dict[str, float]:
    """Run benchmark for a specific batch size."""
    # Split prompts into batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    latencies = []
    total_samples = 0
    
    start_total = time.perf_counter()
    
    for batch in batches:
        start_batch = time.perf_counter()
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_latency = (time.perf_counter() - start_batch) * 1000  # Convert to ms
        latencies.append(batch_latency)
        total_samples += len(batch)
    
    total_time = time.perf_counter() - start_total
    
    # Calculate metrics
    throughput = total_samples / total_time
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    return {
        "batch_size": batch_size,
        "total_samples": total_samples,
        "total_time": round(total_time, 3),
        "throughput": round(throughput, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
    }


def run_benchmarks(model_id: str, 
                   quantization: str, 
                   batch_sizes: List[int],
                   num_samples: int = 1000,
                   balance_lengths: bool = True,
                   inference_engine: str = "transformers",
                   use_optimizations: bool = True,
                   output_dir: str = "./results/local_benchmarks") -> List[Dict]:
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 70)
    print(f"Starting Benchmarks")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Quantization: {quantization}")
    print(f"Inference Engine: {inference_engine}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Number of samples: {num_samples}")
    print(f"Balance sequence lengths: {balance_lengths}")
    print(f"Use optimizations: {use_optimizations}")
    print("=" * 70)
    
    # Load test data
    prompts, true_labels, id2label = load_test_data(
        num_samples=num_samples,
        balance_lengths=balance_lengths
    )
    
    # Initialize model
    model, tokenizer, device, engine_type = initialize_model(
        model_id, quantization, inference_engine, use_optimizations
    )
    
    # Run benchmarks for each batch size
    results = []
    
    print(f"\n{'=' * 70}")
    print(f"Running Benchmarks")
    print(f"{'=' * 70}")
    
    for batch_size in batch_sizes:
        print(f"\n[Batch Size: {batch_size}]")
        
        metrics = benchmark_batch_size(model, tokenizer, device, prompts, batch_size, engine_type)
        metrics['quantization'] = quantization
        metrics['inference_engine'] = inference_engine
        metrics['model_id'] = model_id
        
        results.append(metrics)
        
        # Print results
        print(f"  Throughput:    {metrics['throughput']:>7.2f} samples/s")
        print(f"  Avg Latency:   {metrics['avg_latency_ms']:>7.2f} ms")
        print(f"  P50 Latency:   {metrics['p50_latency_ms']:>7.2f} ms")
        print(f"  P95 Latency:   {metrics['p95_latency_ms']:>7.2f} ms")
        print(f"  P99 Latency:   {metrics['p99_latency_ms']:>7.2f} ms")
    
    # Save results with engine suffix
    config_name = f"{quantization}_{inference_engine}" if inference_engine != "transformers" else quantization
    output_path = Path(output_dir) / config_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to: {results_file}")
    print(f"{'=' * 70}")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Batch Size':<12} {'Throughput':<15} {'Avg Latency':<15} {'P95 Latency':<15}")
    print("-" * 70)
    
    for r in results:
        print(
            f"{r['batch_size']:<12} "
            f"{r['throughput']:<15.2f} "
            f"{r['avg_latency_ms']:<15.2f} "
            f"{r['p95_latency_ms']:<15.2f}"
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM classification with different configurations"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="codefactory4791/intent-classification-qwen",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "bitsandbytes", "awq", "gptq"],
        help="Quantization method"
    )
    parser.add_argument(
        "--inference-engine",
        type=str,
        default="transformers",
        choices=["transformers", "onnx"],
        help="Inference engine to use (transformers=standard PyTorch, onnx=ONNX Runtime)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated list of batch sizes to test"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to use for benchmarking"
    )
    parser.add_argument(
        "--balance-lengths",
        action="store_true",
        default=True,
        help="Balance dataset with short, medium, and long sequences"
    )
    parser.add_argument(
        "--no-balance-lengths",
        action="store_false",
        dest="balance_lengths",
        help="Disable length balancing (use random sampling)"
    )
    parser.add_argument(
        "--use-optimizations",
        action="store_true",
        default=True,
        help="Enable optimizations (BetterTransformer, torch.compile)"
    )
    parser.add_argument(
        "--no-optimizations",
        action="store_false",
        dest="use_optimizations",
        help="Disable optimizations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/local_benchmarks",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Run benchmarks
    results = run_benchmarks(
        model_id=args.model_id,
        quantization=args.quantization,
        batch_sizes=batch_sizes,
        num_samples=args.num_samples,
        balance_lengths=args.balance_lengths,
        inference_engine=args.inference_engine,
        use_optimizations=args.use_optimizations,
        output_dir=args.output_dir
    )
    
    print("\nBenchmarking complete!")
    print(f"\nNext steps:")
    config_name = f"{args.quantization}_{args.inference_engine}" if args.inference_engine != "transformers" else args.quantization
    print(f"  1. Review results in: {args.output_dir}/{config_name}/")
    print(f"  2. Test other configurations")
    print(f"  3. Compare results: python scripts/compare_results.py")


if __name__ == "__main__":
    main()

