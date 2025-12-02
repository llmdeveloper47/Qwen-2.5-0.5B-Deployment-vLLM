#!/usr/bin/env python3
"""
Local benchmarking script for GPU inference with advanced optimizations.
Tests various batch sizes, quantization methods, and GPU-specific optimizations.

Supported optimizations:
- FlashAttention via PyTorch SDPA
- torch.compile (reduce-overhead, max-autotune)
- BetterTransformer (fused attention kernels)
- CUDA Graphs (eliminate CPU overhead for fixed shapes)
- DeepSpeed Inference (optimized transformer kernels)
- BF16/FP16 precision modes
- Pre-tokenization (remove tokenization from timed section)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

# Optional imports
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


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
            print(f"  Warning: No samples found in any length category")
    
    elif num_samples and num_samples < len(df):
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


def pre_tokenize_data(tokenizer, prompts: List[str], max_length: int = 512, 
                      device: torch.device = None) -> List[Dict[str, torch.Tensor]]:
    """
    Pre-tokenize all prompts to remove tokenization from timed inference section.
    Returns list of tokenized batches ready for GPU inference.
    """
    print("\nPre-tokenizing data...")
    start = time.perf_counter()
    
    tokenized_samples = []
    for prompt in prompts:
        tokens = tokenizer(
            prompt,
            padding='max_length',  # Pad to max_length for CUDA graph compatibility
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        if device:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        tokenized_samples.append(tokens)
    
    elapsed = time.perf_counter() - start
    print(f"  Pre-tokenized {len(prompts)} samples in {elapsed:.2f}s")
    
    return tokenized_samples


def apply_optimizations(model, optimization: str, device: torch.device, 
                       dtype: torch.dtype = torch.float16) -> Tuple[Any, str]:
    """
    Apply specified optimization to the model.
    
    Args:
        model: The PyTorch model
        optimization: One of 'none', 'compile', 'compile-max', 'bettertransformer', 'deepspeed'
        device: Target device
        dtype: Model dtype (fp16 or bf16)
    
    Returns:
        Tuple of (optimized_model, optimization_info_string)
    """
    info = []
    
    if optimization == "none":
        info.append("No additional optimizations")
        return model, ", ".join(info)
    
    elif optimization == "compile":
        # torch.compile with reduce-overhead mode (faster compilation, good for inference)
        print("  Applying torch.compile (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")
        info.append("torch.compile(reduce-overhead)")
        return model, ", ".join(info)
    
    elif optimization == "compile-max":
        # torch.compile with max-autotune mode (slower compilation, maximum optimization)
        print("  Applying torch.compile (mode=max-autotune)...")
        model = torch.compile(model, mode="max-autotune")
        info.append("torch.compile(max-autotune)")
        return model, ", ".join(info)
    
    elif optimization == "bettertransformer":
        # BetterTransformer: fused attention and MLP kernels
        print("  Applying BetterTransformer...")
        try:
            model = model.to_bettertransformer()
            info.append("BetterTransformer")
        except Exception as e:
            print(f"    Warning: BetterTransformer failed: {e}")
            print(f"    Falling back to standard model")
            info.append("BetterTransformer (failed, using standard)")
        return model, ", ".join(info)
    
    elif optimization == "deepspeed":
        if not DEEPSPEED_AVAILABLE:
            print("  Warning: DeepSpeed not available, falling back to standard model")
            info.append("DeepSpeed (not available)")
            return model, ", ".join(info)
        
        print("  Applying DeepSpeed Inference...")
        try:
            # DeepSpeed inference with kernel injection
            ds_model = deepspeed.init_inference(
                model,
                mp_size=1,  # Single GPU
                dtype=dtype,
                replace_method="auto",
                replace_with_kernel_inject=True
            )
            info.append("DeepSpeed Inference")
            return ds_model, ", ".join(info)
        except Exception as e:
            print(f"    Warning: DeepSpeed init failed: {e}")
            print(f"    Falling back to standard model")
            info.append("DeepSpeed (failed)")
            return model, ", ".join(info)
    
    else:
        print(f"  Warning: Unknown optimization '{optimization}', using none")
        return model, "none"


def setup_cuda_graph(model, tokenizer, device: torch.device, batch_size: int,
                     max_length: int = 512) -> Tuple[torch.cuda.CUDAGraph, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Capture a CUDA graph for fixed batch size inference.
    
    Returns:
        Tuple of (cuda_graph, static_inputs, static_output)
    """
    print(f"  Capturing CUDA graph for batch_size={batch_size}...")
    
    # Create static input tensors with fixed shapes
    static_input_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
    static_attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
    
    # Warm-up runs (required before graph capture)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids=static_input_ids, attention_mask=static_attention_mask)
    torch.cuda.synchronize()
    
    # Capture the graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            static_output = model(input_ids=static_input_ids, attention_mask=static_attention_mask)
    
    static_inputs = {
        "input_ids": static_input_ids,
        "attention_mask": static_attention_mask
    }
    
    print(f"    CUDA graph captured successfully")
    return g, static_inputs, static_output


def initialize_model(model_id: str, quantization: str = "none", 
                    optimization: str = "none",
                    use_flash_attention: bool = False,
                    dtype: str = "fp16") -> tuple:
    """
    Initialize model and tokenizer with specified configuration.
    
    Args:
        model_id: HuggingFace model ID
        quantization: Quantization method (none, bitsandbytes)
        optimization: Optimization method (none, compile, compile-max, bettertransformer, deepspeed)
        use_flash_attention: Enable FlashAttention via PyTorch SDPA
        dtype: Data type (fp16, bf16)
    
    Returns:
        tuple: (model, tokenizer, device, model_dtype)
    """
    print(f"\nInitializing model...")
    print(f"  Model: {model_id}")
    print(f"  Quantization: {quantization}")
    print(f"  Optimization: {optimization}")
    print(f"  FlashAttention: {use_flash_attention}")
    print(f"  Dtype: {dtype}")
    
    start = time.perf_counter()
    
    # Set TF32 for any remaining FP32 ops (provides speedup on Ampere+)
    torch.set_float32_matmul_precision("high")
    
    # Determine torch dtype
    if dtype == "bf16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16
    
    try:
        # Enable optimized attention backends
        if use_flash_attention:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            print(f"  Enabled optimized SDPA backends (Flash + Memory-Efficient)")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with quantization
        if quantization == "bitsandbytes":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map={"": 0}
            )
        else:
            # No quantization - load with specified dtype
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                torch_dtype=model_dtype
            )
            model = model.to(device)
        
        # Ensure model knows the pad token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        
        # Apply optimization (compile, bettertransformer, deepspeed, etc.)
        model, opt_info = apply_optimizations(model, optimization, device, model_dtype)
        
        load_time = time.perf_counter() - start
        print(f"  Model loaded in {load_time:.2f}s")
        print(f"  Device: {device}")
        print(f"  Applied: {opt_info}")
        
        return model, tokenizer, device, model_dtype
        
    except Exception as e:
        print(f"  Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def benchmark_batch_size(model, tokenizer, device, prompts: List[str], 
                        batch_size: int, use_pretokenized: bool = False,
                        pretokenized_data: List[Dict] = None,
                        use_cuda_graph: bool = False,
                        max_length: int = 512) -> Dict[str, float]:
    """
    Run benchmark for a specific batch size.
    
    Args:
        model: The model to benchmark
        tokenizer: Tokenizer (only used if not pretokenized)
        device: Target device
        prompts: List of text prompts
        batch_size: Batch size for inference
        use_pretokenized: Whether to use pre-tokenized data
        pretokenized_data: Pre-tokenized samples
        use_cuda_graph: Whether to use CUDA graphs for inference
        max_length: Max sequence length (for CUDA graph)
    """
    num_samples = len(prompts)
    
    # Setup for CUDA graph if requested
    cuda_graph = None
    static_inputs = None
    static_output = None
    
    if use_cuda_graph:
        try:
            cuda_graph, static_inputs, static_output = setup_cuda_graph(
                model, tokenizer, device, batch_size, max_length
            )
        except Exception as e:
            print(f"    Warning: CUDA graph capture failed: {e}")
            print(f"    Falling back to standard inference")
            use_cuda_graph = False
    
    # Prepare batches
    if use_pretokenized:
        # Group pretokenized samples into batches
        batches = []
        for i in range(0, num_samples, batch_size):
            batch_tokens = pretokenized_data[i:i+batch_size]
            if len(batch_tokens) == batch_size:  # Only full batches for consistency
                # Stack tensors
                batch_input_ids = torch.cat([t['input_ids'] for t in batch_tokens], dim=0)
                batch_attention_mask = torch.cat([t['attention_mask'] for t in batch_tokens], dim=0)
                batches.append({
                    'input_ids': batch_input_ids,
                    'attention_mask': batch_attention_mask
                })
            else:
                # Handle last incomplete batch
                batch_input_ids = torch.cat([t['input_ids'] for t in batch_tokens], dim=0)
                batch_attention_mask = torch.cat([t['attention_mask'] for t in batch_tokens], dim=0)
                batches.append({
                    'input_ids': batch_input_ids,
                    'attention_mask': batch_attention_mask,
                    '_incomplete': True
                })
    else:
        batches = [prompts[i:i+batch_size] for i in range(0, num_samples, batch_size)]
    
    latencies = []
    total_samples = 0
    
    # Warmup run (especially important for torch.compile)
    print(f"    Warming up...")
    if use_pretokenized:
        warmup_batch = batches[0]
        with torch.no_grad():
            _ = model(input_ids=warmup_batch['input_ids'], 
                     attention_mask=warmup_batch['attention_mask'])
    else:
        warmup_inputs = tokenizer(
            prompts[:batch_size],
            padding='max_length' if use_cuda_graph else True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            _ = model(**warmup_inputs)
    
    torch.cuda.synchronize()
    
    start_total = time.perf_counter()
    
    for batch in batches:
        start_batch = time.perf_counter()
        
        if use_pretokenized:
            inputs = batch
            actual_batch_size = batch['input_ids'].shape[0]
        else:
            # Tokenize batch
            inputs = tokenizer(
                batch,
                padding='max_length' if use_cuda_graph else True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            actual_batch_size = len(batch)
        
        # Run inference
        if use_cuda_graph and cuda_graph is not None and actual_batch_size == batch_size:
            # Copy inputs to static buffers and replay graph
            static_inputs['input_ids'].copy_(inputs['input_ids'] if use_pretokenized else inputs.input_ids)
            static_inputs['attention_mask'].copy_(inputs['attention_mask'] if use_pretokenized else inputs.attention_mask)
            cuda_graph.replay()
            torch.cuda.synchronize()
        else:
            # Standard inference
            with torch.no_grad():
                if use_pretokenized:
                    outputs = model(input_ids=inputs['input_ids'], 
                                  attention_mask=inputs['attention_mask'])
                else:
                    outputs = model(**inputs)
            torch.cuda.synchronize()
        
        batch_latency = (time.perf_counter() - start_batch) * 1000  # Convert to ms
        latencies.append(batch_latency)
        total_samples += actual_batch_size
    
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
                   optimization: str,
                   batch_sizes: List[int],
                   num_samples: int = 1000,
                   balance_lengths: bool = True,
                   use_flash_attention: bool = False,
                   use_pretokenization: bool = False,
                   use_cuda_graph: bool = False,
                   dtype: str = "fp16",
                   output_dir: str = "./results/local_benchmarks") -> List[Dict]:
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 70)
    print(f"Starting Benchmarks")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Quantization: {quantization}")
    print(f"Optimization: {optimization}")
    print(f"FlashAttention: {use_flash_attention}")
    print(f"Pre-tokenization: {use_pretokenization}")
    print(f"CUDA Graphs: {use_cuda_graph}")
    print(f"Dtype: {dtype}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Number of samples: {num_samples}")
    print(f"Balance sequence lengths: {balance_lengths}")
    print("=" * 70)
    
    # Load test data
    prompts, true_labels, id2label = load_test_data(
        num_samples=num_samples,
        balance_lengths=balance_lengths
    )
    
    # Initialize model
    model, tokenizer, device, model_dtype = initialize_model(
        model_id, quantization, optimization, use_flash_attention, dtype
    )
    
    # Pre-tokenize if requested
    pretokenized_data = None
    if use_pretokenization:
        pretokenized_data = pre_tokenize_data(tokenizer, prompts, max_length=512, device=device)
    
    # Run benchmarks for each batch size
    results = []
    
    print(f"\n{'=' * 70}")
    print(f"Running Benchmarks")
    print(f"{'=' * 70}")
    
    for batch_size in batch_sizes:
        print(f"\n[Batch Size: {batch_size}]")
        
        metrics = benchmark_batch_size(
            model, tokenizer, device, prompts, batch_size,
            use_pretokenized=use_pretokenization,
            pretokenized_data=pretokenized_data,
            use_cuda_graph=use_cuda_graph
        )
        metrics['quantization'] = quantization
        metrics['optimization'] = optimization
        metrics['flash_attention'] = use_flash_attention
        metrics['pretokenization'] = use_pretokenization
        metrics['cuda_graph'] = use_cuda_graph
        metrics['dtype'] = dtype
        metrics['model_id'] = model_id
        
        results.append(metrics)
        
        # Print results
        print(f"  Throughput:    {metrics['throughput']:>7.2f} samples/s")
        print(f"  Avg Latency:   {metrics['avg_latency_ms']:>7.2f} ms")
        print(f"  P50 Latency:   {metrics['p50_latency_ms']:>7.2f} ms")
        print(f"  P95 Latency:   {metrics['p95_latency_ms']:>7.2f} ms")
        print(f"  P99 Latency:   {metrics['p99_latency_ms']:>7.2f} ms")
    
    # Build config name for output directory
    config_parts = [quantization]
    if optimization != "none":
        config_parts.append(optimization.replace("-", "_"))
    if use_flash_attention:
        config_parts.append("flash")
    if use_pretokenization:
        config_parts.append("pretok")
    if use_cuda_graph:
        config_parts.append("cudagraph")
    if dtype != "fp16":
        config_parts.append(dtype)
    
    config_name = "_".join(config_parts)
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
        description="Benchmark GPU inference with various optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline FP16 (no optimizations)
  python scripts/benchmark_local.py --quantization none

  # FlashAttention enabled
  python scripts/benchmark_local.py --quantization none --use-flash-attention

  # torch.compile with max-autotune
  python scripts/benchmark_local.py --quantization none --optimization compile-max

  # BetterTransformer with FlashAttention
  python scripts/benchmark_local.py --quantization none --optimization bettertransformer --use-flash-attention

  # CUDA Graphs for maximum throughput (fixed batch sizes)
  python scripts/benchmark_local.py --quantization none --use-cuda-graph --use-pretokenization

  # BF16 precision
  python scripts/benchmark_local.py --quantization none --dtype bf16

  # INT8 quantization with DeepSpeed
  python scripts/benchmark_local.py --quantization bitsandbytes --optimization deepspeed

  # Full optimization stack
  python scripts/benchmark_local.py --quantization none --optimization compile-max \\
      --use-flash-attention --use-pretokenization --dtype bf16
"""
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
        choices=["none", "bitsandbytes"],
        help="Quantization method (none=FP16/BF16, bitsandbytes=INT8)"
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="none",
        choices=["none", "compile", "compile-max", "bettertransformer", "deepspeed"],
        help="Optimization method to apply"
    )
    parser.add_argument(
        "--use-flash-attention",
        action="store_true",
        default=False,
        help="Enable FlashAttention optimization via PyTorch SDPA"
    )
    parser.add_argument(
        "--use-pretokenization",
        action="store_true",
        default=False,
        help="Pre-tokenize data to remove tokenization from timed section"
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        default=False,
        help="Use CUDA graphs for inference (requires fixed batch sizes)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Model precision (fp16 or bf16)"
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
        "--output-dir",
        type=str,
        default="./results/local_benchmarks",
        help="Output directory for results"
    )
    # Legacy arguments for backwards compatibility
    parser.add_argument(
        "--inference-engine",
        type=str,
        default=None,
        help="[DEPRECATED] Use --optimization instead"
    )
    parser.add_argument(
        "--no-optimizations",
        action="store_true",
        default=False,
        help="[DEPRECATED] Run without optimizations (same as --optimization none)"
    )
    
    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.inference_engine:
        print(f"Warning: --inference-engine is deprecated. Use --optimization instead.")
        if args.inference_engine == "onnx":
            print("Note: ONNX Runtime has been removed. Using standard PyTorch inference.")
    
    if args.no_optimizations:
        print("Warning: --no-optimizations is deprecated. Using --optimization none.")
        args.optimization = "none"
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Run benchmarks
    results = run_benchmarks(
        model_id=args.model_id,
        quantization=args.quantization,
        optimization=args.optimization,
        batch_sizes=batch_sizes,
        num_samples=args.num_samples,
        balance_lengths=args.balance_lengths,
        use_flash_attention=args.use_flash_attention,
        use_pretokenization=args.use_pretokenization,
        use_cuda_graph=args.use_cuda_graph,
        dtype=args.dtype,
        output_dir=args.output_dir
    )
    
    print("\nBenchmarking complete!")
    
    # Build config name for display
    config_parts = [args.quantization]
    if args.optimization != "none":
        config_parts.append(args.optimization.replace("-", "_"))
    if args.use_flash_attention:
        config_parts.append("flash")
    if args.use_pretokenization:
        config_parts.append("pretok")
    if args.use_cuda_graph:
        config_parts.append("cudagraph")
    if args.dtype != "fp16":
        config_parts.append(args.dtype)
    config_name = "_".join(config_parts)
    
    print(f"\nNext steps:")
    print(f"  1. Review results in: {args.output_dir}/{config_name}/")
    print(f"  2. Test other configurations")
    print(f"  3. Compare results: python scripts/compare_results.py")
    print(f"\nSuggested optimization combinations to try:")
    print(f"  - Baseline: --quantization none")
    print(f"  - FlashAttn: --quantization none --use-flash-attention")
    print(f"  - Compiled: --quantization none --optimization compile-max --use-flash-attention")
    print(f"  - BetterTF: --quantization none --optimization bettertransformer --use-flash-attention")
    print(f"  - CUDA Graph: --quantization none --use-cuda-graph --use-pretokenization")
    print(f"  - INT8: --quantization bitsandbytes --use-flash-attention")


if __name__ == "__main__":
    main()
