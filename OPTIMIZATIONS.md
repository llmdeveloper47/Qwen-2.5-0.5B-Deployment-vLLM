# GPU Inference Optimizations for Qwen2 Classification

This document summarizes available optimization techniques for reducing inference latency on NVIDIA GPUs.

## Implemented Optimizations

### 1. FlashAttention (NEW)
**Status:** Implemented via PyTorch 2.0+ SDPA backend  
**Expected Speedup:** 1.5-2x faster attention computation  
**Best For:** Long sequences (>512 tokens)  
**Memory:** Reduced memory usage for attention

**How to use:**
```bash
python scripts/benchmark_local.py \
  --quantization none \
  --use-flash-attention \
  --batch-sizes 1,8,16,32 \
  --num-samples 1000
```

**How it works:**
- Enables PyTorch's optimized scaled_dot_product_attention
- Uses FlashAttention v2 CUDA kernels automatically
- O(N) memory complexity vs O(N^2) for standard attention
- Achieves 50-70% of theoretical FLOPs on A100

### 2. BitsAndBytes INT8 Quantization
**Status:** Tested and working  
**Speedup:** 2.5x faster at batch size 32  
**Memory:** 50% reduction  
**Accuracy:** Minimal degradation

**Results:**
- Batch 32: 107.21 samples/s vs 43.74 samples/s (FP16)
- P95 Latency: 301ms vs 733ms (FP16)

**How to use:**
```bash
python scripts/benchmark_local.py \
  --quantization bitsandbytes \
  --batch-sizes 1,8,16,32 \
  --num-samples 1000
```

### 3. Mixed Precision (FP16)
**Status:** Default baseline  
**How:** Models loaded with torch.float16  
**Benefit:** 2x memory reduction, utilizes A100 tensor cores (312 TFLOPs)

## Tested But Removed

### ONNX Runtime (REMOVED)
**Reason:** Qwen2 architecture not supported for sequence classification  
**Error:** No built-in ONNX export configuration  
**Alternative:** Use FlashAttention + BitsAndBytes instead

## Available But Not Yet Implemented

### 1. CUDA Graphs
**Potential Speedup:** 1.3-1.5x (eliminates CPU overhead)  
**Best For:** Static batch sizes in production  
**Complexity:** Moderate  
**Requirements:** Fixed input shapes

**How it works:**
- Captures entire model execution as a static graph
- Single kernel launch replays all operations
- Removes Python and CUDA driver overhead
- Improves p95/p99 latency consistency

**Implementation notes:**
- Requires pre-allocated static tensors
- Must capture separate graph for each batch size
- Works best with padding to fixed length (512 tokens)

### 2. DeepSpeed Inference
**Potential Speedup:** 2x (according to Microsoft research)  
**Best For:** Production deployment  
**Complexity:** High  
**Requirements:** deepspeed library

**How it works:**
- Replaces transformer layers with optimized kernels
- Fuses QKV projections, attention, and MLP operations
- Hand-tuned CUDA kernels for specific architectures

**Implementation example:**
```python
import deepspeed
ds_model = deepspeed.init_inference(
    model, 
    mp_size=1, 
    dtype=torch.half,
    replace_method="auto",
    replace_with_kernel_inject=True
)
```

### 3. Static Padding for torch.compile
**Potential Speedup:** 1.2-1.5x (with no recompilation)  
**Best For:** Production with known max length  
**Complexity:** Low

**How it works:**
- Pad all inputs to max_length=512
- torch.compile optimizes for single shape
- No recompilation overhead

**Why not used in benchmarking:**
- Variable sequence lengths (524-22229 chars) representative of real data
- Padding to 22229 would waste compute
- Benchmarking should reflect realistic conditions

### 4. BFloat16 (BF16)
**Potential Speedup:** Same as FP16 on A100  
**Benefit:** Larger dynamic range, better numerical stability  
**Implementation:** Change dtype to torch.bfloat16

### 5. TensorFloat32 (TF32)
**Status:** Automatically used on A100 for FP32 ops  
**Can enable explicitly:**
```python
torch.set_float32_matmul_precision('high')
```

## Optimization Strategy by Use Case

### Low Latency (Batch Size 1)
**Recommendation:** FP16 + FlashAttention  
**Why:** Minimal overhead, best single-sample performance  
**Expected:** 30-40 samples/s, <35ms P95 latency

### Balanced (Batch Size 8-16)
**Recommendation:** BitsAndBytes INT8 + FlashAttention  
**Why:** Good throughput, manageable latency  
**Expected:** 80-100 samples/s, 130-180ms P95 latency

### High Throughput (Batch Size 32)
**Recommendation:** BitsAndBytes INT8  
**Why:** Best throughput, memory efficient  
**Expected:** 107+ samples/s, 301ms P95 latency

## Testing Matrix

```
Configuration        | Batch 1  | Batch 8  | Batch 16 | Batch 32 |
---------------------|----------|----------|----------|----------|
FP16 Baseline        | 38.71/s  | 42.14/s  | 42.21/s  | 43.74/s  |
BitsAndBytes INT8    | 9.39/s   | 61.05/s  | 88.75/s  | 107.21/s |
FP16 + FlashAttn     | TBD      | TBD      | TBD      | TBD      |
INT8 + FlashAttn     | TBD      | TBD      | TBD      | TBD      |
```

## Commands Reference

### Test FlashAttention with FP16
```bash
python scripts/benchmark_local.py \
  --quantization none \
  --use-flash-attention \
  --batch-sizes 1,8,16,32 \
  --num-samples 1000
```

### Test FlashAttention with INT8
```bash
python scripts/benchmark_local.py \
  --quantization bitsandbytes \
  --use-flash-attention \
  --batch-sizes 1,8,16,32 \
  --num-samples 1000
```

### Compare All Configurations
```bash
python scripts/compare_results.py \
  --quantizations none,bitsandbytes,none_flash,bitsandbytes_flash
```

## Next Steps

1. Test FlashAttention with FP16 and INT8
2. Compare performance gains
3. Choose best configuration for deployment
4. Optionally: Implement CUDA Graphs for production (static batch sizes)
5. Optionally: Try DeepSpeed Inference for maximum throughput

## References

- FlashAttention: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
- CUDA Graphs: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- DeepSpeed Inference: https://www.deepspeed.ai/inference/

