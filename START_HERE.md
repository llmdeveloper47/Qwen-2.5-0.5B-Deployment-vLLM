# START HERE - Project Quick Reference

This document provides a clear, sequential guide to get started. **Follow the steps in order.**

## Important Note: Optimized Transformers (Not vLLM)

This project uses **optimized Transformers** for fast classification inference, not vLLM. vLLM does not support classification models. We achieve comparable or better performance using:
- BetterTransformer optimization
- torch.compile (PyTorch 2.0+)
- FP16 precision
- Static batching
- Optional quantization (BitsAndBytes, AWQ, GPTQ)

## Documentation Structure (Simplified)

**Main Documentation:**
- **README.md** - Complete guide with all instructions (read this first)
- **CONTRIBUTING.md** - Guidelines for contributing code
- **experiments/EXPERIMENT_LOG.md** - Template for tracking your experiment results

All other documentation has been consolidated into README.md for simplicity.

---

## Sequential Execution Guide

### PHASE 1: Initial Setup (30-45 minutes)

**Step 1:** Install prerequisites
- Python 3.10+, Git, Docker
- Create RunPod account and get API key
- See README.md "Step 1: Prerequisites"

**Step 2:** Setup environment
```bash
git clone https://github.com/llmdeveloper47/Qwen-2.5-0.5B-Inference-Deployment.git
cd Qwen-2.5-0.5B-Inference-Deployment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --ignore-installed blinker
```

**Step 3:** Configure credentials
```bash
cp env.example .env
nano .env  # Add your RUNPOD_API_KEY
```

**Step 4:** Download model
```bash
pip install hf_transfer
python scripts/download_model_safe.py
-- python scripts/download_model.py
```

---

### PHASE 2: Local Testing - Optional (1-2 hours, requires GPU)

**Step 5:** Run local benchmarks (Optional)
```bash
# Uninstall current PyTorch ( torch 2.9 is unstable on Runpod )
pip uninstall torch torchvision torchaudio -y

# Install stable PyTorch 2.4.0
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# to sample sequences by length, use below command
python scripts/benchmark_local.py --quantization none --batch-sizes 1,8,16,64 --num-samples 1000

# to sample sequences randomly, use below command
python scripts/benchmark_local.py \
  --quantization none \
  --batch-sizes 1,8,16,32 \
  --num-samples 1000 \
  --no-balance-lengths
```

**Step 6:** Test handler (Optional)
```bash
python scripts/test_local_handler.py
```

**Step 7:** ONNX Runtime Optimization Benchmarks (Optional, Advanced)

ONNX Runtime provides significant speedup through graph optimizations and operator fusion. Expected performance gains: 1.5-3x faster than standard PyTorch.

```bash
# Install ONNX Runtime dependencies
pip install optimum[onnxruntime-gpu] onnx onnxruntime-gpu

# Run ONNX Runtime benchmark
python scripts/benchmark_local.py \
  --quantization none \
  --inference-engine onnx \
  --batch-sizes 1,8,16,32 \
  --num-samples 1000 \
  --no-optimizations

# Expected outcomes:
# - First run: Converts model to ONNX (adds 30-60s)
# - ONNX model cached to ./models_onnx/none/
# - Results saved to results/local_benchmarks/none_onnx/
# - 2-3x speedup at batch size 1
# - 1.5-2x speedup at larger batches
```

**How ONNX Runtime Works:**
- Converts PyTorch model to ONNX graph format
- Applies graph optimizations (operator fusion, constant folding)
- Uses optimized CUDA kernels
- Reduces CPU overhead and kernel launch time
- Cached for reuse in future runs

**Compare all methods:**
```bash
python scripts/compare_results.py --quantizations none,bitsandbytes,none_onnx
```

**Note:** If you don't have a local GPU, skip to Phase 3.

---

### PHASE 3: Deployment (1 hour)

**Step 7:** Build Docker image
```bash
docker build -t intent-classification-transformers:latest .
docker tag intent-classification-transformers:latest ghcr.io/llmdeveloper47/qwen-2.5-0.5b-inference-deployment:latest
```

**Step 8:** Push to GitHub Container Registry
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u llmdeveloper47 --password-stdin
docker push ghcr.io/llmdeveloper47/qwen-2.5-0.5b-inference-deployment:latest
```

**Step 9:** Create RunPod endpoint
- Login to RunPod console
- Create new endpoint with A100 GPU (or L40S, A10, etc.)
- Use image: `ghcr.io/llmdeveloper47/qwen-2.5-0.5b-inference-deployment:latest`
- Set environment variables:
  - MODEL_NAME=codefactory4791/intent-classification-qwen
  - MAX_MODEL_LEN=512
  - QUANTIZATION=none
  - TRUST_REMOTE_CODE=true
  - BATCH_SIZE=16
  - USE_BETTER_TRANSFORMER=true
  - USE_COMPILE=true
- Copy endpoint ID to your `.env` file

**Step 10:** Test endpoint
```bash
python scripts/test_endpoint.py --endpoint-id $RUNPOD_ENDPOINT_ID --api-key $RUNPOD_API_KEY
```

---

### PHASE 4: Experiments (4-6 hours for full suite)

**Step 11:** Run FP16 baseline experiments with optimizations
```bash
# Ensure QUANTIZATION=none and USE_BETTER_TRANSFORMER=true in RunPod endpoint
for bs in 1 4 8 16 32; do
  python scripts/test_endpoint.py \
    --endpoint-id $RUNPOD_ENDPOINT_ID \
    --api-key $RUNPOD_API_KEY \
    --latency-test \
    --batch-size $bs \
    --iterations 20 \
    --output results/experiments/none/batch_${bs}.json
  sleep 30
done
```

**Step 12:** Run BitsAndBytes experiments
```bash
# Update QUANTIZATION=bitsandbytes and USE_BETTER_TRANSFORMER=false in RunPod endpoint
# Wait 2-3 minutes for restart
# Then run same tests as Step 11, saving to results/experiments/bitsandbytes/
```

**Step 13:** (Optional) Run AWQ experiments
- Requires pre-quantized model
- Update QUANTIZATION=awq
- Run same test procedure

**Step 14:** (Optional) Run GPTQ experiments
- Requires pre-quantized model  
- Update QUANTIZATION=gptq
- Run same test procedure

---

### PHASE 5: Analysis (30-60 minutes)

**Step 15:** Generate summary
```bash
python scripts/summarize_results.py --results-dir ./results
```

**Step 16:** Analyze and visualize
```bash
python scripts/analyze_results.py --results-dir ./results
```

**Step 17:** Generate report
```bash
python scripts/generate_report.py --results-dir ./results --output results/report.pdf
```

**Step 18:** Review results
```bash
cat results/analysis/comparison_table.csv
jupyter notebook experiments/analysis/comparison.ipynb
```

---

### PHASE 2B: Advanced Optimization Testing (Optional)

**Available Inference Engines:**

1. **Standard PyTorch (transformers)** - Default, well-tested
2. **ONNX Runtime (onnx)** - Graph optimizations, 1.5-3x faster

**Available Quantization Methods:**

1. **None (FP16)** - Baseline, best accuracy
2. **BitsAndBytes (INT8)** - 50% memory reduction, 2-2.5x faster at large batches

**Hybrid Configurations:**

Test different combinations:
```bash
# FP16 + Standard PyTorch (baseline)
python scripts/benchmark_local.py \
  --quantization none --inference-engine transformers \
  --batch-sizes 1,8,16,32 --num-samples 1000 --no-optimizations

# FP16 + ONNX Runtime (optimized graph)
python scripts/benchmark_local.py \
  --quantization none --inference-engine onnx \
  --batch-sizes 1,8,16,32 --num-samples 1000 --no-optimizations

# INT8 + Standard PyTorch (quantized)
python scripts/benchmark_local.py \
  --quantization bitsandbytes --inference-engine transformers \
  --batch-sizes 1,8,16,32 --num-samples 1000 --no-optimizations

# Compare all configurations
python scripts/compare_results.py
```

---

## Quick Start Options

### Option 1: Fastest Path (Deploy Only)

If you just want to deploy without experiments:

```bash
# 1. Setup (15 min)
git clone https://github.com/llmdeveloper47/Qwen-2.5-0.5B-Inference-Deployment.git
cd Qwen-2.5-0.5B-Deployment-vLLM
./quickstart.sh

# 2. Build & Push (15 min)
make build-push

# 3. Deploy via RunPod console (30 min)
# Follow README.md Step 7

# 4. Test (5 min)
make test-endpoint
```

Total: ~1 hour

### Option 2: Quick Experiment (Recommended)

Test 2 quantization methods with 3 batch sizes:

```bash
# After completing Option 1 deployment:

# Test FP16 (1, 8, 16 batch sizes)
for bs in 1 8 16; do
  python scripts/test_endpoint.py \
    --endpoint-id $RUNPOD_ENDPOINT_ID \
    --api-key $RUNPOD_API_KEY \
    --latency-test \
    --batch-size $bs \
    --iterations 20 \
    --output results/experiments/none/batch_${bs}.json
  sleep 30
done

# Update endpoint to QUANTIZATION=bitsandbytes
# Wait 3 minutes, then test same batch sizes

# Analyze
python scripts/analyze_results.py --results-dir ./results
```

Total: ~2-3 hours, Cost: ~$5

### Option 3: Complete Experiment Suite

Follow PHASE 1-5 above for full analysis.

Total: ~10-12 hours, Cost: ~$18-25

---

## Key Files Reference

### Scripts You Will Use

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `download_model.py` | Download model | Step 4 (once) |
| `download_model_safe.py` | Memory-efficient download | Step 4 (recommended) |
| `benchmark_local.py` | Local performance testing | Step 5, 7 (optional) |
| `compare_results.py` | Compare benchmark results | After benchmarks |
| `test_local_handler.py` | Test handler | Step 6 (optional) |
| `test_endpoint.py` | Test RunPod endpoint | Steps 10, 11-14 |
| `analyze_results.py` | Analyze results | Step 16 |
| `generate_report.py` | Create PDF | Step 17 |

### Configuration Files

| File | Quantization | Use When |
|------|--------------|----------|
| `configs/fp16_baseline.json` | FP16 (none) | Baseline experiments |
| `configs/bitsandbytes_int8.json` | INT8 | BitsAndBytes experiments |
| `configs/awq_4bit.json` | AWQ | AWQ experiments |
| `configs/gptq_4bit.json` | GPTQ | GPTQ experiments |

Use these to see the exact environment variables needed for each configuration.

---

## Expected Results

After completing benchmarks, you will have:

### Performance Metrics (batch_size=16)

| Configuration | Inference Engine | P95 Latency | Throughput | Speedup vs FP16 |
|--------------|------------------|-------------|------------|-----------------|
| FP16 Baseline | Transformers | ~379ms | ~42 samples/s | 1.0x (baseline) |
| BitsAndBytes INT8 | Transformers | ~185ms | ~89 samples/s | 2.1x faster |
| FP16 + ONNX | ONNX Runtime | ~150-200ms | ~70-100 samples/s | 1.7-2.4x faster |
| INT8 + ONNX | ONNX Runtime | ~100-140ms | ~110-150 samples/s | 2.6-3.6x faster |

Note: Actual results depend on GPU, sequence lengths, and PyTorch version.

### Analysis Outputs

- `results/local_benchmarks/none/` - FP16 baseline results
- `results/local_benchmarks/bitsandbytes/` - INT8 quantization results
- `results/local_benchmarks/none_onnx/` - ONNX Runtime FP16 results
- `results/comparison.csv` - Side-by-side comparison
- `results/analysis/` - Detailed analysis and visualizations

---

## Common Issues

### "CUDA out of memory"

Solution: Reduce BATCH_SIZE to 8 or use quantization (bitsandbytes)

### "Endpoint timeout"

Cause: Cold start (normal for first request, takes 30-60s for model loading)

### "BetterTransformer not applying"

Cause: Model architecture may not support BetterTransformer with quantization

Solution: Set USE_BETTER_TRANSFORMER=false when using quantization

### "torch.compile fails"

Cause: Some quantization methods are incompatible with torch.compile

Solution: SET USE_COMPILE=false when using quantization

### "PyTorch 2.9 memory allocation errors"

Cause: PyTorch 2.9.x has known bugs with memory allocation

Solution: Downgrade to PyTorch 2.4.0 (see Phase 2, Step 5)

### "ONNX Runtime not found"

Cause: optimum[onnxruntime-gpu] not installed

Solution: pip install optimum[onnxruntime-gpu] onnx onnxruntime-gpu

### "ONNX conversion slow or fails"

Cause: First-time conversion takes 30-60s

Solution: Be patient. ONNX model is cached for future runs

### "AWQ/GPTQ not working"

Cause: Requires pre-quantized model or specific setup

Solution: Focus on FP16, ONNX Runtime, and BitsAndBytes quantization

**For detailed troubleshooting, see README.md "Troubleshooting" section**

---

## Next Steps

1. Read **README.md** for complete details on each step
2. Run **Phase 1** to setup environment
3. Run **Phase 3** to deploy to RunPod
4. Run **Phase 4** to execute experiments
5. Run **Phase 5** to analyze results

---

**Repository:** https://github.com/llmdeveloper47/Qwen-2.5-0.5B-Inference-Deployment  
**Model:** https://huggingface.co/codefactory4791/intent-classification-qwen

