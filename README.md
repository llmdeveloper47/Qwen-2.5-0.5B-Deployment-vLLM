# Qwen 2.5-0.5B Intent Classification on RunPod with vLLM

Complete deployment pipeline for intent classification model testing quantization methods and batch sizes on RunPod A100 GPU.

## Project Overview

This repository provides an end-to-end solution for:

1. Deploying the fine-tuned model `codefactory4791/intent-classification-qwen` on RunPod
2. Testing 4 quantization methods: FP16, BitsAndBytes (INT8), AWQ (4-bit), GPTQ (4-bit)  
3. Benchmarking 5 batch sizes: 1, 4, 8, 16, 32
4. Measuring latency (P50, P95, P99) and throughput for each configuration
5. Identifying optimal deployment settings for production use

**Model Details:**
- Base: Qwen 2.5-0.5B-Instruct
- Task: Intent classification (23 categories)
- Accuracy: 92% on test set
- Model URL: https://huggingface.co/codefactory4791/intent-classification-qwen

**Target Hardware:** NVIDIA A100 40GB GPU on RunPod Serverless

---

## Table of Contents

**PART 1: SETUP**
- [Prerequisites](#step-1-prerequisites)
- [Environment Setup](#step-2-environment-setup)
- [Model Download](#step-3-download-and-verify-model)

**PART 2: LOCAL TESTING (Optional)**
- [Local Benchmarking](#step-4-optional-local-benchmarking)
- [Test Handler Locally](#step-5-optional-test-handler-locally)

**PART 3: DEPLOYMENT**
- [Build Docker Image](#step-6-build-docker-image)
- [Deploy to RunPod](#step-7-deploy-to-runpod)
- [Test Deployed Endpoint](#step-8-test-deployed-endpoint)

**PART 4: EXPERIMENTS**
- [Run Quantization Experiments](#step-9-run-quantization-experiments)
- [Load Testing](#step-10-load-testing-with-locust)
- [Analyze Results](#step-11-analyze-results)

**APPENDICES**
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [API Usage Examples](#api-usage-examples)

---

# PART 1: SETUP

## Step 1: Prerequisites

### Required Accounts

**RunPod Account:**
1. Sign up at https://www.runpod.io/
2. Add credits (minimum $50 recommended for experiments)
3. Generate API key:
   - Go to Settings → API Keys
   - Click "Create API Key"
   - Save the key securely

**GitHub Account:**
1. Fork or clone repository: https://github.com/llmdeveloper47/Qwen-Qwen2.5-0.5B-Deployment-vLLM
2. Generate Personal Access Token:
   - Settings → Developer settings → Personal access tokens
   - Create token with `repo` and `write:packages` scopes
   - Save the token

### Required Software

**For All Users:**
- Python 3.10 or higher
- Git
- Docker 20.10+

**For Local GPU Testing (Optional):**
- NVIDIA GPU with 16GB+ VRAM
- CUDA Toolkit 12.1+
- NVIDIA drivers

**Installation Commands:**

Ubuntu/Debian:
```bash
# Python and Git
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip git

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# CUDA (for local GPU testing)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

macOS:
```bash
brew install python@3.10 git
# Download Docker Desktop from https://www.docker.com/products/docker-desktop
```

---

## Step 2: Environment Setup

### Clone Repository

```bash
git clone https://github.com/llmdeveloper47/Qwen-Qwen2.5-0.5B-Deployment-vLLM.git
cd Qwen-Qwen2.5-0.5B-Deployment-vLLM
```

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- vllm (inference engine)
- torch (PyTorch with CUDA)
- transformers (HuggingFace)
- datasets (data loading)
- locust (load testing)
- Analysis tools (pandas, matplotlib, seaborn)

### Configure Credentials

```bash
cp env.example .env
nano .env  # or use any text editor
```

Edit `.env` and add your credentials:
```
RUNPOD_API_KEY=your-api-key-here
RUNPOD_ENDPOINT_ID=your-endpoint-id-here
```

Note: You will get the RUNPOD_ENDPOINT_ID after creating the endpoint in Step 7.

---

## Step 3: Download and Verify Model

```bash
python scripts/download_model.py
```

**Expected Output:**
```
======================================================================
Downloading Model: codefactory4791/intent-classification-qwen
======================================================================

[1/3] Downloading tokenizer...
  ✓ Tokenizer downloaded successfully

[2/3] Downloading model...
  ✓ Model downloaded successfully
  - Number of parameters: 494,033,920
  - Number of labels: 23

[3/3] Verifying configuration...
  ✓ Model size: 1,234.56 MB
  ✓ Number of classes: 23

[Testing] Running quick inference test...
  ✓ Test inference successful

======================================================================
✓ Model download and verification complete!
======================================================================
```

This creates `models/model_info.json` with model metadata.

---

# PART 2: LOCAL TESTING (Optional)

Note: Local testing requires an NVIDIA GPU. Skip to Part 3 if you don't have a GPU.

## Step 4 (Optional): Local Benchmarking

Test different quantization methods locally before deploying:

### Test FP16 Baseline

```bash
python scripts/benchmark_local.py \
  --quantization none \
  --batch-sizes 1,4,8,16,32 \
  --num-samples 1000
```

### Test BitsAndBytes INT8

```bash
python scripts/benchmark_local.py \
  --quantization bitsandbytes \
  --batch-sizes 1,4,8,16,32 \
  --num-samples 1000
```

### Quick Test (Recommended for First Run)

```bash
python scripts/benchmark_local.py \
  --quantization none \
  --batch-sizes 1,8,16 \
  --num-samples 100
```

**Expected Output:**
```
======================================================================
Starting Benchmarks
======================================================================
Model: codefactory4791/intent-classification-qwen
Quantization: none
Batch sizes: [1, 8, 16]
======================================================================

Running Benchmarks
======================================================================

[Batch Size: 1]
  Throughput:      12.45 samples/s
  Avg Latency:     80.32 ms
  P95 Latency:     95.21 ms

[Batch Size: 8]
  Throughput:      64.15 samples/s
  Avg Latency:    124.68 ms
  P95 Latency:    145.32 ms

Summary Table:
Batch Size   Throughput      Avg Latency     P95 Latency
1            12.45           80.32           95.21
8            64.15           124.68          145.32
```

Results saved to `results/local_benchmarks/none/benchmark_results.json`

---

## Step 5 (Optional): Test Handler Locally

Before deployment, verify the RunPod handler works correctly:

```bash
python scripts/test_local_handler.py
```

**Expected Output:**
```
======================================================================
Testing RunPod Handler Locally
======================================================================

======================================================================
Test 1: Single Prompt
======================================================================
✓ Single prompt test passed

======================================================================
Test 2: Batch Prompts
======================================================================
✓ Batch prompts test passed

======================================================================
Test 3: Error Handling
======================================================================
✓ Error handling tests passed

======================================================================
✓ ALL TESTS PASSED
======================================================================
```

---

# PART 3: DEPLOYMENT

## Step 6: Build Docker Image

### Build Image

```bash
docker build -t intent-classification-vllm:latest .
```

This takes 10-15 minutes as it pre-downloads the model.

### Tag for GitHub Container Registry

```bash
docker tag intent-classification-vllm:latest \
  ghcr.io/llmdeveloper47/qwen-qwen2.5-0.5b-deployment-vllm:latest
```

### Push to Registry

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u llmdeveloper47 --password-stdin

# Push image
docker push ghcr.io/llmdeveloper47/qwen-qwen2.5-0.5b-deployment-vllm:latest
```

**Alternative: Use Makefile**
```bash
make build-push
```

---

## Step 7: Deploy to RunPod

### Create Endpoint via Console

1. **Login** to RunPod Console: https://www.runpod.io/console/serverless

2. **Create New Endpoint:**
   - Click "+ New Endpoint"
   - Name: `intent-classification-qwen-fp16`
   - GPU: Select "A100-SXM4-40GB"
   
3. **Container Configuration:**
   - Container Image: `ghcr.io/llmdeveloper47/qwen-qwen2.5-0.5b-deployment-vllm:latest`
   - Container Disk: 20 GB
   - Volume Disk: 0 GB

4. **Environment Variables** (click "Advanced" → "Public Environment Variables"):
   ```
   MODEL_NAME=codefactory4791/intent-classification-qwen
   MAX_NUM_SEQS=16
   MAX_MODEL_LEN=512
   QUANTIZATION=none
   DTYPE=auto
   GPU_MEMORY_UTILIZATION=0.95
   TRUST_REMOTE_CODE=true
   ENFORCE_EAGER=true
   ```

5. **Scaling Configuration:**
   - Idle Timeout: 60 seconds
   - Min Workers: 0
   - Max Workers: 1

6. **Deploy:**
   - Click "Deploy"
   - Wait for status to show "Running" (2-3 minutes)

7. **Save Endpoint ID:**
   - Copy the Endpoint ID from the details page
   - Add to your `.env` file:
     ```
     RUNPOD_ENDPOINT_ID=your-endpoint-id-here
     ```

---

## Step 8: Test Deployed Endpoint

### Basic Functionality Test

```bash
python scripts/test_endpoint.py \
  --endpoint-id $RUNPOD_ENDPOINT_ID \
  --api-key $RUNPOD_API_KEY
```

**Expected Output:**
```
======================================================================
Testing RunPod Endpoint
======================================================================
Sending request to: https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run
Number of prompts: 3
Response received in 0.156s
Status code: 200

✓ Response validation passed

======================================================================
Classification Results
======================================================================

[Result 1]
  Prompt: Book me a flight to San Francisco next Tuesday...
  Predicted Class: 18
  Confidence: 0.8523
  
✓ All endpoint tests passed!
```

### Test with Custom Prompts

```bash
python scripts/test_endpoint.py \
  --endpoint-id $RUNPOD_ENDPOINT_ID \
  --api-key $RUNPOD_API_KEY \
  --prompts \
    "Book me a flight to New York" \
    "Order a large pepperoni pizza" \
    "Play some rock music"
```

### Using cURL

```bash
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompts": ["Book me a flight to San Francisco"]
    }
  }' \
  https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run
```

---

# PART 4: EXPERIMENTS

## Step 9: Run Quantization Experiments

This is the main goal: systematically test all quantization methods and batch sizes to measure latency and throughput.

### Experiment Matrix

You will test 20 configurations total:

| Quantization | Batch Sizes | Total Configs |
|--------------|-------------|---------------|
| none (FP16) | 1, 4, 8, 16, 32 | 5 |
| bitsandbytes (INT8) | 1, 4, 8, 16, 32 | 5 |
| awq (4-bit) | 1, 4, 8, 16, 32 | 5 |
| gptq (4-bit) | 1, 4, 8, 16, 32 | 5 |

**Total:** 20 configurations

### Method A: Automated Experiments (Requires Manual Endpoint Updates)

The `run_experiments.py` script provides structure, but you must manually update the RunPod endpoint for each quantization method.

**Note:** The script will prompt you to update the endpoint configuration between quantization methods.

```bash
python scripts/run_experiments.py \
  --endpoint-id $RUNPOD_ENDPOINT_ID \
  --api-key $RUNPOD_API_KEY \
  --quantization-methods none bitsandbytes \
  --batch-sizes 1,4,8,16,32 \
  --iterations 20 \
  --num-samples 1000
```

### Method B: Manual Step-by-Step (Recommended for Understanding)

#### Experiment 1: FP16 Baseline

**1. Ensure endpoint is configured with:**
```
QUANTIZATION=none
```

**2. Run tests for each batch size:**
```bash
for BATCH_SIZE in 1 4 8 16 32; do
  python scripts/test_endpoint.py \
    --endpoint-id $RUNPOD_ENDPOINT_ID \
    --api-key $RUNPOD_API_KEY \
    --latency-test \
    --batch-size $BATCH_SIZE \
    --iterations 20 \
    --output results/experiments/none/batch_${BATCH_SIZE}.json
  
  echo "Completed batch size $BATCH_SIZE"
  sleep 30
done
```

**3. Review results:**
```bash
cat results/experiments/none/batch_16.json
```

#### Experiment 2: BitsAndBytes INT8

**1. Update RunPod endpoint:**
- Go to RunPod Console → Your Endpoint → Manage → Edit Endpoint
- Change environment variable: `QUANTIZATION=bitsandbytes`
- Save and wait 2-3 minutes for restart

**2. Verify endpoint is ready:**
```bash
python scripts/test_endpoint.py \
  --endpoint-id $RUNPOD_ENDPOINT_ID \
  --api-key $RUNPOD_API_KEY \
  --prompts "warmup request"
```

**3. Run tests for each batch size:**
```bash
for BATCH_SIZE in 1 4 8 16 32; do
  python scripts/test_endpoint.py \
    --endpoint-id $RUNPOD_ENDPOINT_ID \
    --api-key $RUNPOD_API_KEY \
    --latency-test \
    --batch-size $BATCH_SIZE \
    --iterations 20 \
    --output results/experiments/bitsandbytes/batch_${BATCH_SIZE}.json
  
  sleep 30
done
```

#### Experiment 3 & 4: AWQ and GPTQ (Optional)

**Important:** AWQ and GPTQ require pre-quantized models. If you don't have pre-quantized versions:

**Option 1:** Skip AWQ/GPTQ and focus on FP16 and BitsAndBytes  
**Option 2:** Pre-quantize the model (requires additional setup - see Troubleshooting section)

If you have pre-quantized models:
1. Update `MODEL_NAME` to point to quantized model
2. Set `QUANTIZATION=awq` or `gptq`
3. Run same test procedure as above

### Expected Time and Cost

**Per Quantization Method:**
- Setup: 5 minutes
- Testing 5 batch sizes × 20 iterations each: 30-45 minutes
- Total per method: ~45-60 minutes

**Full Experiment Suite:**
- 4 methods × 1 hour = 4 hours
- Plus analysis time: ~30 minutes
- **Total: ~5 hours**

**Cost:**
- A100 on-demand: $1.89/hour
- 5 hours × $1.89 = **~$9.45**
- With spot instances: **~$6.50**

---

## Step 10: Load Testing with Locust

Load testing measures performance under sustained concurrent traffic.

### Configure Locust

Edit `locustfile.py` or use environment variables:

```bash
export RUNPOD_API_KEY="your-api-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export BATCH_SIZE=8
```

### Run Headless Load Test

```bash
locust -f locustfile.py \
  --headless \
  -u 10 \
  -r 2 \
  --run-time 5m \
  --html results/load_tests/report.html
```

Parameters:
- `-u 10`: 10 concurrent users
- `-r 2`: Spawn 2 users per second
- `--run-time 5m`: Run for 5 minutes

### Run with Web UI

```bash
locust -f locustfile.py
```

Then open http://localhost:8089 in browser.

### Test Different Batch Sizes

```bash
for BATCH_SIZE in 1 4 8 16 32; do
  export BATCH_SIZE=$BATCH_SIZE
  
  locust -f locustfile.py \
    --headless \
    -u 50 \
    -r 5 \
    --run-time 5m \
    --html results/load_tests/batch_${BATCH_SIZE}/report.html
  
  echo "Completed batch size $BATCH_SIZE load test"
done
```

**Expected Output:**
```
======================================================================
Starting Locust Load Test
======================================================================
Configuration:
  Batch Size: 8
  Users: 10

... test runs ...

======================================================================
Load Test Complete
======================================================================
Overall Statistics:
  Total Requests: 2,543
  Total Failures: 12 (0.47%)
  Median Response Time: 125ms
  95th Percentile: 198ms
  Requests/sec: 8.48
======================================================================
```

---

## Step 11: Analyze Results

After running experiments, analyze and visualize the results.

### Generate Summary CSV

```bash
python scripts/summarize_results.py \
  --results-dir ./results \
  --output ./results/summary.csv
```

Creates `results/summary.csv` with all experiment data.

### Create Visualizations

```bash
python scripts/analyze_results.py \
  --results-dir ./results \
  --output-dir ./results/analysis
```

This generates:
- `comparison_table.csv` - Summary table
- `latency_comparison.png` - Latency vs batch size plots
- `efficiency_analysis.png` - Efficiency plots
- `recommendations.json` - Best configurations

### Generate PDF Report

```bash
python scripts/generate_report.py \
  --results-dir ./results \
  --output ./results/experiment_report.pdf
```

### Interactive Analysis with Jupyter

```bash
jupyter notebook experiments/analysis/comparison.ipynb
```

The notebook includes:
- Interactive plots
- Statistical analysis
- Cost calculations
- Configuration recommendations

---

# APPENDICES

## Project Structure

```
Qwen-Qwen2.5-0.5B-Deployment-vLLM/
├── README.md                          # This file
├── CONTRIBUTING.md                    # Contribution guidelines
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container definition
├── Makefile                          # Common commands
├── quickstart.sh                     # Automated setup
├── locustfile.py                     # Locust load testing
├── env.example                       # Environment variables template
│
├── .github/workflows/
│   └── deploy.yml                    # CI/CD pipeline
│
├── .runpod/
│   └── hub.json                      # RunPod configuration
│
├── app/
│   ├── handler.py                    # RunPod handler
│   └── requirements.txt              # Handler dependencies
│
├── scripts/
│   ├── download_model.py             # Download and verify model
│   ├── benchmark_local.py            # Local benchmarking
│   ├── test_local_handler.py         # Test handler before deployment
│   ├── test_endpoint.py              # Test deployed endpoint
│   ├── run_experiments.py            # Automated experiment runner
│   ├── load_test.py                  # Parameterized load testing
│   ├── analyze_results.py            # Results analysis
│   ├── generate_report.py            # PDF report generation
│   ├── summarize_results.py          # CSV summary creation
│   ├── setup_endpoint.py             # Display endpoint configuration
│   ├── example_client.py             # Example API client
│   ├── check_results_completeness.py # Validate results
│   ├── show_dashboard.py             # Terminal dashboard
│   └── init_directories.py           # Initialize directory structure
│
├── configs/
│   ├── fp16_baseline.json            # FP16 configuration
│   ├── bitsandbytes_int8.json        # INT8 configuration
│   ├── awq_4bit.json                 # AWQ configuration
│   └── gptq_4bit.json                # GPTQ configuration
│
├── experiments/
│   ├── EXPERIMENT_LOG.md             # Experiment tracking template
│   ├── analysis/
│   │   └── comparison.ipynb          # Jupyter analysis notebook
│   └── results/                      # Experiment results directory
│
└── results/
    ├── local_benchmarks/             # Local benchmark results
    ├── experiments/                  # Endpoint test results
    ├── load_tests/                   # Load testing results
    └── analysis/                     # Analysis outputs
```

---

## Configuration Reference

### Environment Variables

All configurations are controlled via environment variables in RunPod.

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `codefactory4791/intent-classification-qwen` | HuggingFace model ID |
| `MAX_NUM_SEQS` | `16` | Maximum sequences per batch |
| `MAX_MODEL_LEN` | `512` | Maximum input length |
| `DTYPE` | `auto` | Data type (auto, float16, bfloat16) |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory utilization (0.0-1.0) |
| `TRUST_REMOTE_CODE` | `true` | Allow custom model code |
| `ENFORCE_EAGER` | `true` | Use eager execution mode |

#### Quantization Settings

| Variable | Options | Description |
|----------|---------|-------------|
| `QUANTIZATION` | `none`, `bitsandbytes`, `awq`, `gptq` | Quantization method |

**Quantization Methods:**

| Method | Memory Usage | Speed vs FP16 | Accuracy Loss | Pre-quantization Required |
|--------|--------------|---------------|---------------|---------------------------|
| `none` (FP16) | ~3.5GB | Baseline | 0% | No |
| `bitsandbytes` | ~2.0GB | 5-10% faster | <1% | No |
| `awq` | ~1.5GB | 10-20% faster | 1-2% | Yes (or slow first load) |
| `gptq` | ~1.5GB | 5-15% faster | 1-2% | Yes |

### Configuration Presets

Pre-configured settings are available in `configs/` directory:

```bash
# Display configuration for specific method
python scripts/setup_endpoint.py --config fp16
python scripts/setup_endpoint.py --config bitsandbytes
python scripts/setup_endpoint.py --config awq
python scripts/setup_endpoint.py --config gptq
```

This shows the exact environment variables to copy-paste into RunPod console.

---

## Troubleshooting

### Issue: vLLM Installation Fails

**Error:** `Failed building wheel for vllm`

**Solution:**
```bash
# Install with specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Issue: CUDA Out of Memory

**Error:** `RuntimeError: CUDA error: out of memory`

**Solutions (apply in order):**

1. Lower memory utilization:
   ```
   GPU_MEMORY_UTILIZATION=0.85
   ```

2. Reduce max sequences:
   ```
   MAX_NUM_SEQS=8
   ```

3. Enable quantization:
   ```
   QUANTIZATION=bitsandbytes
   ```

4. Use smaller batch sizes in your tests

### Issue: Endpoint Returns 401 Unauthorized

**Cause:** Invalid API key or missing Authorization header

**Solutions:**
- Verify API key is correct in `.env`
- Check API key hasn't been revoked in RunPod console
- Ensure using Bearer token: `Authorization: Bearer YOUR_KEY`

### Issue: First Request Takes 30-60 Seconds

**Cause:** Cold start - endpoint scales to 0 when idle

**Solutions:**
- This is normal behavior
- Set Min Workers: 1 to keep endpoint warm (costs more)
- Increase Idle Timeout to 300 seconds
- Send periodic warmup requests

### Issue: Inconsistent Latency Results

**Cause:** GPU warming up, network variance, system load

**Solutions:**
- Run more iterations (20-50 instead of 10)
- Send warmup request before benchmarking
- Wait 30 seconds between tests
- Check GPU temperature in RunPod console

### Issue: AWQ/GPTQ Quantization Fails

**Error:** `Quantization method 'awq' is not supported for this model`

**Cause:** Model needs to be pre-quantized

**Solutions:**

**Option 1:** Skip AWQ/GPTQ and focus on FP16 and BitsAndBytes

**Option 2:** Pre-quantize the model locally:

```bash
# Install quantization library
pip install auto-gptq  # or autoawq

# Quantize model (example for GPTQ)
python - << 'EOF'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_id = "codefactory4791/intent-classification-qwen"
output_dir = "./models/gptq-quantized"

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantize (simplified - actual process more complex)
# See: https://huggingface.co/docs/transformers/quantization

# Save quantized model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
EOF
```

Then update `MODEL_NAME` to point to the quantized model.

### Issue: Docker Build Takes Too Long

**Cause:** Model is downloaded during build

**Solutions:**

**Option 1:** Accept the build time (10-15 minutes)

**Option 2:** Skip pre-download in Dockerfile:

Comment out this section in `Dockerfile`:
```dockerfile
# RUN python3 -c "\
# from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
# ...
```

Trade-off: Faster builds but slower cold starts in production.

---

## API Usage Examples

### Python Client

```python
import requests
import os

# Configuration
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")

# Single prompt
def classify_single(prompt: str):
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "prompts": [prompt]
        }
    }
    
    response = requests.post(url, json=payload, timeout=30)
    result = response.json()
    
    # Extract result
    if "output" in result and "results" in result["output"]:
        classification = result["output"]["results"][0]
        return {
            "class": classification["predicted_class"],
            "confidence": classification["confidence"]
        }
    
    return None

# Usage
result = classify_single("Book me a flight to San Francisco")
print(f"Class: {result['class']}, Confidence: {result['confidence']:.4f}")
```

### Batch Classification

```python
def classify_batch(prompts: list):
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "prompts": prompts
        }
    }
    
    response = requests.post(url, json=payload, timeout=60)
    result = response.json()
    
    if "output" in result and "results" in result["output"]:
        return result["output"]["results"]
    
    return []

# Usage
prompts = [
    "Book a flight",
    "Order pizza", 
    "Play music"
]
results = classify_batch(prompts)

for r in results:
    print(f"{r['prompt']}: Class {r['predicted_class']} ({r['confidence']:.4f})")
```

### Complete Client Class

See `scripts/example_client.py` for a full-featured client implementation with error handling and retry logic.

---

## Makefile Commands Reference

Quick reference for common tasks:

```bash
# Setup
make setup              # Create venv and install dependencies
make download-model     # Download model from HuggingFace

# Local testing (requires GPU)
make benchmark          # Run local benchmarks
make test-handler       # Test handler locally

# Docker
make build              # Build Docker image
make push               # Push to registry
make build-push         # Build and push

# Deployment testing
make test-endpoint      # Test deployed endpoint
make load-test          # Run Locust load test

# Analysis
make analyze            # Analyze results
make report             # Generate PDF report
make notebook           # Launch Jupyter notebook

# Utilities
make clean              # Clean generated files
make help               # Show all commands
```

---

## Expected Results Summary

After completing all experiments, you should have:

### Performance Data

**FP16 Baseline (batch_size=16):**
- P95 Latency: ~100-120ms
- Throughput: ~80-95 samples/s
- GPU Memory: ~3.5GB

**BitsAndBytes INT8 (batch_size=16):**
- P95 Latency: ~90-110ms (5-10% faster)
- Throughput: ~85-105 samples/s
- GPU Memory: ~2.0GB (43% reduction)
- Accuracy: >91.5% (<0.5% degradation)

**AWQ 4-bit (batch_size=16):**
- P95 Latency: ~80-100ms (10-20% faster)
- Throughput: ~95-120 samples/s
- GPU Memory: ~1.5GB (57% reduction)
- Accuracy: >91.0% (~1% degradation)

### Analysis Outputs

- `results/summary.csv` - All results in CSV format
- `results/analysis/comparison_table.csv` - Summary comparison
- `results/analysis/latency_comparison.png` - Latency visualizations
- `results/analysis/efficiency_analysis.png` - Efficiency plots
- `results/analysis/recommendations.json` - Best configurations
- `results/experiment_report.pdf` - Professional PDF report

### Recommendations

The analysis will identify:

1. **Best for Latency:** Lowest P95 latency configuration
2. **Best for Throughput:** Highest samples/second configuration
3. **Best Balanced:** Optimal latency/throughput trade-off
4. **Most Cost-Effective:** Lowest cost per 1K requests

---

## Quick Start Paths

### Path 1: Minimal Testing (1-2 hours, $2-3)

```bash
# Setup
./quickstart.sh
make build-push

# Deploy via RunPod console (see Step 7)
# Test with BitsAndBytes only

python scripts/test_endpoint.py \
  --endpoint-id $RUNPOD_ENDPOINT_ID \
  --api-key $RUNPOD_API_KEY \
  --latency-test \
  --batch-size 16 \
  --iterations 20
```

### Path 2: Focused Experiment (4-5 hours, $8-10)

```bash
# Setup and build
./quickstart.sh
make build-push

# Test FP16 and BitsAndBytes with key batch sizes (1, 8, 16)
# Follow Step 9, Method B

# Analyze
make analyze
```

### Path 3: Complete Suite (10-12 hours, $18-25)

```bash
# Follow all steps in sequence
# Test all 4 quantization methods
# Test all 5 batch sizes
# Run load tests
# Complete analysis with all tools
```

---

## Cost Estimation

### RunPod Pricing (A100 40GB)

- On-demand: $1.89/hour
- Spot: ~$1.29/hour (30% cheaper, can be interrupted)

### Experiment Costs

| Experiment Type | Time | Cost (On-demand) | Cost (Spot) |
|----------------|------|------------------|-------------|
| Single config test | 30 min | $0.95 | $0.65 |
| One quantization method (5 batch sizes) | 1 hour | $1.89 | $1.29 |
| Two methods (FP16 + INT8) | 2 hours | $3.78 | $2.58 |
| Complete suite (4 methods) | 5-6 hours | $10.43 | $7.12 |

### Production Monthly Cost (Example)

Assumptions:
- 100,000 requests per day
- Batch size 16
- BitsAndBytes quantization
- Throughput: ~90 samples/s

Calculation:
```
Samples per month: 100,000 × 30 = 3,000,000
Throughput: 90 samples/s = 324,000 samples/hour
Hours needed: 3,000,000 / 324,000 = 9.26 hours
Cost: 9.26 × $1.89 = $17.50/month (on-demand)
```

---

## Frequently Asked Questions

### Q: Do I need a GPU locally?

**A:** No. Local GPU testing (Steps 4-5) is optional. You can skip directly to Docker build and RunPod deployment.

### Q: Which quantization method should I use?

**A:**
- **Best accuracy:** FP16 (none)
- **Best balance:** BitsAndBytes (INT8)
- **Best speed/memory:** AWQ (if you can pre-quantize)

**Recommendation:** Start with BitsAndBytes for good balance of performance and simplicity.

### Q: What batch size should I use?

**A:** Depends on use case:
- **Real-time chat:** Batch size 1-4 (low latency)
- **API service:** Batch size 8-16 (balanced)
- **Batch processing:** Batch size 16-32 (high throughput)

Run experiments to find optimal for your specific latency requirements.

### Q: How long does each experiment take?

**A:**
- Single batch size test: ~5-10 minutes
- One quantization method (5 batch sizes): ~45-60 minutes
- Complete suite (4 methods, 5 batch sizes): ~4-6 hours

### Q: The model is public - do I need a HuggingFace token?

**A:** No. The model `codefactory4791/intent-classification-qwen` is public. No authentication needed.

### Q: Can I run this without Docker?

**A:** Docker is required for RunPod deployment. But you can test the handler locally without Docker using `scripts/test_local_handler.py`.

### Q: How do I monitor costs during experiments?

**A:** 
1. RunPod Console → Billing → Usage
2. Set up billing alerts in RunPod
3. Each test iteration logs to RunPod metrics
4. Estimate: ~$2/hour on A100

---

## Support and Resources

### Project Resources
- **Repository:** https://github.com/llmdeveloper47/Qwen-Qwen2.5-0.5B-Deployment-vLLM
- **Model Card:** https://huggingface.co/codefactory4791/intent-classification-qwen
- **Dataset:** https://huggingface.co/datasets/codefactory4791/amazon_test

### External Documentation
- **vLLM Docs:** https://docs.vllm.ai/
- **RunPod Docs:** https://docs.runpod.io/
- **RunPod Discord:** https://discord.gg/runpod
- **Qwen Models:** https://huggingface.co/Qwen

### Getting Help

1. Check the Troubleshooting section above
2. Review error messages in RunPod logs
3. Open an issue: https://github.com/llmdeveloper47/Qwen-Qwen2.5-0.5B-Deployment-vLLM/issues
4. Ask on RunPod Discord

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

---

## License

MIT License - see LICENSE file for details.

---

## Citation

```bibtex
@misc{qwen-intent-classification-runpod,
  title={Deploying Fine-tuned Qwen 2.5 for Intent Classification on RunPod},
  author={llmdeveloper47},
  year={2025},
  publisher={GitHub},
  url={https://github.com/llmdeveloper47/Qwen-Qwen2.5-0.5B-Deployment-vLLM}
}
```

---

**Version:** 1.0.0  
**Last Updated:** 2025  
**Repository:** https://github.com/llmdeveloper47/Qwen-Qwen2.5-0.5B-Deployment-vLLM
