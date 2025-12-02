"""
RunPod Handler for Intent Classification using Optimized Transformers
This handler uses optimized transformers for fast classification inference.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Union

import torch
import runpod
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Read configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "codefactory4791/intent-classification-qwen")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "512"))
QUANTIZATION = os.getenv("QUANTIZATION", "none")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
USE_BETTER_TRANSFORMER = os.getenv("USE_BETTER_TRANSFORMER", "true").lower() == "true"
USE_COMPILE = os.getenv("USE_COMPILE", "true").lower() == "true"

logger.info(f"Starting optimized classification handler with configuration:")
logger.info(f"  MODEL_NAME: {MODEL_NAME}")
logger.info(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
logger.info(f"  QUANTIZATION: {QUANTIZATION}")
logger.info(f"  TRUST_REMOTE_CODE: {TRUST_REMOTE_CODE}")
logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"  USE_BETTER_TRANSFORMER: {USE_BETTER_TRANSFORMER}")
logger.info(f"  USE_COMPILE: {USE_COMPILE}")

# Initialize model once at startup
logger.info("Initializing model with optimizations...")
start_time = time.time()

try:
    # Load tokenizer (matching reference notebook)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with quantization
    if QUANTIZATION == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif QUANTIZATION == "awq":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            device_map="auto"
        )
    elif QUANTIZATION == "gptq":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            device_map="auto"
        )
    else:
        # No quantization - simple loading like reference notebook
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model = model.to(device)
    
    # Ensure model knows the pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set to evaluation mode
    model.eval()
    
    # Apply BetterTransformer optimization if enabled and supported
    if USE_BETTER_TRANSFORMER and QUANTIZATION == "none":
        try:
            model = model.to_bettertransformer()
            logger.info("  Applied BetterTransformer optimization")
        except Exception as e:
            logger.warning(f"  Could not apply BetterTransformer: {e}")
    
    # Apply torch.compile if enabled (PyTorch 2.0+)
    if USE_COMPILE and QUANTIZATION == "none" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("  Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"  Could not apply torch.compile: {e}")
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f}s")
    logger.info(f"  Device: {device}")
    logger.info(f"  Num labels: {model.config.num_labels}")
    
    # Store global references
    MODEL = model
    TOKENIZER = tokenizer
    DEVICE = device
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    import traceback
    traceback.print_exc()
    raise


def classify_batch(prompts: List[str]) -> List[Dict[str, Any]]:
    """
    Run classification on a batch of prompts using optimized inference.
    
    Args:
        prompts: List of text prompts to classify
        
    Returns:
        List of dictionaries containing classification results
    """
    try:
        start_time = time.time()
        
        # Tokenize batch
        inputs = TOKENIZER(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Run inference with no gradient computation
        with torch.no_grad():
            outputs = MODEL(**inputs)
            logits = outputs.logits
            
            # Get probabilities using softmax
            probs = torch.softmax(logits, dim=-1)
            
            # Get predicted classes and confidence scores
            confidence_scores, predicted_classes = torch.max(probs, dim=-1)
        
        inference_time = time.time() - start_time
        
        # Convert to CPU and numpy for processing
        probs_np = probs.cpu().numpy()
        predicted_classes_np = predicted_classes.cpu().numpy()
        confidence_scores_np = confidence_scores.cpu().numpy()
        
        # Process results
        results = []
        for idx in range(len(prompts)):
            result = {
                "prompt": prompts[idx],
                "predicted_class": int(predicted_classes_np[idx]),
                "confidence": float(confidence_scores_np[idx]),
                "probabilities": probs_np[idx].tolist(),
            }
            results.append(result)
        
        logger.info(
            f"Classified {len(prompts)} prompts in {inference_time:.3f}s "
            f"({len(prompts)/inference_time:.2f} samples/s)"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.
    
    Expected input format:
    {
        "input": {
            "prompts": ["text1", "text2", ...] or "single text"
        }
    }
    
    Returns:
    {
        "results": [
            {
                "prompt": "text",
                "predicted_class": 0,
                "confidence": 0.95,
                "probabilities": [...]
            },
            ...
        ],
        "metadata": {
            "num_prompts": 2,
            "inference_time": 0.123,
            "model_name": "...",
            "quantization": "..."
        }
    }
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not event or 'input' not in event:
            return {
                "error": "Missing 'input' field in request",
                "status": "error"
            }
        
        data = event['input']
        prompts = data.get('prompts')
        
        if not prompts:
            return {
                "error": "No 'prompts' provided in input",
                "status": "error"
            }
        
        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]
        elif not isinstance(prompts, list):
            return {
                "error": "Invalid prompt format. Expected string or list of strings.",
                "status": "error"
            }
        
        # Validate prompt types
        if not all(isinstance(p, str) for p in prompts):
            return {
                "error": "All prompts must be strings",
                "status": "error"
            }
        
        # Run classification
        results = classify_batch(prompts)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Return successful response
        return {
            "results": results,
            "metadata": {
                "num_prompts": len(prompts),
                "total_time": round(total_time, 4),
                "avg_time_per_prompt": round(total_time / len(prompts), 4),
                "model_name": MODEL_NAME,
                "quantization": QUANTIZATION,
                "max_num_seqs": MAX_NUM_SEQS,
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "status": "error"
        }


# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})

