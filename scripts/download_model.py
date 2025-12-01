#!/usr/bin/env python3
"""
Download and verify the intent classification model from Hugging Face.
This script downloads the model, checks its configuration, and saves metadata.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def download_model(model_id: str, output_dir: str = "./models") -> dict:
    """
    Download model and tokenizer from Hugging Face.
    
    Args:
        model_id: Hugging Face model ID
        output_dir: Directory to save model info
        
    Returns:
        Dictionary with model information
    """
    print("=" * 70)
    print(f"Downloading Model: {model_id}")
    print("=" * 70)
    
    try:
        # Create output directory first
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        print("\n[1/4] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Fix pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  ✓ Set pad_token to eos_token")
        
        print(f"  ✓ Tokenizer downloaded successfully")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Pad token: {tokenizer.pad_token}")
        
        # Save tokenizer to disk
        print("\n[2/4] Saving tokenizer to disk...")
        tokenizer.save_pretrained(output_dir)
        print(f"  ✓ Tokenizer saved to: {output_dir}")
        
        # Download model
        print("\n[3/4] Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Set pad token id in model config
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        print(f"  ✓ Model downloaded successfully")
        print(f"  - Model type: {model.config.model_type}")
        print(f"  - Number of parameters: {model.num_parameters():,}")
        print(f"  - Number of labels: {model.config.num_labels}")
        
        # Save model to disk
        print(f"\n  Saving model to disk...")
        model.save_pretrained(output_dir)
        print(f"  ✓ Model saved to: {output_dir}")
        
        # Get model size
        model_size_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Extract label mappings
        id2label = model.config.id2label
        label2id = model.config.label2id
        
        print(f"\n[4/4] Verifying configuration...")
        print(f"  ✓ Model size: {model_size_mb:.2f} MB")
        print(f"  ✓ Number of classes: {len(id2label)}")
        
        # Display first few labels
        print("\n  Sample labels:")
        for idx in sorted(id2label.keys())[:5]:
            print(f"    {idx}: {id2label[idx]}")
        print(f"    ... ({len(id2label) - 5} more)")
        
        # Save model info
        model_info = {
            "model_id": model_id,
            "model_type": model.config.model_type,
            "num_parameters": model.num_parameters(),
            "model_size_mb": round(model_size_mb, 2),
            "num_labels": model.config.num_labels,
            "id2label": id2label,
            "label2id": label2id,
            "vocab_size": tokenizer.vocab_size,
            "max_position_embeddings": getattr(
                model.config, "max_position_embeddings", None
            ),
            "hidden_size": getattr(model.config, "hidden_size", None),
        }
        
        output_path = os.path.join(output_dir, "model_info.json")
        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n  ✓ Model info saved to: {output_path}")
        
        # Test model (optional)
        print(f"\n[Testing] Running quick inference test...")
        try:
            test_text = """Beautiful design. Twelve rolls of the most beautiful washi tape you could own is sold right here on Amazon and by Allydrew. The colors are gorgeous, the tape is easy to lay down and lift up if you need to readjust it, and it tears easy which makes it a joy to work with. I have a humongous washi collection and Allydrew is some of the best I have found. The patterns are beautiful and unlike any I have found at the craft stores. They are 10 yard rolls, which is a great amount per roll. If you break down the price to a per roll price it breaks down to around $1.60 per roll. That is a really great price.

This retails for $18.99 per twelve roll pack but I received it at the discounted amount of $9.30 in exchange for my honest review.

I review all the Allydrew washi I get from AMZ on my YouTube channel (sprngbrd *) if you would like to see each individual roll that way."""
            expected_label = "arts-crafts-and-sewing_crafting"
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()
            
            # Get label name, handling potential missing keys
            label_name = id2label.get(str(predicted_class), f"UNKNOWN_CLASS_{predicted_class}")
            
            print(f"  ✓ Test inference successful")
            print(f"    Input: '{test_text[:80]}...'")
            print(f"    Expected label: {expected_label}")
            print(f"    Predicted class: {predicted_class} ({label_name})")
            print(f"    Confidence: {confidence:.4f}")
            
            # Check if prediction matches expected
            if label_name == expected_label:
                print(f"    ✓ Prediction matches expected label!")
            else:
                print(f"    ⚠ Prediction differs from expected (this is okay for initial test)")
        except Exception as test_error:
            print(f"  ⚠ Warning: Test inference failed ({str(test_error)})")
            print(f"    This is non-critical - model files are downloaded correctly")
        
        print("\n" + "=" * 70)
        print("✓ Model download and verification complete!")
        print("=" * 70)
        
        return model_info
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {str(e)}")
        print(f"\nDebug info:")
        print(f"  Model ID: {model_id}")
        print(f"  Output directory: {output_dir}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify intent classification model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="codefactory4791/intent-classification-qwen",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save model info"
    )
    
    args = parser.parse_args()
    
    # Download model
    model_info = download_model(args.model_id, args.output_dir)
    
    print(f"\nNext steps:")
    print(f"  1. Run benchmarks: python scripts/benchmark_local.py")
    print(f"  2. Test handler: python scripts/test_local_handler.py")
    print(f"  3. Build Docker: docker build -t intent-classification-vllm .")


if __name__ == "__main__":
    main()

