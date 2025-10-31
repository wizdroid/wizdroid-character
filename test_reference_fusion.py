#!/usr/bin/env python3
"""
Test script for ClothesExtractionNode.
Creates dummy images and tests the vision API call.
"""

import sys
import os

# Add the nodes directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from nodes.clothes_extraction_node import ClothesExtractionNode

def create_dummy_image(color=(128, 128, 128), size=(512, 512)):
    """Create a dummy torch image tensor with specified color."""
    # ComfyUI uses BHWC format (Batch, Height, Width, Channels)
    # Values should be in range [0, 1]
    img_array = np.zeros((1, size[1], size[0], 3), dtype=np.float32)
    img_array[:, :, :, 0] = color[0] / 255.0  # R
    img_array[:, :, :, 1] = color[1] / 255.0  # G
    img_array[:, :, :, 2] = color[2] / 255.0  # B
    return torch.from_numpy(img_array)

def test_image_conversion():
    """Test that images convert to base64 correctly."""
    print("\n=== Testing Image Conversion ===")
    from nodes.clothes_extraction_node import _image_to_base64
    
    dummy_img = create_dummy_image((255, 0, 0))  # Red image
    b64_result = _image_to_base64(dummy_img)
    
    if b64_result and len(b64_result) > 100:
        print(f"✓ Image conversion successful: {len(b64_result)} chars")
        print(f"  First 50 chars: {b64_result[:50]}...")
        return True
    else:
        print(f"✗ Image conversion failed: {b64_result}")
        return False

def test_ollama_connection():
    """Test connection to Ollama API."""
    print("\n=== Testing Ollama Connection ===")
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama is running")
            print(f"  Available models: {len(models)}")
            for model in models[:5]:
                print(f"    - {model['name']}")
            return True
        else:
            print(f"✗ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        return False

def test_vision_models():
    """Test vision model detection."""
    print("\n=== Testing Vision Model Detection ===")
    from nodes.clothes_extraction_node import ClothesExtractionNode
    
    models = ClothesExtractionNode._collect_ollama_models("http://localhost:11434")
    if models:
        print(f"✓ Found {len(models)} vision models:")
        for model in models:
            print(f"    - {model}")
        return True
    else:
        print("✗ No vision models found")
        return False

def test_full_prompt_generation():
    """Test the full prompt generation pipeline."""
    print("\n=== Testing Full Prompt Generation ===")
    
    # Create dummy image
    character_img = create_dummy_image((255, 0, 0))    # Red
    
    # Get available vision models
    models = ClothesExtractionNode._collect_ollama_models("http://localhost:11434")
    if not models:
        print("✗ No vision models available for testing")
        return False
    
    test_model = models[0]
    print(f"Using model: {test_model}")
    
    # Initialize node
    node = ClothesExtractionNode()
    
    params = {
        "ollama_url": "http://localhost:11434/api/generate",
        "ollama_model": test_model,
        "style": "photorealistic",
        "gender": "woman",
        "age_group": "adult",
        "character_image": character_img,
        "custom_prompt_1": "cinematic lighting",
        "custom_prompt_2": "4k ultra detailed",
        "custom_prompt_3": "highly realistic",
    }
    
    print(f"\nGenerating clothing description prompt...")
    
    try:
        result = node.extract_clothing(**params)
        prompt = result[0]
        
        if prompt and not prompt.startswith("[ERROR"):
            print(f"\n✓ Prompt generated successfully!")
            print(f"\n--- Generated Clothing Prompt ---")
            print(prompt)
            print(f"--- End ({len(prompt)} chars) ---")
            return True
        else:
            print(f"✗ Prompt generation failed: {prompt}")
            return False
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("ClothesExtractionNode Test Suite")
    print("="*60)
    
    tests = [
        ("Image Conversion", test_image_conversion),
        ("Ollama Connection", test_ollama_connection),
        ("Vision Model Detection", test_vision_models),
        ("Full Prompt Generation", test_full_prompt_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    print("="*60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
