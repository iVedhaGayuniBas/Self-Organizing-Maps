#!/usr/bin/env python3
"""
Simple test script to verify Ollama connectivity and basic functionality
"""

import requests
import json


def test_ollama_connection(base_url="http://localhost:11434"):
    """Test basic Ollama connection"""
    print(f"Testing Ollama connection to {base_url}...")
    
    try:
        # Test if Ollama is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        print(f"✅ Ollama is running with {len(models)} models available")
        
        # List available models
        for model in models:
            print(f"   - {model['name']}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("   Start with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False


def test_ollama_query(base_url="http://localhost:11434", model="llama3:latest"):
    """Test a simple query to Ollama"""
    print(f"\nTesting Ollama query with model {model}...")
    
    try:
        prompt = "Answer with only YES or NO: Is Paris the capital of France?"
        
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()["response"].strip()
        print(f"✅ Query successful. Response: '{result}'")
        
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Query timed out. Ollama might be slow or busy.")
        return False
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return False


def main():
    """Run all tests"""
    print("="*50)
    print("OLLAMA CONNECTIVITY TEST")
    print("="*50)
    
    # Test connection
    if not test_ollama_connection():
        return
    
    # Test query
    if not test_ollama_query():
        return
    
    print("\n✅ All tests passed! Ollama is ready for evaluation.")


if __name__ == "__main__":
    main()
