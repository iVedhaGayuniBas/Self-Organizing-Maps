#!/usr/bin/env python3
"""
Test script to verify Ollama evaluation system works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from context_evaluation_with_ollama import OllamaEvaluator
import time

def test_ollama_connection():
    """Test basic Ollama connection and model response"""
    print("🔍 Testing Ollama connection...")
    
    try:
        evaluator = OllamaEvaluator()
        
        # Simple test prompt
        test_prompt = "Answer with only 'Yes' or 'No': Is the sky blue?"
        
        print(f"Sending test prompt: {test_prompt}")
        response = evaluator.query_llm(test_prompt)
        print(f"Response: '{response}'")
        
        if response.lower().strip() in ['yes', 'no']:
            print("✅ Ollama connection successful!")
            return True
        else:
            print(f"⚠️  Unexpected response format: '{response}'")
            return False
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_context_evaluation():
    """Test context evaluation functionality"""
    print("\n🔍 Testing context evaluation...")
    
    try:
        evaluator = OllamaEvaluator()
        
        # Test case 1: Context contains answer
        question = "What is the capital of France?"
        answer = "Paris"
        context = "Paris is the capital and largest city of France. It is known for the Eiffel Tower."
        
        print(f"Test 1: Question='{question}', Answer='{answer}'")
        print(f"Context: '{context}'")
        
        result1 = evaluator.evaluate_context_contains_answer(question, answer, context)
        print(f"Result: {result1}")
        
        # Test case 2: Context doesn't contain answer
        question2 = "What is the capital of Japan?"
        answer2 = "Tokyo"
        context2 = "Paris is the capital and largest city of France. It is known for the Eiffel Tower."
        
        print(f"\nTest 2: Question='{question2}', Answer='{answer2}'")
        print(f"Context: '{context2}'")
        
        result2 = evaluator.evaluate_context_contains_answer(question2, answer2, context2)
        print(f"Result: {result2}")
        
        # Verify results make sense
        if result1 == True and result2 == False:
            print("✅ Context evaluation working correctly!")
            return True
        else:
            print(f"⚠️  Unexpected evaluation results: {result1}, {result2}")
            return False
            
    except Exception as e:
        print(f"❌ Context evaluation failed: {e}")
        return False

def test_performance():
    """Test evaluation performance with multiple requests"""
    print("\n🔍 Testing performance...")
    
    try:
        evaluator = OllamaEvaluator()
        
        # Test data
        test_cases = [
            ("What is 2+2?", "4", "The answer is 4."),
            ("What color is the sky?", "Blue", "The sky appears blue during the day."),
            ("What is the largest planet?", "Jupiter", "Jupiter is the largest planet in our solar system."),
        ]
        
        start_time = time.time()
        
        for i, (question, answer, context) in enumerate(test_cases, 1):
            print(f"Processing test case {i}/{len(test_cases)}...")
            result = evaluator.evaluate_context_contains_answer(question, answer, context)
            print(f"  Result: {result}")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(test_cases)
        
        print(f"✅ Performance test completed!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average time per evaluation: {avg_time:.2f} seconds")
        print(f"   Evaluations per second: {len(test_cases)/total_time:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Ollama Evaluation System")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Context Evaluation", test_context_evaluation),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The evaluation system is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your questions_answers.xlsx file")
        print("2. Replace dummy context data with your actual SOM and cosine results")
        print("3. Run: python context_evaluation_with_ollama.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check if llama3:latest is available: ollama list")
        print("3. Verify network connectivity to localhost:11434")

if __name__ == "__main__":
    main() 