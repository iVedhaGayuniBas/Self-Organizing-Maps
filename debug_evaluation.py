#!/usr/bin/env python3
"""
Debug script to test LLM evaluation with actual data
"""

import pandas as pd
import requests
import json
import time

class OllamaEvaluator:
    def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def query_llm(self, prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["response"].strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to query LLM after {max_retries} attempts: {e}")
                    return "NO"
                time.sleep(retry_delay * (2 ** attempt))
    
    def evaluate_context_contains_answer(self, question: str, answer: str, context: str) -> bool:
        prompt = f"""You are evaluating whether a given context contains the answer to a question.

Question: {question}
Expected Answer: {answer}
Context: {context}

Does the context contain the answer to the question? Respond with only "YES" or "NO".

Response:"""
        
        response = self.query_llm(prompt)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Context: {context[:200]}...")
        print(f"LLM Response: {response}")
        print(f"Contains Answer: {response.upper().startswith('YES')}")
        print("-" * 80)
        return response.upper().startswith("YES")

def main():
    # Load the CSV data
    df = pd.read_csv('wikipedia_context_comparison.csv')
    
    # Get the first row
    row = df.iloc[0]
    question = row['question']
    answer = row['answer']
    
    # Parse the contexts (they're stored as tuples)
    import ast
    import numpy as np
    
    def safe_eval(x):
        try:
            # Use eval with numpy available
            return eval(x, {"__builtins__": {}}, {"array": np.array, "np": np})
        except:
            return ([], [])
    
    som_tuple = safe_eval(row['som_context'])
    cosine_tuple = safe_eval(row['cosine_context'])
    
    som_contexts = som_tuple[0]  # List of context strings
    cosine_contexts = cosine_tuple[0]  # List of context strings
    
    print("Testing LLM evaluation with first example:")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Number of SOM contexts: {len(som_contexts)}")
    print(f"Number of Cosine contexts: {len(cosine_contexts)}")
    print()
    
    evaluator = OllamaEvaluator()
    
    # Test SOM contexts
    print("Testing SOM contexts:")
    som_contains_answer = False
    for i, ctx in enumerate(som_contexts):
        print(f"Context {i+1}:")
        if evaluator.evaluate_context_contains_answer(question, answer, ctx):
            som_contains_answer = True
            break
    
    print(f"\nSOM contains answer: {som_contains_answer}")
    
    # Test Cosine contexts
    print("\nTesting Cosine contexts:")
    cosine_contains_answer = False
    for i, ctx in enumerate(cosine_contexts):
        print(f"Context {i+1}:")
        if evaluator.evaluate_context_contains_answer(question, answer, ctx):
            cosine_contains_answer = True
            break
    
    print(f"\nCosine contains answer: {cosine_contains_answer}")

if __name__ == "__main__":
    main() 