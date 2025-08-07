#!/usr/bin/env python3
"""
Scale Up Evaluation Script
Loads actual SOM and Cosine context data from notebook results and scales up evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pickle
import warnings
import ast
warnings.filterwarnings('ignore')

# Optional Ray imports
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

class OllamaEvaluator:
    """Evaluates contexts using local Ollama LLM"""
    
    def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def query_llm(self, prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
        """Query Ollama with retry logic"""
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
                    return "NO"  # Default to negative response
                time.sleep(retry_delay * (2 ** attempt))
    
    def evaluate_context_contains_answer(self, question: str, answer: str, context: str) -> bool:
        """Evaluate if context contains the answer using Ollama"""
        prompt = f"""You are evaluating whether a given context contains the answer to a question.

Question: {question}
Expected Answer: {answer}
Context: {context}

Does the context contain the answer to the question? Respond with only "YES" or "NO".

Response:"""
        
        response = self.query_llm(prompt)
        return response.upper().startswith("YES")

# Ray-enabled evaluator for distributed processing
if RAY_AVAILABLE:
    @ray.remote
    class RayOllamaEvaluator:
        """Ray remote actor for distributed evaluation"""
        
        def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://host.docker.internal:11434"):
            self.evaluator = OllamaEvaluator(model_name, base_url)
        
        def evaluate_batch(self, batch_data: List[Tuple]) -> List[Dict[str, Any]]:
            """Evaluate a batch of questions"""
            results = []
            for question, answer, som_contexts, cosine_contexts in batch_data:
                som_contains_answer = any(
                    self.evaluator.evaluate_context_contains_answer(question, answer, ctx)
                    for ctx in som_contexts
                )
                
                cosine_contains_answer = any(
                    self.evaluator.evaluate_context_contains_answer(question, answer, ctx)
                    for ctx in cosine_contexts
                )
                
                results.append({
                    'question': question,
                    'answer': answer,
                    'som_contains_answer': som_contains_answer,
                    'cosine_contains_answer': cosine_contains_answer
                })
            return results

def load_qa_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load questions and answers from Excel file (Title | Question | Answer)"""
    try:
        df = pd.read_excel(file_path)
        if 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("Excel file must contain 'Question' and 'Answer' columns")
        
        questions = df['question'].astype(str).str.strip().tolist()
        answers = df['answer'].astype(str).str.strip().tolist()
        
        print(f"Loaded {len(questions)} questions and answers from {file_path}")
        return questions, answers
    except Exception as e:
        print(f"Error loading QA data: {e}")
        return [], []

def load_from_csv_results(csv_file: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load SOM and Cosine contexts from a CSV where each context column contains a stringified list.
    Columns: 'question' | 'answer' | 'som_context' | 'cosine_context'
    """
    try:
        df = pd.read_csv(csv_file)
        if 'som_context' not in df.columns or 'cosine_context' not in df.columns:
            raise ValueError("CSV must contain 'som_context' and 'cosine_context' columns")

        def safe_eval(x):
            try:
                # Use ast.literal_eval for safer evaluation
                result = ast.literal_eval(x)
                # If it's a tuple, extract the first element (the context list)
                if isinstance(result, tuple):
                    return result[0]
                return result
            except:
                return []

        som_contexts = df['som_context'].apply(safe_eval).tolist()
        cosine_contexts = df['cosine_context'].apply(safe_eval).tolist()

        print(f"Loaded {len(som_contexts)} context pairs from {csv_file}")
        return som_contexts, cosine_contexts
    except Exception as e:
        print(f"Error loading contexts from CSV: {e}")
        return [], []


def evaluate_single_item(args: Tuple) -> Dict[str, Any]:
    question, answer, som_contexts, cosine_contexts, evaluator = args

    som_contains_answer = any(
        evaluator.evaluate_context_contains_answer(question, answer, ctx)
        for ctx in som_contexts
    )

    cosine_contains_answer = any(
        evaluator.evaluate_context_contains_answer(question, answer, ctx)
        for ctx in cosine_contexts
    )

    return {
        'question': question,
        'answer': answer,
        'som_contains_answer': som_contains_answer,
        'cosine_contains_answer': cosine_contains_answer
    }


def evaluate_context_retrieval(
    questions: List[str], 
    answers: List[str], 
    som_contexts: List[Tuple[List[str], np.ndarray]], 
    cosine_contexts: List[Tuple[List[str], np.ndarray]], 
    max_workers: int = 4,
    use_ray: bool = False,
    ray_evaluators: int = 8,
    batch_size: int = 10
) -> Tuple[pd.DataFrame, Dict, Dict, Dict, Dict]:
    """
    Evaluate context retrieval performance using Ollama
    """
    print(f"Starting evaluation of {len(questions)} questions...")
    
    if use_ray and RAY_AVAILABLE:
        print(f"Using Ray distributed evaluation with {ray_evaluators} evaluators")
        return _evaluate_with_ray(questions, answers, som_contexts, cosine_contexts, 
                                ray_evaluators, batch_size)
    else:
        print(f"Using standard evaluation with {max_workers} workers")
        return _evaluate_standard(questions, answers, som_contexts, cosine_contexts, max_workers)

def _evaluate_standard(questions, answers, som_contexts, cosine_contexts, max_workers):
    """Standard evaluation using ThreadPoolExecutor"""
    evaluator = OllamaEvaluator()
    
    args_list = [
        (question, answer, som_ctx, cos_ctx, evaluator)
        for question, answer, som_ctx, cos_ctx in zip(questions, answers, som_contexts, cosine_contexts)
    ]
    
    results = []
    
    # Process in batches to avoid overwhelming the system
    batch_size = 50
    for i in tqdm(range(0, len(args_list), batch_size), desc="Processing batches"):
        batch = args_list[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(evaluate_single_item, batch))
            results.extend(batch_results)
        
        # Small delay between batches
        time.sleep(1)
    
    return _process_results(results)

def _evaluate_with_ray(questions, answers, som_contexts, cosine_contexts, ray_evaluators, batch_size):
    """Ray distributed evaluation"""
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(address='auto', ignore_reinit_error=True)
    
    # Create evaluator actors
    evaluators = [RayOllamaEvaluator.remote() for _ in range(ray_evaluators)]
    
    # Prepare data batches
    data = list(zip(questions, answers, som_contexts, cosine_contexts))
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    # Distribute work across evaluators
    futures = []
    for i, batch in enumerate(batches):
        evaluator = evaluators[i % len(evaluators)]
        future = evaluator.evaluate_batch.remote(batch)
        futures.append(future)
    
    # Collect results
    print(f"Processing {len(batches)} batches with {len(evaluators)} evaluators...")
    results = []
    for future in tqdm(ray.get(futures), desc="Collecting results"):
        results.extend(future)
    
    return _process_results(results)

def _process_results(results):
    """Process evaluation results and calculate metrics"""
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate confusion matrices
    actual = (df['answer'].str.strip() != "").astype(int)
    som_cm = confusion_matrix(actual, df['som_contains_answer'])
    cosine_cm = confusion_matrix(actual, df['cosine_contains_answer'])

    # Calculate metrics
    som_metrics = {
        'Precision': precision_score(actual, df['som_contains_answer'], zero_division=0),
        'Recall': recall_score(df['som_contains_answer'], df['som_contains_answer'], zero_division=0),
        'F1 Score': f1_score(df['som_contains_answer'], df['som_contains_answer'], zero_division=0)
    }
    
    cosine_metrics = {
        'Precision': precision_score(actual, df['cosine_contains_answer'], zero_division=0),
        'Recall': recall_score(df['cosine_contains_answer'], df['cosine_contains_answer'], zero_division=0),
        'F1 Score': f1_score(df['cosine_contains_answer'], df['cosine_contains_answer'], zero_division=0)
    }
    
    return df, som_cm, cosine_cm, som_metrics, cosine_metrics

def create_comparison_chart(som_metrics: Dict, cosine_metrics: Dict, save_path: str = "scaled_comparison_chart.png"):
    """Create a bar chart comparing SOM vs Cosine metrics"""
    metrics = ['Precision', 'Recall', 'F1 Score']
    som_values = [som_metrics[metric] for metric in metrics]
    cosine_values = [cosine_metrics[metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, som_values, width, label='SOMpy', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, cosine_values, width, label='Cosine Similarity', color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('SOMpy vs Cosine Similarity: Context Retrieval Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as {save_path}")

def print_evaluation_summary(df: pd.DataFrame, som_metrics: Dict, cosine_metrics: Dict):
    """Print a summary of the evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal Questions Evaluated: {len(df)}")
    
    print(f"\nSOMpy Performance:")
    print(f"  - Precision: {som_metrics['Precision']:.3f}")
    print(f"  - Recall: {som_metrics['Recall']:.3f}")
    print(f"  - F1 Score: {som_metrics['F1 Score']:.3f}")
    print(f"  - Questions with correct context: {df['som_contains_answer'].sum()}")
    
    print(f"\nCosine Similarity Performance:")
    print(f"  - Precision: {cosine_metrics['Precision']:.3f}")
    print(f"  - Recall: {cosine_metrics['Recall']:.3f}")
    print(f"  - F1 Score: {cosine_metrics['F1 Score']:.3f}")
    print(f"  - Questions with correct context: {df['cosine_contains_answer'].sum()}")
    
    # Calculate improvement
    som_f1 = som_metrics['F1 Score']
    cosine_f1 = cosine_metrics['F1 Score']
    improvement = ((som_f1 - cosine_f1) / cosine_f1 * 100) if cosine_f1 > 0 else 0
    
    print(f"\nSOMpy vs Cosine Improvement:")
    print(f"  - F1 Score Improvement: {improvement:.1f}%")
    print(f"  - Absolute F1 Difference: {som_f1 - cosine_f1:.3f}")

def main():
    """Main function to run the scaled evaluation"""
    parser = argparse.ArgumentParser(description='Scale up context evaluation with actual data')
    parser.add_argument('--qa_file', type=str, default='questions_answers.xlsx', 
                       help='Path to Excel file with questions and answers')
    parser.add_argument('--csv_file', type=str, default='wikipedia_context_comparison.csv',
                       help='Path to CSV file with notebook results (alternative to context_file)')
    parser.add_argument('--max_questions', type=int, default=5000,
                       help='Maximum number of questions to evaluate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--model', type=str, default='llama3:latest',
                       help='Ollama model to use')
    parser.add_argument('--use_ray', action='store_true',
                       help='Use Ray for distributed evaluation')
    parser.add_argument('--ray_evaluators', type=int, default=8,
                       help='Number of Ray evaluator actors (only with --use_ray)')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for Ray evaluation (only with --use_ray)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SCALE UP EVALUATION WITH ACTUAL DATA")
    print("="*60)
    
    # Check Ray availability
    if args.use_ray and not RAY_AVAILABLE:
        print("Warning: Ray is not available. Falling back to standard evaluation.")
        args.use_ray = False
    
    if args.use_ray:
        print(f"Ray distributed evaluation enabled with {args.ray_evaluators} evaluators")
    
    # Load Q&A data
    questions, answers = load_qa_data(args.qa_file)
    if not questions:
        print("Failed to load Q&A data. Exiting.")
        return
    
    # Limit to max_questions
    if len(questions) > args.max_questions:
        print(f"Limiting evaluation to {args.max_questions} questions (from {len(questions)} total)")
        questions = questions[:args.max_questions]
        answers = answers[:args.max_questions]
    
    # Load contexts - try pickle first, then CSV
    som_contexts, cosine_contexts = [], []
    
    # Try loading from pickle files
    if os.path.exists("contexts/som_contexts_scores.pkl") and os.path.exists("contexts/cosine_contexts_scores.pkl"):
        try:
            from save_contexts_simple import load_contexts
            som_contexts_scores, cosine_contexts_scores = load_contexts()
            print(f"Loaded {len(som_contexts_scores)} context pairs from pickle files")
            
            # Convert to the format expected by evaluation
            som_contexts = [ctx[0] for ctx in som_contexts_scores]  # Extract just the context lists
            cosine_contexts = [ctx[0] for ctx in cosine_contexts_scores]  # Extract just the context lists
            
        except Exception as e:
            print(f"Error loading from pickle: {e}")
    
    # Fallback to CSV if pickle failed
    if not som_contexts and args.csv_file and os.path.exists(args.csv_file):
        som_contexts, cosine_contexts = load_from_csv_results(args.csv_file)
    
    
    # Ensure we have contexts for all questions
    if len(som_contexts) < len(questions):
        print(f"Warning: Only {len(som_contexts)} SOM contexts available for {len(questions)} questions")
        questions = questions[:len(som_contexts)]
        answers = answers[:len(som_contexts)]
        som_contexts = som_contexts[:len(questions)]
        cosine_contexts = cosine_contexts[:len(questions)]
    
    print(f"\nReady to evaluate {len(questions)} questions with actual context data")
    print(f"SOM contexts available: {len(som_contexts)}")
    print(f"Cosine contexts available: {len(cosine_contexts)}")
    
    # Run evaluation
    df, som_cm, cosine_cm, som_metrics, cosine_metrics = evaluate_context_retrieval(
        questions, answers, som_contexts, cosine_contexts, 
        max_workers=args.workers,
        use_ray=args.use_ray,
        ray_evaluators=args.ray_evaluators,
        batch_size=args.batch_size
    )
    
    # Print results
    print_evaluation_summary(df, som_metrics, cosine_metrics)
    
    # Create visualization
    create_comparison_chart(som_metrics, cosine_metrics)
    
    # Save results
    results_file = f"scaled_evaluation_results_{len(questions)}_questions.csv"
    df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to {results_file}")
    
    # Save metrics
    metrics_file = f"scaled_metrics_{len(questions)}_questions.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'som_metrics': som_metrics,
            'cosine_metrics': cosine_metrics,
            'som_confusion_matrix': som_cm.tolist(),
            'cosine_confusion_matrix': cosine_cm.tolist(),
            'total_questions': len(questions)
        }, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 