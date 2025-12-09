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
from datetime import datetime
import re

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
    
    def evaluate_context_contains_answer(self, question: str, answer: str, context: list) -> bool:
        """Evaluate if context contains the answer using Ollama"""

        prompt = f"""You are evaluating whether a given context contains the answer to a question.

        Question: {question}
        Expected Answer: {answer}
        Context: {context}

        Does the context contain the expected answer to the question? Respond with only "TRUE" if the context contains the answer.
        or else "FALSE".

        Response:"""
        
        response = self.query_llm(prompt)
        print(f"LLM response: {response} | Question: {question[:20]}")
        return response.upper().startswith("TRUE")

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

def create_comparison_chart(som_metrics: Dict, cosine_metrics: Dict, save_path: str = "results/scaled_comparison_chart.png"):
    """Create a bar chart comparing SOM vs Cosine metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    som_values = [som_metrics[metric] for metric in metrics]
    cosine_values = [cosine_metrics[metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))

    patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

    bars1 = ax.bar(x - width/2, som_values, width, label='SOMpy', fill=False, edgecolor='black', linewidth=2, hatch=patterns[0]*3)
    bars2 = ax.bar(x + width/2, cosine_values, width, label='Cosine Similarity', fill=False, edgecolor='black', linewidth=2, hatch=patterns[6]*3)
    
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
    print(f"  - Accuracy: {som_metrics['Accuracy']:.3f}")
    print(f"  - Precision: {som_metrics['Precision']:.3f}")
    print(f"  - Recall: {som_metrics['Recall']:.3f}")
    print(f"  - F1 Score: {som_metrics['F1 Score']:.3f}")
    print(f"  - Questions with correct context: {df['som_contains_answer'].sum()}")
    
    print(f"\nCosine Similarity Performance:")
    print(f"  - Accuracy: {cosine_metrics['Accuracy']:.3f}") 
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

def _process_results(results):
    """Process evaluation results and calculate metrics according to user's definitions"""
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Since we always attempt retrieval, we can calculate:
    total_questions = len(df)
    
    # Determine which questions have good contexts available
    # If either method found a good context, we know good contexts exist for that question
    questions_with_good_contexts = (df['som_contains_answer'] | df['cosine_contains_answer']).sum()
    
    # Note on the Confusion Matrix
    # TP: True Positives - Retrieved context contains the true answer
    # FP: False Positives - Retrieved context claims to contain answer, but answer is wrong or nonexistent
    # FN: False Negatives - Answer exists in corpus, but retrieved context does NOT contain it
    # TN: True Negatives - Answer does NOT exist in corpus, and retrieved context also does NOT contain it

    # For SOM method
    som_tp = df['som_contains_answer'].sum()  # SOM found good contexts
    som_fp = total_questions - som_tp  # SOM retrieved bad contexts 
    # FN: Questions where good contexts exist (either method found them) but SOM missed them
    som_fn = df[df['cosine_contains_answer'] & ~df['som_contains_answer']].shape[0]
    # TN: assume all questions have answers in the corpus
    som_tn = 0
    print(f"som_tp: {som_tp}, som_fp: {som_fp}, som_fn: {som_fn}, som_tn: {som_tn}")
    
    # For Cosine method  
    cosine_tp = df['cosine_contains_answer'].sum()  # Cosine found good contexts
    cosine_fp = total_questions - cosine_tp  # Cosine retrieved bad contexts
    # FN: Questions where good contexts exist (either method found them) but Cosine missed them  
    cosine_fn = df[df['som_contains_answer'] & ~df['cosine_contains_answer']].shape[0]
    # TN: assume all questions have answers in the corpus
    cosine_tn = 0
    print(f"cosine_tp: {cosine_tp}, cosine_fp: {cosine_fp}, cosine_fn: {cosine_fn}, cosine_tn: {cosine_tn}")

    # Calculate metrics using the standard formulas
    som_precision = som_tp / (som_tp + som_fp) if (som_tp + som_fp) > 0 else 0
    som_recall = som_tp / (som_tp + som_fn) if (som_tp + som_fn) > 0 else 0
    som_f1 = 2 * (som_precision * som_recall) / (som_precision + som_recall) if (som_precision + som_recall) > 0 else 0
    som_accuracy = (som_tp + som_tn) / (som_tp + som_fp + som_fn + som_tn) if total_questions > 0 else 0
    print(f"som_precision: {som_precision}, som_recall: {som_recall}, som_f1: {som_f1}, som_accuracy: {som_accuracy}")
    
    cosine_precision = cosine_tp / (cosine_tp + cosine_fp) if (cosine_tp + cosine_fp) > 0 else 0
    cosine_recall = cosine_tp / (cosine_tp + cosine_fn) if (cosine_tp + cosine_fn) > 0 else 0
    cosine_f1 = 2 * (cosine_precision * cosine_recall) / (cosine_precision + cosine_recall) if (cosine_precision + cosine_recall) > 0 else 0
    cosine_accuracy = (cosine_tp + cosine_tn) / (cosine_tp + cosine_fp + cosine_fn + cosine_tn) if total_questions > 0 else 0
    print(f"cosine_precision: {cosine_precision}, cosine_recall: {cosine_recall}, cosine_f1: {cosine_f1}, cosine_accuracy: {cosine_accuracy}")
    
    # Create confusion matrices
    som_cm = np.array([[som_tn, som_fp], [som_fn, som_tp]])
    cosine_cm = np.array([[cosine_tn, cosine_fp], [cosine_fn, cosine_tp]])

    # Package metrics
    som_metrics = {
        'Accuracy': som_accuracy,
        'Precision': som_precision,
        'Recall': som_recall,
        'F1 Score': som_f1
    }
    
    cosine_metrics = {
        'Accuracy': cosine_accuracy,
        'Precision': cosine_precision,
        'Recall': cosine_recall,
        'F1 Score': cosine_f1
    }
    
    return df, som_cm, cosine_cm, som_metrics, cosine_metrics

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

def _evaluate_standard(questions, answers, som_contexts, cosine_contexts, max_workers):
    """Standard evaluation using ThreadPoolExecutor"""
    evaluator = OllamaEvaluator()
    
    args_list = [
        (question, answer, som_ctx, cos_ctx, evaluator)
        for question, answer, som_ctx, cos_ctx in zip(questions, answers, som_contexts, cosine_contexts)
    ]

    results = []
    batch_size = 50
    for i in tqdm(range(0, len(args_list), batch_size), desc="Processing batches"):
        batch = args_list[i:i + batch_size]
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for arg in batch:
                futures.append(executor.submit(evaluate_single_item, arg))
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    # log and optionally retry or append a failure placeholder
                    print("Task failed:", e)
                    results.append({'question': None, 'error': str(e)})
        time.sleep(1)
    
    return _process_results(results)

# def _evaluate_with_ray(questions, answers, som_contexts, cosine_contexts, ray_evaluators, batch_size):
#     """Ray distributed evaluation"""
#     # Initialize Ray if not already done
#     if not ray.is_initialized():
#         ray.init(address='auto', ignore_reinit_error=True)
    
#     # Create evaluator actors
#     evaluators = [RayOllamaEvaluator.remote() for _ in range(ray_evaluators)]
    
#     # Prepare data batches
#     data = list(zip(questions, answers, som_contexts, cosine_contexts))
#     batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
#     # Distribute work across evaluators
#     futures = []
#     for i, batch in enumerate(batches):
#         evaluator = evaluators[i % len(evaluators)]
#         future = evaluator.evaluate_batch.remote(batch)
#         futures.append(future)
    
#     # Collect results
#     print(f"Processing {len(batches)} batches with {len(evaluators)} evaluators...")
#     results = []
#     for future in tqdm(ray.get(futures), desc="Collecting results"):
#         results.extend(future)
    
#     return _process_results(results)

def _evaluate_with_ray(questions, answers, som_contexts, cosine_contexts, ray_evaluators, batch_size):
    """Ray distributed evaluation"""
    # Initialize Ray if not already done
    if not ray.is_initialized():
        # Try to connect to an existing cluster first; if that fails, start a local Ray instance.
        try:
            ray_address = os.environ.get("RAY_ADDRESS", "auto")
            ray.init(address=ray_address, ignore_reinit_error=True)
            print(f"Connected to Ray at address={ray_address}")
        except Exception as e:
            print(f"Warning: failed to connect to a Ray cluster ({e}). Starting a local Ray instance.")
            try:
                # Ensure any previous state is cleared
                try:
                    ray.shutdown()
                except Exception:
                    pass
                ray.init(ignore_reinit_error=True)
                print("Started local Ray instance via ray.init().")
            except Exception as e2:
                # If even local init fails, raise and let caller handle fallback
                print(f"Error: failed to start local Ray instance ({e2}). Falling back to standard evaluation.")
                # Fall back to standard (non-Ray) evaluation
                return _evaluate_standard(questions, answers, som_contexts, cosine_contexts, max_workers=4)
    
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
        return _evaluate_with_ray(questions, answers, som_contexts, cosine_contexts, ray_evaluators, batch_size)
    else:
        print(f"Using standard evaluation with {max_workers} workers")
        return _evaluate_standard(questions, answers, som_contexts, cosine_contexts, max_workers)

def load_from_csv_results(csv_file: str) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """
    Load Question, Answer, SOM contexts and Cosine contexts from a CSV where each context column contains a stringified list.
    Columns: 'question' | 'answer' | 'som_context' | 'cosine_context'
    """
    try:
        df = pd.read_csv(csv_file)
        if 'som_context' not in df.columns or 'cosine_context' not in df.columns or 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("CSV must contain 'question', 'answer', 'som_context' and 'cosine_context' columns")

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

        questions = df['question'].astype(str).str.strip().tolist()
        answers = df['answer'].astype(str).str.strip().tolist()
        som_contexts = df['som_context'].apply(safe_eval).tolist()
        cosine_contexts = df['cosine_context'].apply(safe_eval).tolist()

        print(f"Loaded {len(questions)} questions and answers from {csv_file}")
        print(f"Loaded {len(som_contexts)} context pairs from {csv_file}")
        return questions, answers, som_contexts, cosine_contexts
    except Exception as e:
        print(f"Error loading questions, answers, and contexts from CSV: {e}")
        return [], [], [], []

def avg_confidence(conf_list):
    if conf_list is None or len(conf_list) == 0:
        return 0.0
    return float(np.mean(conf_list))

def main():
    """Main function to run the scaled evaluation"""
    parser = argparse.ArgumentParser(description='Scale up context evaluation with actual data')
    parser.add_argument('--contexts_file', type=str, default='retrieved_contexts',
                        help='Base path for context files (without extension)')
    parser.add_argument('--pkl_file', type=str, default='contexts/retrieved_contexts.pkl', 
                       help='Path to Excel file with questions and answers')
    parser.add_argument('--csv_file', type=str, default='contexts/retrieved_contexts.csv',
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
    parser.add_argument('--sampling', type=str, default='random', choices=['sequential', 'random'],
                       help='Question sampling method: sequential (first N) or random')
    
    args = parser.parse_args()

    args.pkl_file = f"contexts/{args.contexts_file}.pkl"
    args.csv_file = f"contexts/{args.contexts_file}.csv"
    
    print("="*60)
    print("SCALE UP EVALUATION WITH ACTUAL DATA")
    print("="*60)
    
    # Check Ray availability
    if args.use_ray and not RAY_AVAILABLE:
        print("Warning: Ray is not available. Falling back to standard evaluation.")
        args.use_ray = False
    
    if args.use_ray:
        print(f"Ray distributed evaluation enabled with {args.ray_evaluators} evaluators")

    loaded = False

    # Load questions, answers, and contexts from pickle file
    if args.pkl_file and os.path.exists(args.pkl_file):
        try:
            print(f"Loading questions, answers and contexts from pickle: {args.pkl_file}")
            with open(args.pkl_file, 'rb') as f:
                retrieved = pickle.load(f)
            questions = [item["question"] for item in retrieved]
            answers = [item["answer"] for item in retrieved]
            som_contexts = [item["som_contexts"] for item in retrieved]
            som_scores = [item["som_scores"] for item in retrieved]
            cosine_contexts = [item["cosine_contexts"] for item in retrieved]
            cosine_scores = [item["cosine_scores"] for item in retrieved]
            print(f"Loaded {len(questions)} questions, answers and contexts from pickle: {args.pkl_file}")
            loaded = True
        except Exception as e:
            print(f"Error loading pickle file {args.pkl_file}: {e}")

    # # Potential pickle locations (project root and Self-Organizing-Maps subdir)
    # possible_paths = [
    #     ("contexts/som_contexts_scores.pkl", "contexts/cosine_contexts_scores.pkl"),
    #     ("Self-Organizing-Maps/contexts/som_contexts_scores.pkl", "Self-Organizing-Maps/contexts/cosine_contexts_scores.pkl"),
    #     (os.path.join(os.getcwd(), "contexts/som_contexts_scores.pkl"), os.path.join(os.getcwd(), "contexts/cosine_contexts_scores.pkl")),
    # ]

    # for som_p, cos_p in possible_paths:
    #     if os.path.exists(som_p) and os.path.exists(cos_p):
    #         try:
    #             with open(som_p, 'rb') as f:
    #                 som_contexts_scores = pickle.load(f)
    #             with open(cos_p, 'rb') as f:
    #                 cosine_contexts_scores = pickle.load(f)

    #             print(f"Loaded {len(som_contexts_scores)} context pairs from pickles: {som_p}, {cos_p}")

    #             # Convert to the format expected by evaluation: extract context lists
    #             som_contexts = [ctx[0] if isinstance(ctx, (list, tuple)) and len(ctx) > 0 else ctx for ctx in som_contexts_scores]
    #             cosine_contexts = [ctx[0] if isinstance(ctx, (list, tuple)) and len(ctx) > 0 else ctx for ctx in cosine_contexts_scores]
    #             loaded = True
    #             break
    #         except Exception as e:
    #             print(f"Error loading pickles from {som_p} / {cos_p}: {e}")

    # Fallback to CSV if pickle loading failed
    if not loaded and args.csv_file and os.path.exists(args.csv_file):
        print(f"Loading questions, answers and contexts from CSV: {args.csv_file}")
        questions, answers, som_contexts, som_scores, cosine_contexts, cosine_scores = load_from_csv_results(args.csv_file)
        print(f"Loaded {len(questions)} questions, answers and contexts from CSV: {args.csv_file}")
        if not questions or not answers or not som_contexts or not cosine_contexts:
            print("Failed to load Q&A data and contexts. Exiting.")
            return
        loaded = True
    
    # Limit to max_questions
    if len(questions) > args.max_questions:
        print(f"Limiting evaluation to {args.max_questions} questions (from {len(questions)} total)")
        
        if args.sampling == 'random':
            # Random sampling for balanced difficulty distribution
            import random
            random.seed(42)  # For reproducible results
            indices = random.sample(range(len(questions)), args.max_questions)
            indices.sort()  # Keep some order for consistency
            questions = [questions[i] for i in indices]
            answers = [answers[i] for i in indices]
            som_contexts = [som_contexts[i] for i in indices]
            som_scores = [som_scores[i] for i in indices]
            cosine_contexts = [cosine_contexts[i] for i in indices]
            cosine_scores = [cosine_scores[i] for i in indices]
            print(f"Using random sampling (seed=42) for representative evaluation")
        else:
            # Sequential sampling (first N questions)
            questions = questions[:args.max_questions]
            answers = answers[:args.max_questions]
            som_contexts = som_contexts[:args.max_questions]
            som_scores = som_scores[:args.max_questions]
            cosine_contexts = cosine_contexts[:args.max_questions]
            cosine_scores = cosine_scores[:args.max_questions]
            print(f"Using sequential sampling (first {args.max_questions} questions)")
    
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

    # Calculate average confidence scores and add to DataFrame
    df['som_avg_confidence'] = [avg_confidence(scores) for scores in som_scores]
    df['cosine_avg_confidence'] = [avg_confidence(scores) for scores in cosine_scores]
    
    # Print results
    print_evaluation_summary(df, som_metrics, cosine_metrics)
    
    # Create visualization
    chart_file = f"evaluation_results_logs/comparison_chart_{len(questions)}_questions_{args.contexts_file}.png"
    create_comparison_chart(som_metrics, cosine_metrics, chart_file)
    
    # Save results : CSV
    results_file = f"evaluation_results_logs/evaluation_results_{len(questions)}_questions_{args.contexts_file}.csv"
    df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to {results_file}")

    # Save results : pickle
    results_pickle_file = f"evaluation_results_logs/evaluation_results_{len(questions)}_questions_{args.contexts_file}.pkl"
    with open(results_pickle_file, 'wb') as f:
        pickle.dump(df, f)
    print(f"Detailed results pickle saved to {results_pickle_file}")
    
    # Save metrics
    metrics_file = f"evaluation_results_logs/metrics_{len(questions)}_questions_{args.contexts_file}.json"
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