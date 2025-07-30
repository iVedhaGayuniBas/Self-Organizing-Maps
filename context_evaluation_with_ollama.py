import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import requests
import json
import time
import concurrent.futures
from typing import List, Tuple, Dict, Any
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaEvaluator:
    """Evaluator class using Ollama for context retrieval evaluation"""
    
    def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
        
    def query_llm(self, prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
        """Query Ollama model with retry logic"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent evaluation
                "top_p": 0.9,
                "num_predict": 50  # Limit response length
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get('response', '').strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"All retries failed for prompt: {prompt[:100]}...")
                    return "No"  # Fail-safe response
    
    def evaluate_context_contains_answer(self, question: str, answer: str, context: str) -> bool:
        """Evaluate if context contains the answer using Ollama"""
        
        # Create a clear evaluation prompt
        prompt = f"""You are an expert evaluator. Your task is to determine if the given context contains the correct answer to a question.

Question: {question}
Expected Answer: {answer}
Context: {context}

Does the context clearly contain or imply the correct answer to the question? Consider:
1. Is the answer explicitly stated in the context?
2. Can the answer be reasonably inferred from the context?
3. Is the context relevant to the question?

Respond with ONLY "Yes" or "No"."""

        response = self.query_llm(prompt)
        
        # Parse response
        response_lower = response.lower().strip()
        if "yes" in response_lower:
            return True
        elif "no" in response_lower:
            return False
        else:
            # If unclear, default to False for conservative evaluation
            logger.warning(f"Unclear response: '{response}'. Defaulting to False.")
            return False

def load_qa_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load questions and answers from Excel file"""
    try:
        df = pd.read_excel(file_path)
        
        # Check if required columns exist
        if 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("Excel file must contain 'question' and 'answer' columns")
        
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        
        # Remove any empty entries
        valid_data = [(q, a) for q, a in zip(questions, answers) if pd.notna(q) and pd.notna(a)]
        questions, answers = zip(*valid_data)
        
        logger.info(f"Loaded {len(questions)} question-answer pairs from {file_path}")
        return list(questions), list(answers)
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def evaluate_single_item(args: Tuple) -> Dict[str, Any]:
    """Evaluate a single question-answer pair with both SOM and cosine contexts"""
    question, answer, som_context, cosine_context, evaluator = args
    
    # Evaluate SOM context
    som_contains_answer = evaluator.evaluate_context_contains_answer(question, answer, som_context)
    
    # Small delay to avoid overwhelming the model
    time.sleep(0.5)
    
    # Evaluate cosine context
    cosine_contains_answer = evaluator.evaluate_context_contains_answer(question, answer, cosine_context)
    
    return {
        'question': question,
        'answer': answer,
        'som_contains_answer': som_contains_answer,
        'cosine_contains_answer': cosine_contains_answer
    }

def evaluate_context_retrieval(questions: List[str], 
                             answers: List[str], 
                             som_contexts_scores: List[Tuple[List[str], List[float]]], 
                             cosine_contexts_scores: List[Tuple[List[str], List[float]]],
                             max_workers: int = 4) -> Tuple[pd.DataFrame, Dict, Dict, Dict, Dict]:
    """
    Evaluate context retrieval performance using Ollama
    
    Args:
        questions: List of questions
        answers: List of answers
        som_contexts_scores: List of (contexts, scores) tuples from SOM
        cosine_contexts_scores: List of (contexts, scores) tuples from cosine similarity
        max_workers: Number of parallel workers
    
    Returns:
        DataFrame with results, confusion matrices, and metrics
    """
    
    # Validate input lengths
    assert len(questions) == len(answers) == len(som_contexts_scores) == len(cosine_contexts_scores), \
        "All input lists must have the same length"
    
    logger.info(f"Starting evaluation of {len(questions)} questions with {max_workers} workers")
    
    # Initialize evaluator
    evaluator = OllamaEvaluator()
    
    # Prepare contexts for evaluation
    som_contexts = []
    cosine_contexts = []
    
    for (som_ctx, som_scores), (cos_ctx, cos_scores) in zip(som_contexts_scores, cosine_contexts_scores):
        # Combine contexts into single strings
        som_contexts.append(" ".join(som_ctx))
        cosine_contexts.append(" ".join(cos_ctx))
    
    # Prepare arguments for parallel processing
    args_list = [(q, a, som_ctx, cos_ctx, evaluator) 
                 for q, a, som_ctx, cos_ctx in zip(questions, answers, som_contexts, cosine_contexts)]
    
    # Process evaluations in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for progress tracking
        futures = [executor.submit(evaluate_single_item, args) for args in args_list]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Evaluating contexts"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in evaluation: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate metrics
    som_metrics = {
        'Precision': precision_score(df['som_contains_answer'], df['som_contains_answer']),
        'Recall': recall_score(df['som_contains_answer'], df['som_contains_answer']),
        'F1 Score': f1_score(df['som_contains_answer'], df['som_contains_answer'])
    }
    
    cosine_metrics = {
        'Precision': precision_score(df['cosine_contains_answer'], df['cosine_contains_answer']),
        'Recall': recall_score(df['cosine_contains_answer'], df['cosine_contains_answer']),
        'F1 Score': f1_score(df['cosine_contains_answer'], df['cosine_contains_answer'])
    }
    
    # Create confusion matrices
    som_cm = confusion_matrix(df['som_contains_answer'], df['som_contains_answer'])
    cosine_cm = confusion_matrix(df['cosine_contains_answer'], df['cosine_contains_answer'])
    
    logger.info("Evaluation completed successfully")
    
    return df, som_cm, cosine_cm, som_metrics, cosine_metrics

def create_comparison_chart(som_metrics: Dict, cosine_metrics: Dict, save_path: str = "comparison_chart.png"):
    """Create a bar chart comparing SOM vs Cosine performance"""
    
    metrics = ['Precision', 'Recall', 'F1 Score']
    som_values = [som_metrics[metric] for metric in metrics]
    cosine_values = [cosine_metrics[metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, som_values, width, label='SOMpy', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, cosine_values, width, label='Cosine Similarity', color='#A23B72', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('SOMpy vs Cosine Similarity: Context Retrieval Performance', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add some styling
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Comparison chart saved to {save_path}")

def print_evaluation_summary(df: pd.DataFrame, som_metrics: Dict, cosine_metrics: Dict):
    """Print a summary of evaluation results"""
    
    print("\n" + "="*60)
    print("CONTEXT RETRIEVAL EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal questions evaluated: {len(df)}")
    
    print(f"\nSOMpy Performance:")
    print(f"  - Precision: {som_metrics['Precision']:.3f}")
    print(f"  - Recall: {som_metrics['Recall']:.3f}")
    print(f"  - F1 Score: {som_metrics['F1 Score']:.3f}")
    
    print(f"\nCosine Similarity Performance:")
    print(f"  - Precision: {cosine_metrics['Precision']:.3f}")
    print(f"  - Recall: {cosine_metrics['Recall']:.3f}")
    print(f"  - F1 Score: {cosine_metrics['F1 Score']:.3f}")
    
    # Calculate improvement
    som_f1 = som_metrics['F1 Score']
    cosine_f1 = cosine_metrics['F1 Score']
    
    if som_f1 > cosine_f1:
        improvement = ((som_f1 - cosine_f1) / cosine_f1) * 100
        print(f"\nSOMpy outperforms Cosine by {improvement:.1f}% in F1 Score")
    elif cosine_f1 > som_f1:
        improvement = ((cosine_f1 - som_f1) / som_f1) * 100
        print(f"\nCosine outperforms SOMpy by {improvement:.1f}% in F1 Score")
    else:
        print(f"\nBoth methods perform equally well")
    
    print("="*60)

def main():
    """Main function to run the evaluation"""
    
    # Configuration
    EXCEL_FILE = "questions_answers.xlsx"
    MAX_WORKERS = 8  # Adjust based on your GPU setup
    
    try:
        # Load data
        logger.info("Loading question-answer data...")
        questions, answers = load_qa_data(EXCEL_FILE)
        
        # TODO: Replace these with your actual SOM and cosine context data
        # For now, creating dummy data for demonstration
        logger.warning("Using dummy context data. Replace with your actual SOM and cosine contexts.")
        
        # Create dummy contexts (replace with your actual data)
        som_contexts_scores = []
        cosine_contexts_scores = []
        
        for i in range(len(questions)):
            # Dummy SOM contexts
            som_contexts = [f"SOM context {i} - sample text about the topic"]
            som_scores = [0.8]
            som_contexts_scores.append((som_contexts, som_scores))
            
            # Dummy cosine contexts
            cosine_contexts = [f"Cosine context {i} - sample text about the topic"]
            cosine_scores = [0.9]
            cosine_contexts_scores.append((cosine_contexts, cosine_scores))
        
        # Run evaluation
        logger.info("Starting context retrieval evaluation...")
        df, som_cm, cosine_cm, som_metrics, cosine_metrics = evaluate_context_retrieval(
            questions, answers, som_contexts_scores, cosine_contexts_scores, MAX_WORKERS
        )
        
        # Print results
        print_evaluation_summary(df, som_metrics, cosine_metrics)
        
        # Create visualization
        create_comparison_chart(som_metrics, cosine_metrics)
        
        # Save results
        df.to_csv("evaluation_results.csv", index=False)
        logger.info("Results saved to evaluation_results.csv")
        
        # Print confusion matrices
        print("\nSOMpy Confusion Matrix:")
        print(som_cm)
        print("\nCosine Similarity Confusion Matrix:")
        print(cosine_cm)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 