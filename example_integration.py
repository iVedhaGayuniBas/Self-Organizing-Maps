"""
Example integration script showing how to use the context evaluation with your existing SOM and cosine data.
This script demonstrates how to connect the evaluation framework with your actual context retrieval results.
"""

import pandas as pd
import numpy as np
from context_evaluation_with_ollama import (
    load_qa_data, 
    evaluate_context_retrieval, 
    create_comparison_chart, 
    print_evaluation_summary
)

def load_existing_contexts():
    """
    Load your existing SOM and cosine contexts from your notebook results.
    Replace this function with your actual data loading logic.
    """
    
    # Example: Load from CSV files if you saved them
    try:
        # If you have saved your contexts to CSV files
        som_df = pd.read_csv("som_contexts_results.csv")
        cosine_df = pd.read_csv("cosine_contexts_results.csv")
        
        # Extract contexts and scores
        som_contexts_scores = []
        cosine_contexts_scores = []
        
        for _, row in som_df.iterrows():
            contexts = eval(row['contexts'])  # Assuming contexts are stored as string representation
            scores = eval(row['scores'])
            som_contexts_scores.append((contexts, scores))
        
        for _, row in cosine_df.iterrows():
            contexts = eval(row['contexts'])
            scores = eval(row['scores'])
            cosine_contexts_scores.append((contexts, scores))
            
        return som_contexts_scores, cosine_contexts_scores
        
    except FileNotFoundError:
        print("Context files not found. Using dummy data for demonstration.")
        return create_dummy_contexts()

def create_dummy_contexts():
    """Create dummy context data for demonstration"""
    
    # Sample contexts that might be retrieved
    sample_contexts = [
        "The 24-hour clock is a way of telling the time in which the day runs from midnight to midnight and is divided into 24 hours, numbered from 0 to 23. It does not use a.m. or p.m. This system is also referred to as military time. Also, the international standard notation of time (ISO 8601) is based on this format.",
        "Jews divide the Hebrew Bible into three parts and call it the Tanakh. The three parts are the Torah, which is the first five books; the Nevi'im, which are the books of the prophets; and the Ketuvim, meaning the Writings.",
        "The Dark Knight Rises was filmed in multiple cities including Pittsburgh, Pennsylvania; New York City, New York; and Los Angeles, California.",
        "Cristiano Ronaldo scored his famous bicycle-kick goal against Juventus in the 2018 UEFA Champions League quarter-finals.",
        "The Schleswig-Holstein court ruled on 5 April 2018 that Puigdemont would not be extradited on charges of rebellion.",
        "In digital logic, the On state in a logic gate typically uses a voltage range of 3.5 to 5 volts.",
        "Canada's national capital is Ottawa, where the federal government meets.",
        "According to the Copenhagen Interpretation of quantum mechanics, the cat is in a superposition state of both dead and alive before the box is opened.",
        "Carbon dioxide (CO2) is identified as the main cause of global warming due to burning fossil fuels.",
        "Mahatma Gandhi used peaceful tactics including 'ahimsa' (non-violence) to lead the freedom movement against British rule in India."
    ]
    
    som_contexts_scores = []
    cosine_contexts_scores = []
    
    for i, context in enumerate(sample_contexts):
        # SOM contexts (might be slightly different)
        som_contexts = [context + " (SOM retrieval)"]
        som_scores = [0.8 + (i * 0.02)]  # Varying scores
        som_contexts_scores.append((som_contexts, som_scores))
        
        # Cosine contexts (might be slightly different)
        cosine_contexts = [context + " (Cosine retrieval)"]
        cosine_scores = [0.9 + (i * 0.01)]  # Varying scores
        cosine_contexts_scores.append((cosine_contexts, cosine_scores))
    
    return som_contexts_scores, cosine_contexts_scores

def integrate_with_notebook_results():
    """
    Example of how to integrate with your notebook results.
    Replace the dummy data with your actual SOM and cosine context results.
    """
    
    # Configuration
    EXCEL_FILE = "questions_answers.xlsx"
    MAX_WORKERS = 8  # Adjust based on your GPU setup
    
    print("Loading question-answer data...")
    questions, answers = load_qa_data(EXCEL_FILE)
    
    print("Loading context data...")
    som_contexts_scores, cosine_contexts_scores = load_existing_contexts()
    
    # Validate data lengths
    assert len(questions) == len(answers) == len(som_contexts_scores) == len(cosine_contexts_scores), \
        f"Data length mismatch: questions={len(questions)}, answers={len(answers)}, " \
        f"som_contexts={len(som_contexts_scores)}, cosine_contexts={len(cosine_contexts_scores)}"
    
    print(f"Evaluating {len(questions)} questions with {MAX_WORKERS} parallel workers...")
    
    # Run evaluation
    df, som_cm, cosine_cm, som_metrics, cosine_metrics = evaluate_context_retrieval(
        questions, answers, som_contexts_scores, cosine_contexts_scores, MAX_WORKERS
    )
    
    # Print results
    print_evaluation_summary(df, som_metrics, cosine_metrics)
    
    # Create visualization
    create_comparison_chart(som_metrics, cosine_metrics, "som_vs_cosine_comparison.png")
    
    # Save detailed results
    df.to_csv("detailed_evaluation_results.csv", index=False)
    
    # Print confusion matrices
    print("\nSOMpy Confusion Matrix:")
    print(som_cm)
    print("\nCosine Similarity Confusion Matrix:")
    print(cosine_cm)
    
    return df, som_metrics, cosine_metrics

def convert_notebook_data_format():
    """
    Example function showing how to convert your notebook data format to the evaluation format.
    Based on your notebook, you might have data in this format:
    """
    
    # Example: Your notebook might have data like this:
    # som_contexts_scores = [
    #     (['context1', 'context2'], [0.8, 0.7]),
    #     (['context3', 'context4'], [0.9, 0.6]),
    #     ...
    # ]
    
    # cosine_contexts_scores = [
    #     (['context1', 'context2'], [0.85, 0.75]),
    #     (['context3', 'context4'], [0.95, 0.65]),
    #     ...
    # ]
    
    # The evaluation function expects exactly this format!
    # So if your notebook produces this format, you can use it directly.
    
    pass

if __name__ == "__main__":
    # Run the integration example
    try:
        df, som_metrics, cosine_metrics = integrate_with_notebook_results()
        print("\n✅ Evaluation completed successfully!")
        print("📊 Check the generated files:")
        print("   - detailed_evaluation_results.csv")
        print("   - som_vs_cosine_comparison.png")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Make sure:")
        print("1. Ollama is running with llama3:latest model")
        print("2. questions_answers.xlsx file exists")
        print("3. Replace dummy context data with your actual SOM and cosine results") 