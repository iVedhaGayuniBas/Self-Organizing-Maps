# Context Retrieval Evaluation with Ollama

This project provides a comprehensive evaluation framework for comparing SOMpy vs Cosine Similarity context retrieval performance using a locally hosted LLM (Ollama).

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Ollama installed and running
- llama3:latest model downloaded
- 10 GPUs available (for parallel processing)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure Ollama is Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If llama3:latest is not available, pull it
ollama pull llama3:latest
```

### 4. Prepare Your Data

Create an Excel file named `questions_answers.xlsx` with columns:
- `question`: Your questions
- `answer`: Expected answers

### 5. Run Evaluation

```bash
python context_evaluation_with_ollama.py
```

## 📊 Features

- **Local LLM Evaluation**: Uses Ollama with llama3:latest instead of cloud APIs
- **Parallel Processing**: Utilizes multiple GPUs for faster evaluation
- **Comprehensive Metrics**: Precision, Recall, F1 Score, and Confusion Matrices
- **Visualization**: Automatic generation of comparison charts
- **Robust Error Handling**: Retry logic and fail-safe responses
- **Progress Tracking**: Real-time progress bars for long evaluations

## 🔧 Configuration

### Model Configuration

The evaluation uses `llama3:latest` by default. You can modify the model in `OllamaEvaluator`:

```python
evaluator = OllamaEvaluator(model_name="llama3:latest")
```

### Parallel Processing

Adjust the number of workers based on your GPU setup:

```python
MAX_WORKERS = 8  # Adjust based on your GPU count
```

### Evaluation Parameters

- **Temperature**: 0.1 (low for consistent evaluation)
- **Top-p**: 0.9
- **Max tokens**: 50 (sufficient for Yes/No responses)
- **Timeout**: 30 seconds per request
- **Retries**: 3 attempts with exponential backoff

## 📁 File Structure

```
Self-Organizing-Maps/
├── context_evaluation_with_ollama.py  # Main evaluation script
├── example_integration.py             # Integration examples
├── requirements.txt                   # Python dependencies
├── README_evaluation.md              # This file
├── questions_answers.xlsx            # Your Q&A data
└── evaluation_results/               # Generated results
    ├── detailed_evaluation_results.csv
    ├── som_vs_cosine_comparison.png
    └── evaluation_summary.txt
```

## 🔄 Integration with Your Existing Code

### Option 1: Direct Integration

If your notebook produces data in the expected format:

```python
# Your notebook data format
som_contexts_scores = [
    (['context1', 'context2'], [0.8, 0.7]),
    (['context3', 'context4'], [0.9, 0.6]),
    ...
]

cosine_contexts_scores = [
    (['context1', 'context2'], [0.85, 0.75]),
    (['context3', 'context4'], [0.95, 0.65]),
    ...
]

# Use directly with evaluation
df, som_cm, cosine_cm, som_metrics, cosine_metrics = evaluate_context_retrieval(
    questions, answers, som_contexts_scores, cosine_contexts_scores, max_workers=8
)
```

### Option 2: Save and Load

Save your contexts to CSV files:

```python
# In your notebook
import pandas as pd

# Save SOM contexts
som_data = []
for i, (contexts, scores) in enumerate(som_contexts_scores):
    som_data.append({
        'question_id': i,
        'contexts': str(contexts),
        'scores': str(scores)
    })
pd.DataFrame(som_data).to_csv('som_contexts_results.csv', index=False)

# Save cosine contexts
cosine_data = []
for i, (contexts, scores) in enumerate(cosine_contexts_scores):
    cosine_data.append({
        'question_id': i,
        'contexts': str(contexts),
        'scores': str(scores)
    })
pd.DataFrame(cosine_data).to_csv('cosine_contexts_results.csv', index=False)
```

Then load in the evaluation script:

```python
# In evaluation script
som_contexts_scores, cosine_contexts_scores = load_existing_contexts()
```

## 📈 Output

### 1. Console Output

```
============================================================
CONTEXT RETRIEVAL EVALUATION SUMMARY
============================================================

Total questions evaluated: 5000

SOMpy Performance:
  - Precision: 0.847
  - Recall: 0.823
  - F1 Score: 0.835

Cosine Similarity Performance:
  - Precision: 0.891
  - Recall: 0.876
  - F1 Score: 0.883

Cosine outperforms SOMpy by 5.7% in F1 Score
============================================================
```

### 2. Generated Files

- `detailed_evaluation_results.csv`: Complete evaluation results
- `som_vs_cosine_comparison.png`: Bar chart comparison
- `evaluation_summary.txt`: Summary statistics

### 3. Confusion Matrices

```
SOMpy Confusion Matrix:
[[1234  156]
 [ 234  456]]

Cosine Similarity Confusion Matrix:
[[1345   45]
 [ 123  567]]
```

## ⚡ Performance Optimization

### GPU Utilization

With 10 GPUs, you can run up to 10 parallel evaluations:

```python
MAX_WORKERS = 10  # One per GPU
```

### Batch Processing

For large datasets, consider processing in batches:

```python
BATCH_SIZE = 1000
for i in range(0, len(questions), BATCH_SIZE):
    batch_questions = questions[i:i+BATCH_SIZE]
    batch_answers = answers[i:i+BATCH_SIZE]
    # Process batch...
```

### Memory Management

The script includes automatic memory management, but for very large datasets:

```python
# Clear memory between batches
import gc
gc.collect()
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   # Start Ollama
   ollama serve
   ```

2. **Model not found**
   ```bash
   # Pull the model
   ollama pull llama3:latest
   ```

3. **Memory issues**
   - Reduce `MAX_WORKERS`
   - Process in smaller batches
   - Restart Ollama service

4. **Timeout errors**
   - Increase timeout in `query_llm` method
   - Check network connectivity
   - Reduce parallel workers

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔬 Advanced Usage

### Custom Evaluation Prompts

Modify the evaluation prompt in `evaluate_context_contains_answer`:

```python
prompt = f"""Your custom evaluation prompt here.
Question: {question}
Answer: {answer}
Context: {context}
"""
```

### Multiple Model Comparison

Compare different Ollama models:

```python
models = ["llama3:latest", "llama3.2:latest", "llava:latest"]
for model in models:
    evaluator = OllamaEvaluator(model_name=model)
    # Run evaluation...
```

### Custom Metrics

Add your own evaluation metrics:

```python
def custom_metric(df):
    # Your custom metric calculation
    return custom_score

# Add to evaluation results
results['custom_metric'] = custom_metric(df)
```

## 📚 API Reference

### OllamaEvaluator

Main class for LLM-based evaluation.

```python
evaluator = OllamaEvaluator(model_name="llama3:latest")
result = evaluator.evaluate_context_contains_answer(question, answer, context)
```

### evaluate_context_retrieval

Main evaluation function.

```python
df, som_cm, cosine_cm, som_metrics, cosine_metrics = evaluate_context_retrieval(
    questions, answers, som_contexts_scores, cosine_contexts_scores, max_workers=8
)
```

### create_comparison_chart

Generate visualization.

```python
create_comparison_chart(som_metrics, cosine_metrics, "output.png")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ollama team for the excellent local LLM framework
- Meta for the Llama 3 model
- The open-source community for the supporting libraries 