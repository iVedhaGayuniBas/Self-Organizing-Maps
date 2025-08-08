# Self-Organizing Maps vs Cosine Similarity Evaluation

This project evaluates the performance of Self-Organizing Maps (SOMpy) against traditional cosine similarity for context retrieval tasks using distributed computing with Ray.io and local LLM evaluation with Ollama.

## Features

- **SOM vs Cosine Comparison**: Compare context retrieval performance between Self-Organizing Maps and cosine similarity
- **Distributed Computing**: Optional Ray.io support for distributed evaluation across multiple workers
- **Local LLM Evaluation**: Use Ollama (llama3) for context relevance evaluation without external API calls
- **Docker Support**: Complete containerized setup with Docker Compose
- **Scalable**: Process thousands of questions with parallel evaluation

## Quick Start

### Prerequisites

1. **Docker & Docker Compose**: For containerized deployment
2. **Ollama** (optional): For LLM-based evaluation
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Pull the model
   ollama pull llama3:latest
   ```

### Basic Usage

1. **Setup and check dependencies**:
   ```bash
   ./docker-setup.sh setup
   ```

2. **Build Docker images**:
   ```bash
   ./docker-setup.sh build
   ```

3. **Start Ray cluster**:
   ```bash
   ./docker-setup.sh start
   ```

4. **Run evaluation**:
   ```bash
   ./docker-setup.sh evaluate 1000 8  # 1000 questions, 8 evaluators
   ```

### Local Development

If you prefer to run without Docker:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Ollama connection**:
   ```bash
   python test_ollama.py
   ```

3. **Run evaluation (standard mode)**:
   ```bash
   python scale_up_evaluation.py --max_questions 100 --workers 4
   ```

4. **Run evaluation (Ray distributed mode)**:
   ```bash
   # Start Ray cluster first
   ray start --head
   
   # Run with Ray
   python scale_up_evaluation.py --max_questions 1000 --use_ray --ray_evaluators 8
   ```

## Configuration

### Environment Variables

- `COHERE_API_KEY`: API key for Cohere embeddings (optional, has default)
- `OLLAMA_BASE_URL`: Ollama service URL (default: http://localhost:11434)

### Key Parameters

- `--max_questions`: Number of questions to evaluate
- `--workers`: Number of parallel workers (standard mode)
- `--use_ray`: Enable Ray distributed processing
- `--ray_evaluators`: Number of Ray evaluator actors
- `--batch_size`: Batch size for Ray evaluation

## File Structure

```
├── scale_up_evaluation.py     # Main evaluation script (supports Ray)
├── generate_contexts.py       # Context generation script
├── docker-setup.sh           # Docker management script
├── docker-compose.yml        # Docker services configuration
├── Dockerfile               # Multi-stage Docker build
├── requirements.txt         # Python dependencies
├── test_ollama.py          # Ollama connectivity test
└── data/                   # Input data directory
    ├── questions_answers.xlsx
    └── wikipedia_context_comparison.csv
```

## Docker Commands

| Command | Description |
|---------|-------------|
| `./docker-setup.sh setup` | Check dependencies and setup directories |
| `./docker-setup.sh build` | Build Docker images |
| `./docker-setup.sh start` | Start Ray cluster |
| `./docker-setup.sh evaluate [questions] [evaluators]` | Run evaluation |
| `./docker-setup.sh generate [questions]` | Generate contexts |
| `./docker-setup.sh jupyter` | Start Jupyter development environment |
| `./docker-setup.sh test-ollama` | Test Ollama connectivity |
| `./docker-setup.sh monitor` | Monitor Ray cluster status |
| `./docker-setup.sh cleanup` | Stop services and cleanup |

## Ray.io Integration

The project supports both standard and distributed evaluation modes:

### Standard Mode
- Uses ThreadPoolExecutor for parallel processing
- Good for smaller datasets (< 1000 questions)
- No additional setup required

### Ray Distributed Mode
- Uses Ray actors for distributed processing
- Scales to multiple machines
- Handles larger datasets efficiently (1000+ questions)
- Automatic load balancing across workers

### Example Ray Usage

```python
# Enable Ray mode
python scale_up_evaluation.py \
    --use_ray \
    --max_questions 5000 \
    --ray_evaluators 12 \
    --batch_size 20
```

## Performance Comparison

The evaluation generates:
- **Precision, Recall, F1 scores** for both SOM and Cosine methods
- **Confusion matrices** showing classification performance
- **Visualization charts** comparing the methods
- **Detailed CSV results** for further analysis

## Results Interpretation

- **Higher F1 scores** indicate better context retrieval performance
- **Precision** measures how many retrieved contexts are relevant
- **Recall** measures how many relevant contexts were retrieved
- **SOM typically shows improvement** in handling semantic similarity vs pure cosine distance

## Troubleshooting

### Common Issues

1. **Ollama connection failed**:
   ```bash
   # Check if Ollama is running
   python test_ollama.py
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Ray cluster not starting**:
   ```bash
   # Check Ray dashboard
   ./docker-setup.sh monitor
   
   # Restart cluster
   ./docker-setup.sh stop
   ./docker-setup.sh start
   ```

3. **Docker build issues**:
   ```bash
   # Clean rebuild
   ./docker-setup.sh cleanup
   ./docker-setup.sh build
   ```

## Development

### Adding New Features

1. **Extend evaluation metrics**: Modify `_process_results()` in `scale_up_evaluation.py`
2. **Add new context retrieval methods**: Extend the evaluation comparison
3. **Custom LLM models**: Modify `OllamaEvaluator` class

### Testing

```bash
# Test Ollama connectivity
python test_ollama.py

# Test Docker setup
./docker-setup.sh setup

# Run small evaluation test
python scale_up_evaluation.py --max_questions 10
```

## Contributing

1. Keep the consolidated structure - avoid creating multiple files for similar functionality
2. Update this README when adding new features
3. Test both standard and Ray modes when making changes
4. Ensure Docker compatibility for new dependencies
