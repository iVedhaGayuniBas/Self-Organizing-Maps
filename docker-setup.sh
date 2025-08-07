#!/bin/bash
# Docker setup and management script for SOM Context Retrieval project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="som-context-retrieval"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_header "Checking Dependencies"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed ✓"
    
    # Check Ollama
    if command -v ollama &> /dev/null; then
        print_status "Ollama is installed ✓"
        
        # Check if Ollama is running
        if pgrep -f "ollama serve" > /dev/null; then
            print_status "Ollama is running ✓"
            
            # Test connectivity
            if python3 test_ollama.py > /dev/null 2>&1; then
                print_status "Ollama connectivity test passed ✓"
            else
                print_warning "Ollama connectivity test failed"
                print_warning "Run 'python3 test_ollama.py' to debug"
            fi
        else
            print_warning "Ollama is not running. Start with: ollama serve"
        fi
    else
        print_warning "Ollama is not installed. Install from: https://ollama.ai"
        print_warning "Some evaluation features will not work without Ollama"
    fi
}

# Create necessary directories
setup_directories() {
    print_header "Setting Up Directories"
    
    mkdir -p data outputs contexts logs
    
    # Copy questions file if it doesn't exist in data/
    if [ ! -f "data/questions_answers.xlsx" ] && [ -f "questions_answers.xlsx" ]; then
        cp questions_answers.xlsx data/
        print_status "Copied questions_answers.xlsx to data/ directory"
    fi
    
    # Copy CSV file if it exists
    if [ ! -f "data/wikipedia_context_comparison.csv" ] && [ -f "wikipedia_context_comparison.csv" ]; then
        cp wikipedia_context_comparison.csv data/
        print_status "Copied wikipedia_context_comparison.csv to data/ directory"
    fi
    
    print_status "Directory structure created ✓"
}

# Build Docker images
build_images() {
    print_header "Building Docker Images"
    
    print_status "Building SOM evaluation image..."
    docker-compose build som-evaluation
    
    print_status "Building Ray head node image..."
    docker-compose build ray-head
    
    print_status "All images built successfully ✓"
}

# Start Ray cluster
start_ray_cluster() {
    print_header "Starting Ray Cluster"
    
    print_status "Starting Ray head node..."
    docker-compose up -d ray-head
    
    # Wait for Ray head to be ready
    print_status "Waiting for Ray head node to be ready..."
    sleep 10
    
    print_status "Starting Ray worker nodes..."
    docker-compose up -d ray-worker
    
    print_status "Ray cluster started ✓"
    print_status "Ray Dashboard available at: http://localhost:8265"
}

# Stop Ray cluster
stop_ray_cluster() {
    print_header "Stopping Ray Cluster"
    
    docker-compose down
    print_status "Ray cluster stopped ✓"
}

# Run context generation
run_context_generation() {
    print_header "Running Context Generation"
    
    # Check if Cohere API key is set
    if [ -z "$COHERE_API_KEY" ]; then
        print_warning "COHERE_API_KEY environment variable not set"
        print_warning "Using default key from script (may have rate limits)"
    fi
    
    print_status "Starting context generation..."
    docker-compose run --rm som-evaluation python generate_contexts.py \
        --use_ray \
        --max_questions 1000
    
    print_status "Context generation completed ✓"
}

# Run evaluation
run_evaluation() {
    print_header "Running SOM vs Cosine Evaluation"
    
    local max_questions=${1:-1000}
    local evaluators=${2:-8}
    
    print_status "Starting evaluation..."
    print_status "Max questions: $max_questions"
    print_status "Number of evaluators: $evaluators"
    
    docker-compose run --rm som-evaluation python scale_up_evaluation.py \
        --qa_file /app/data/questions_answers.xlsx \
        --csv_file /app/data/wikipedia_context_comparison.csv \
        --max_questions $max_questions \
        --workers $evaluators \
        --use_ray \
        --ray_evaluators $evaluators \
        --batch_size 10
    
    print_status "Evaluation completed ✓"
    print_status "Results saved to outputs/ directory"
}

# Run Jupyter development environment
run_jupyter() {
    print_header "Starting Jupyter Development Environment"
    
    print_status "Starting Jupyter Lab..."
    docker-compose up -d jupyter
    
    print_status "Jupyter Lab started ✓"
    print_status "Access Jupyter at: http://localhost:8888"
    print_status "Use 'docker-compose logs jupyter' to see the access token"
}

# Run with Ollama
run_with_ollama() {
    print_header "Connecting to Ollama LLM"
    
    print_status "Testing connection to host Ollama service..."
    if python3 test_ollama.py; then
        print_status "Ollama connection verified ✓"
        print_status "Ready for evaluation with LLM support"
    else
        print_error "Cannot connect to Ollama"
        print_status "Make sure Ollama is running: ollama serve"
        print_status "And the model is available: ollama pull llama3:latest"
    fi
}

# View logs
view_logs() {
    local service=${1:-som-evaluation}
    print_header "Viewing Logs for $service"
    
    docker-compose logs -f $service
}

# Cleanup
cleanup() {
    print_header "Cleaning Up"
    
    print_status "Stopping all services..."
    docker-compose down
    
    print_status "Removing unused Docker resources..."
    docker system prune -f
    
    print_status "Cleanup completed ✓"
}

# Test Ollama connectivity
test_ollama() {
    print_header "Testing Ollama Connectivity"
    
    if [ ! -f "test_ollama.py" ]; then
        print_error "test_ollama.py not found!"
        return 1
    fi
    
    print_status "Running Ollama connectivity test..."
    python3 test_ollama.py
}

# Monitor Ray cluster
monitor_ray() {
    print_header "Ray Cluster Status"
    
    if docker-compose ps ray-head | grep -q "Up"; then
        print_status "Ray cluster is running ✓"
        print_status "Ray Dashboard: http://localhost:8265"
        
        # Show cluster resources
        print_status "Getting cluster resources..."
        docker-compose exec ray-head python -c "
import ray
ray.init(address='auto')
print('Cluster Resources:', ray.cluster_resources())
print('Available Resources:', ray.available_resources())
"
    else
        print_warning "Ray cluster is not running"
        print_status "Use './docker-setup.sh start' to start the cluster"
    fi
}

# Show usage
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                 - Setup directories and check dependencies"
    echo "  build                 - Build Docker images"
    echo "  start                 - Start Ray cluster"
    echo "  stop                  - Stop Ray cluster"
    echo "  generate [max_q]      - Run context generation (default: 1000 questions)"
    echo "  evaluate [max_q] [ev] - Run evaluation (default: 1000 questions, 8 evaluators)"
    echo "  test-ollama           - Test Ollama connectivity and functionality"
    echo "  jupyter               - Start Jupyter development environment"
    echo "  ollama                - Start with Ollama LLM support"
    echo "  monitor               - Monitor Ray cluster status"
    echo "  logs [service]        - View logs for a service"
    echo "  cleanup               - Stop services and cleanup Docker resources"
    echo "  all                   - Setup, build, and start everything"
    echo ""
    echo "Examples:"
    echo "  $0 setup              # Initial setup"
    echo "  $0 build              # Build images"
    echo "  $0 start              # Start Ray cluster"
    echo "  $0 generate 500       # Generate contexts for 500 questions"
    echo "  $0 evaluate 1000 12   # Evaluate 1000 questions with 12 evaluators"
    echo "  $0 jupyter            # Start Jupyter for development"
    echo "  $0 ollama             # Start with Ollama LLM"
    echo "  $0 all                # Complete setup and start"
}

# Main script logic
main() {
    case "$1" in
        setup)
            check_dependencies
            setup_directories
            ;;
        build)
            build_images
            ;;
        start)
            start_ray_cluster
            ;;
        stop)
            stop_ray_cluster
            ;;
        generate)
            run_context_generation
            ;;
        evaluate)
            run_evaluation $2 $3
            ;;
        test-ollama)
            test_ollama
            ;;
        jupyter)
            run_jupyter
            ;;
        ollama)
            run_with_ollama
            ;;
        monitor)
            monitor_ray
            ;;
        logs)
            view_logs $2
            ;;
        cleanup)
            cleanup
            ;;
        all)
            check_dependencies
            setup_directories
            build_images
            start_ray_cluster
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
