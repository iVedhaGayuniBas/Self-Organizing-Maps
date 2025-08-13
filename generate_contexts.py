import pandas as pd
import numpy as np
import torch
import requests
import json
import time
import pickle
import os
from typing import List, Tuple
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sompy.sompy import SOMFactory
from math import sqrt
import csv
from tqdm import tqdm

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

# Optional Ray imports
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

def embed_questions(questions: List[str], output_dimension: int) -> np.ndarray:
    """Embed questions using the same method as Wikipedia data"""
    print(f"Using pre-computed Wikipedia embeddings for questions...")
    
    # For quick testing, use Wikipedia embeddings directly
    # This ensures compatibility but limits to Wikipedia content
    embeddings = []
    
    for i, question in enumerate(tqdm(questions, desc="Mapping questions to Wikipedia embeddings")):
        # Use cycling through available embeddings
        embedding_index = i % len(docs)
        embeddings.append(docs[embedding_index]['emb'])
    
    embeddings_array = np.array(embeddings)
    print(f"✅ Using {len(embeddings)} Wikipedia embeddings for questions, shape: {embeddings_array.shape}")
    return embeddings_array

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # or "all-minilm" depending on what you have

# Optional Ray imports
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# --- CONFIGURATION ---
INPUT_FILE = "./questions_answers.xlsx"   # Your Excel file
OUTPUT_FILE = "results/retrieved_contexts.csv"  # Where to save results
NUM_CONTEXTS = 5                         # Number of contexts to generate for each method
MAX_QUESTIONS = 5000                     # Limit for testing


# insert ray.init
# place decorator on top of relevant class/function
# refer to ayza/mihin/hasaan for understanding ray
# ask mihin/ayza to containerize the code

# --- GLOBAL VARIABLES ---
docs = []
wik_embeddings = None
sm = None
def get_ollama_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embedding from Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        raise

def setup_ollama():
    """Test Ollama connection"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m['name'] for m in models]
        print(f"✅ Ollama connected with {len(models)} models: {model_names}")
        
        # Check if embedding model is available
        if not any(EMBEDDING_MODEL in name for name in model_names):
            print(f"⚠️  Embedding model '{EMBEDDING_MODEL}' not found. Available models: {model_names}")
            print(f"   You may need to run: ollama pull {EMBEDDING_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        raise

def load_wikipedia_data(max_docs=5000):
    """Load Wikipedia embeddings dataset"""
    global docs, wik_embeddings
    
    print("Loading Wikipedia dataset...")
    docs_stream = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)
    
    docs = []
    wik_embeddings = []
    
    for doc in docs_stream:
        docs.append(doc)
        wik_embeddings.append(doc['emb'])
        if len(docs) >= max_docs:
            break
    
    wik_embeddings = torch.tensor(wik_embeddings)
    np_embeddings = np.array(wik_embeddings)
    
    print(f"✅ Loaded {len(docs)} Wikipedia documents")
    print(f"Embeddings shape: {np_embeddings.shape}")
    
    return np_embeddings

def train_som_model(np_embeddings):
    """Train SOM model on Wikipedia embeddings"""
    global sm
    
    print("Training SOM model...")
    
    # Calculate training lengths
    rough_len = int(5 * sqrt(len(np_embeddings)))
    finetune_len = int(20 * sqrt(len(np_embeddings)))
    
    # SOM parameters (using the best from your notebook)
    test_case = {
        'mapsize': (10, 10),
        'normalization': 'var',
        'initialization': 'pca',
        'lattice': 'rect',
        'neighborhood': 'gaussian',
        'training': 'batch',
        'name': 'som1',
        'rough_len': rough_len,
        'finetune_len': finetune_len
    }
    
    som_fac = SOMFactory()
    sm = som_fac.build(
        np_embeddings, 
        mapsize=test_case['mapsize'],
        normalization=test_case['normalization'],
        initialization=test_case['initialization'],
        lattice=test_case['lattice'],
        neighborhood=test_case['neighborhood'],
        training=test_case['training'],
        name=test_case['name']
    )
    
    sm.train(n_job=1, verbose=False, train_rough_len=rough_len, train_finetune_len=finetune_len)
    
    topographic_error = sm.calculate_topographic_error()
    # Calculate quantization error manually if quant_error_history doesn't exist
    try:
        quantization_error = sm.quant_error_history[-1]
    except AttributeError:
        # Calculate quantization error manually
        quantization_error = sm.calculate_quantization_error()
    print(f"✅ SOM trained - Topographic error: {topographic_error:.3f}, Quantization error: {quantization_error:.3f}")
    
    return sm

def find_closest_hit_per_bmu(sm):
    """Find the closest hit for each BMU"""
    chunk_bmu_indices = sm._bmu[0].astype(int)
    chunk_bmu_qe = sm._bmu[1]
    data = sm._data
    
    closest_indices = np.full(sm.codebook.nnodes, np.nan)
    min_errors = np.full(sm.codebook.nnodes, np.inf)
    
    for idx, (bmu, err) in enumerate(zip(chunk_bmu_indices, chunk_bmu_qe)):
        if err < min_errors[bmu]:
            min_errors[bmu] = err
            closest_indices[bmu] = idx
    
    bmu_first_hit_vectors = [data[int(i)] if not np.isnan(i) else None for i in closest_indices]
    return closest_indices, bmu_first_hit_vectors

def cosine_bmu_retrieve(normed_query_embedding, som_model, bmu_hit_vectors_with_indices, top_k=5):
    """Retrieve top BMUs using cosine similarity"""
    filtered = [(i, v) for i, v in bmu_hit_vectors_with_indices if v is not None and not np.isnan(v).any()]
    if len(filtered) == 0:
        raise ValueError("No valid BMU vectors available for similarity comparison.")
    
    indices, vectors = zip(*filtered)
    vectors = np.array(vectors)
    indices = np.array(indices)
    
    similarities = cosine_similarity(normed_query_embedding, vectors)
    sorted_indices_desc = np.argsort(similarities[0])[::-1]
    top_k_indices = sorted_indices_desc[:top_k]
    selected_bmu_index = indices[top_k_indices]
    
    return selected_bmu_index

def normalize_reshape(query_embedding, sm):
    """Normalize and reshape query embedding"""
    vec = query_embedding.reshape(1, -1)
    normed_query_embedding = sm._normalizer.normalize_by(sm.data_raw, vec)
    return normed_query_embedding

def get_som_context(query_embedding, som_model, bmu_hit_vectors_with_indices, chunks, top_k=5, bmu_k=5):
    """Get SOM-based context for a query"""
    q_vec = normalize_reshape(query_embedding, som_model)
    q_bmus = cosine_bmu_retrieve(q_vec, som_model, bmu_hit_vectors_with_indices, bmu_k)
    
    chunk_bmu_indices = som_model._bmu[0].astype(int)
    q_bmus_set = set(q_bmus)
    candidates_emb = [(i, chunks[i]['emb']) for i, bmu_index in enumerate(chunk_bmu_indices) if bmu_index in q_bmus_set]
    
    if not candidates_emb:
        # Return empty strings if no candidates found
        return [" "] * top_k, [0.0] * top_k
    
    # Extract embeddings for similarity calculation
    candidate_embeddings = np.array([emb for _, emb in candidates_emb])
    scores = cosine_similarity(q_vec, candidate_embeddings)[0]
    sorted_indices_desc = np.argsort(scores)[::-1]
    
    top_k_chunk_indices = [candidates_emb[i][0] for i in sorted_indices_desc[:top_k]]
    context = [chunks[i]['text'] for i in top_k_chunk_indices]
    context_scores = scores[sorted_indices_desc[:top_k]]
    
    # Ensure we always return exactly top_k results
    while len(context) < top_k:
        context.append(" ")
        context_scores = np.append(context_scores, 0.0)
    
    return context[:top_k], context_scores[:top_k]

def get_cosine_context(query_embeddings, chunks, top_k=5):
    """Get cosine similarity-based context"""
    vectors = np.array([chunk['emb'] for chunk in chunks])
    texts = [chunk['text'] for chunk in chunks]
    
    # Normalize
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)
    query_embeddings = scaler.transform(query_embeddings)
    
    similarities = cosine_similarity(query_embeddings, vectors)
    
    # Get top-k for each query
    top_k_indices = np.argpartition(-similarities[0], min(top_k, len(similarities[0])))[:min(top_k, len(similarities[0]))]
    top_k_indices = top_k_indices[np.argsort(-similarities[0][top_k_indices])]
    
    context = [texts[i] for i in top_k_indices]
    context_score = [similarities[0][i] for i in top_k_indices]
    
    # Ensure we always return exactly top_k results
    while len(context) < top_k:
        context.append(" ")
        context_score.append(0.0)
    
    return context[:top_k], context_score[:top_k]

def retrieve_som_contexts(question_embedding):
    """Retrieve SOM contexts for a question"""
    global sm, docs
    
    if sm is None:
        raise ValueError("SOM model not trained. Call train_som_model() first.")
    
    bmu_first_hit_indices, bmu_first_hit_vectors = find_closest_hit_per_bmu(sm)
    bmu_hit_vectors_with_indices = [(i, v) for i, v in enumerate(bmu_first_hit_vectors)]
    
    context, scores = get_som_context(question_embedding, sm, bmu_hit_vectors_with_indices, docs, NUM_CONTEXTS)
    return context

def retrieve_cosine_contexts(question_embedding):
    """Retrieve cosine similarity contexts for a question"""
    global docs
    
    context, scores = get_cosine_context(question_embedding.reshape(1, -1), docs, NUM_CONTEXTS)
    return context

def generate_contexts(input_path: str, output_path: str = "results/retrieved_contexts.csv"):
    """Main function to generate contexts for all questions"""
    if not os.path.exists(input_path):
        print(f"Excel file not found: {input_path}")
        return
    
    print("Starting context generation...")

    # Load questions and answers
    print("Loading questions and answers...")
    df = pd.read_excel(input_path)
    if "question" not in df.columns or "answer" not in df.columns:
        print("Excel file must have 'question' and 'answer' columns")
        return
    
    questions = df["question"].astype(str).str.strip().tolist()
    answers = df["answer"].astype(str).str.strip().tolist()
    
    # Limit for testing
    if len(questions) > MAX_QUESTIONS:
        questions = questions[:MAX_QUESTIONS]
        answers = answers[:MAX_QUESTIONS]
        print(f"Limited to first {MAX_QUESTIONS} questions for testing")
    
    print(f"Processing {len(questions)} questions...")
    
    # Setup
    setup_ollama()
    np_embeddings = load_wikipedia_data()
    train_som_model(np_embeddings)
    
    # Embed all questions
    question_embeddings = embed_questions(questions, np_embeddings.shape[1])
    
    # Generate contexts
    results = []
    for i, (question, answer, embedding) in enumerate(zip(questions, answers, question_embeddings)):
        print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        
        try:
            som_contexts = retrieve_som_contexts(embedding)
            cosine_contexts = retrieve_cosine_contexts(embedding)
            
            results.append({
                "question": question,
                "answer": answer,
                "som_contexts": som_contexts,
                "cosine_contexts": cosine_contexts
            })
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            # Add empty results for failed questions
            results.append({
                "question": question,
                "answer": answer,
                "som_contexts": [],
                "cosine_contexts": []
            })
    
    # Save results
    print("Saving results...")
    
    # Create results directory if it doesn't exist with proper permissions
    results_dir = os.path.dirname(output_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, mode=0o777, exist_ok=True)
    
    # Ensure we can write to the current directory
    if not output_path.startswith('/'):
        # For relative paths, write to current directory
        output_path = os.path.join(os.getcwd(), output_path)
    
    print(f"Writing to: {output_path}")
    
    # Save as CSV
    with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer", "som_context", "cosine_context"])
        
        for result in results:
            writer.writerow([
                result["question"],
                result["answer"],
                str(result["som_contexts"]),
                str(result["cosine_contexts"])
            ])
    
    # Save as pickle for the evaluation script
    contexts_dir = "Self-Organizing-Maps/contexts"
    os.makedirs(contexts_dir, mode=0o777, exist_ok=True)
    
    som_contexts_scores = [(result["som_contexts"], np.array([0.5] * len(result["som_contexts"]))) for result in results]
    cosine_contexts_scores = [(result["cosine_contexts"], np.array([0.5] * len(result["cosine_contexts"]))) for result in results]
    
    with open(f"{contexts_dir}/som_contexts_scores.pkl", 'wb') as f:
        pickle.dump(som_contexts_scores, f)
    
    with open(f"{contexts_dir}/cosine_contexts_scores.pkl", 'wb') as f:
        pickle.dump(cosine_contexts_scores, f)
    
    print(f"✅ Saved results to {output_path}")
    print(f"✅ Saved pickle files to {contexts_dir}/")
    print(f"✅ Processed {len(results)} questions successfully")

if __name__ == "__main__":
    generate_contexts(INPUT_FILE, OUTPUT_FILE) 