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
from sompy.visualization.bmuhits import BmuHitsView
from math import sqrt
import csv
from tqdm import tqdm
import logging

# # Ollama configuration
# OLLAMA_BASE_URL = "http://localhost:11434"
# EMBEDDING_MODEL = "nomic-embed-text"

# Optional Ray imports
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# ensure INFO and above are printed to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def embed_questions(questions: List[str]) -> np.ndarray:
    """Load precomputed question embeddings from `stored_embeddings/question_embeddings_cohere.npy`.

    Mapping strategy:
    - If `stored_embeddings/questions_cohere.pkl` exists, map each input question to the stored question
      by normalized exact match (strip + lower); if found, use that stored embedding.
    - If a question is not found in the stored list, fall back to positional mapping: use the
      embedding at the same index `i` (requires that the saved array is at least as long as
      the number of questions requested).
    - If no `questions_cohere.pkl` exists, positional mapping is used (index i -> embedding i).
    """
    print("Loading question embeddings from 'stored_embeddings/question_embeddings_cohere.npy'...")

    emb_path = os.path.join("stored_embeddings", "question_embeddings_cohere.npy")
    qlist_path = os.path.join("stored_embeddings", "questions_cohere.pkl")

    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"Question embeddings file not found: {emb_path}")

    all_q_embs = np.load(emb_path)

    stored_qs = None
    if os.path.isfile(qlist_path):
        with open(qlist_path, "rb") as f:
            try:
                stored_qs = pickle.load(f)
            except Exception:
                stored_qs = None

    # Build mapping from normalized stored question -> index
    mapping = {}
    if stored_qs is not None:
        for idx, q in enumerate(stored_qs):
            try:
                key = str(q).strip().lower()
            except Exception:
                key = str(q)
            if key not in mapping:
                mapping[key] = idx

    embeddings = []
    for i, question in enumerate(tqdm(questions, desc="Loading question embeddings")):
        key = str(question).strip().lower()
        idx = None

        if stored_qs is not None and key in mapping:
            idx = mapping[key]
        else:
            # fallback to positional mapping if possible
            if i < len(all_q_embs):
                idx = i
            else:
                raise IndexError(f"No stored embedding found for question index {i} and question not present in stored questions.")

        emb = np.array(all_q_embs[idx])
        embeddings.append(emb)

    embeddings_array = np.vstack(embeddings)
    print(f"✅ Loaded {embeddings_array.shape[0]} question embeddings, shape: {embeddings_array.shape}")
    return embeddings_array

# # Ollama configuration
# OLLAMA_BASE_URL = "http://localhost:11434"
# EMBEDDING_MODEL = "nomic-embed-text"  # or "all-minilm" depending on what you have

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
MAX_CHUNKS = 5000                        # Limit for Wikipedia chunks


# insert ray.init
# place decorator on top of relevant class/function
# refer to ayza/mihin/hasaan for understanding ray
# ask mihin/ayza to containerize the code

# --- GLOBAL VARIABLES ---
docs = []
wik_embeddings = None
sm = None
# def get_ollama_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
#     """Get embedding from Ollama"""
#     try:
#         response = requests.post(
#             f"{OLLAMA_BASE_URL}/api/embeddings",
#             json={
#                 "model": model,
#                 "prompt": text
#             },
#             timeout=30
#         )
#         response.raise_for_status()
#         return response.json()["embedding"]
#     except Exception as e:
#         print(f"Error getting embedding from Ollama: {e}")
#         raise

# def setup_ollama():
#     """Test Ollama connection"""
#     try:
#         response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
#         response.raise_for_status()
#         models = response.json().get("models", [])
#         model_names = [m['name'] for m in models]
#         print(f"✅ Ollama connected with {len(models)} models: {model_names}")
        
#         # Check if embedding model is available
#         if not any(EMBEDDING_MODEL in name for name in model_names):
#             print(f"⚠️  Embedding model '{EMBEDDING_MODEL}' not found. Available models: {model_names}")
#             print(f"   You may need to run: ollama pull {EMBEDDING_MODEL}")
#         return True
#     except Exception as e:
#         print(f"❌ Cannot connect to Ollama: {e}")
#         print("   Make sure Ollama is running: ollama serve")
#         raise

def load_wikipedia_data(max_chunks=5000):
    """Load Wikipedia embeddings dataset"""
    global docs, wik_embeddings
    print("Loading Wikipedia dataset from 'stored_embeddings'...")

    emb_path = os.path.join("stored_embeddings", "wiki_embeddings_cohere.npy")
    docs_path = os.path.join("stored_embeddings", "wiki_docs_cohere.pkl")

    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not os.path.isfile(docs_path):
        raise FileNotFoundError(f"Docs pickle not found: {docs_path}")

    # Load numpy embeddings and truncate to max_chunks
    np_embeddings = np.load(emb_path)
    if max_chunks is not None and len(np_embeddings) > max_chunks:
        np_embeddings = np_embeddings[:max_chunks]

    # Load docs pickle and truncate to max_chunks
    with open(docs_path, "rb") as f:
        loaded_docs = pickle.load(f)
    if max_chunks is not None and len(loaded_docs) > max_chunks:
        loaded_docs = loaded_docs[:max_chunks]

    # Ensure global `docs` list contains dicts with at least 'text' and 'emb'
    docs = []
    for i, item in enumerate(loaded_docs):
        emb = np.array(np_embeddings[i])
        if isinstance(item, dict):
            d = item.copy()
            d['emb'] = emb
        else:
            d = {'text': str(item), 'emb': emb}
        docs.append(d)

    wik_embeddings = torch.tensor(np_embeddings)

    print(f"✅ Loaded {len(docs)} Wikipedia documents from 'stored_embeddings'")
    print(f"Embeddings shape: {np_embeddings.shape}")

    return np_embeddings

# Function to visualize the trained SOM map and save in results directory
def visualize_som_map(som_model, save_path="results/som_bmu_hits.png"):
    """Visualize and save SOM BMU hits"""
    bmu_view = BmuHitsView(10, 10, "Hits Map", text_size=7)
    bmu_view.show(som_model, anotate=True, onlyzeros=False, labelsize=12, cmap="plasma", logaritmic=False)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.dirname(save_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, mode=0o777, exist_ok=True)
    
    bmu_view.save(save_path)
    print(f"✅ Saved SOM BMU hits visualization to {save_path}")

def train_som_model(np_embeddings, mapsize=(10,10), rough_len=1000, finetune_len=4000, lattice="rect"):
    """Train SOM model on Wikipedia embeddings"""
    global sm
    
    print(f"Training SOM model with mapsize={mapsize}, rough_len={rough_len}, finetune_len={finetune_len}...")
    
    # SOM parameters (using the best from your notebook)
    test_case = {
        'mapsize': mapsize,
        'normalization': 'var',
        'initialization': 'pca',
        'lattice': lattice, #['rect', 'hexa']
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
    
    sm.train(n_job=1, verbose='info', train_rough_len=rough_len, train_finetune_len=finetune_len)

    # Visualize and save the SOM map
    visualize_som_map(sm, save_path="results/som_bmu_hits.png")
    
    topographic_error = sm.calculate_topographic_error()
    # Calculate quantization error manually if quant_error_history doesn't exist
    try:
        quantization_error = sm.quant_error_history[-1]
    except AttributeError:
        # Calculate quantization error manually
        quantization_error = sm.calculate_quantization_error()
    print(f"✅ SOM trained - Topographic error: {topographic_error:.3f}, Quantization error: {quantization_error:.3f}")

    return rough_len, finetune_len, topographic_error, quantization_error

def log_som_training( number_of_chunks: int,
                    number_of_questions: int,
                    map_size_tuple: Tuple[int, int],
                    lattice: str,
                    rough_len: int,
                    finetune_len: int,
                    top_k: int,
                    topographic_error: float,
                    quantization_error: float,
                    log_path: str = "results/som_training_logs.csv"):
    """Append SOM training info to CSV (creates results dir if missing)."""
    # Ensure results dir exists
    results_dir = os.path.dirname(log_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, mode=0o777, exist_ok=True)

    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline='') as csvfile:
        fieldnames = ['timestamp', 'number_of_chunks', 'number_of_questions', 'map_size', 'lattice', 'top_k', 'rough_len', 'finetune_len', 'topographic_error', 'quantization_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'number_of_chunks': number_of_chunks,
            'number_of_questions': number_of_questions,
            'map_size': f"{map_size_tuple[0]},{map_size_tuple[1]}",
            'lattice':f"{lattice}",
            'top_k': top_k,
            'rough_len': rough_len,
            'finetune_len': finetune_len,
            'topographic_error': f"{topographic_error:.6f}",
            'quantization_error': f"{quantization_error:.6f}"
        })

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

def find_query_k_bmus_cosine(normed_query_embedding, som_model, bmu_first_hit_vectors_with_indices, bmu_k):
    """Retrieve top BMUs using cosine similarity"""
    filtered = [(i, v) for i, v in bmu_first_hit_vectors_with_indices if v is not None and not np.isnan(v).any()]
    if len(filtered) == 0:
        raise ValueError("No valid BMU vectors available for similarity comparison.")
    
    indices, vectors = zip(*filtered)
    vectors = np.array(vectors)
    indices = np.array(indices)
    
    similarities = cosine_similarity(normed_query_embedding, vectors)
    sorted_indices_desc = np.argsort(similarities[0])[::-1]
    bmu_k_indices = sorted_indices_desc[:bmu_k]
    selected_bmu_index = indices[bmu_k_indices]
    print(f"Top {bmu_k} BMU indices: {selected_bmu_index}")
    
    return selected_bmu_index

def normalize_reshape_query(query_embedding, sm): #####Find a different way 
    """Normalize and reshape query embedding"""
    vec = query_embedding.reshape(1, -1)
    normed_query_embedding = sm._normalizer.normalize_by(sm.data_raw, vec)
    return normed_query_embedding

def get_query_context_som_with_scores(query_embedding, som_model, bmu_hit_vectors_with_indices, chunks, top_k=1):
    """Get SOM-based context for a query"""
    print(f"Calculating som similarity for context retrieval of k = {top_k}...")
    q_vec = normalize_reshape_query(query_embedding, som_model)
    q_bmus = find_query_k_bmus_cosine(q_vec, som_model, bmu_hit_vectors_with_indices, top_k)
    
    chunk_bmu_indices = som_model._bmu[0].astype(int)
    q_bmus_set = set(q_bmus)
    print("BMU indices for candidate chunks:", q_bmus_set)

    context = []
    context_scores = []

    for bmu in q_bmus_set:
        candidates_emb = [(i, chunks[i]['emb']) for i, bmu_index in enumerate(chunk_bmu_indices) if bmu_index == bmu]
    
        if not candidates_emb:
            # Return empty strings if no candidates found
            return [" "] * 1, [0.0] * 1
    
        # Extract embeddings for similarity calculation
        candidate_embeddings = np.array([emb for _, emb in candidates_emb])
        scores = cosine_similarity(q_vec, candidate_embeddings)[0]
        sorted_indices_desc = np.argsort(scores)[::-1]
    
        top_ONE_chunk_index = [candidates_emb[i][0] for i in sorted_indices_desc[:1]] #1 indices of that specific bmu
        bmu_context = [chunks[i]['text'] for i in top_ONE_chunk_index]
        bmu_context_scores = scores[sorted_indices_desc[:1]]
        context.extend(bmu_context)
        context_scores.extend(bmu_context_scores.tolist())

        #calculate quantization error between query and the selected chunk (only 1 chunk)
        selected_chunk_vector = chunks[top_ONE_chunk_index[0]]['emb']
        quantization_error = np.linalg.norm(query_embedding - selected_chunk_vector)
        print(f"Quantization error for BMU {bmu}: {quantization_error:.4f}")
    
    # Ensure we always return exactly top_k results
    while len(context) < top_k:
        context.append(" ")
        context_scores = np.append(context_scores, 0.0)
    
    return context[:top_k], context_scores[:top_k], top_ONE_chunk_index, quantization_error

def get_query_context_cosine_with_scores(query_embeddings, chunks, top_k=1): ##change chunks into wiki emb
    """Get cosine similarity-based context"""
    print(f"Calculating cosine similarity for context retrieval of k = {top_k}...")

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

def retrieve_final_contexts_som(question_embedding, top_k=NUM_CONTEXTS):
    """Retrieve SOM contexts for a question"""
    global sm, docs
    
    if sm is None:
        raise ValueError("SOM model not trained. Call train_som_model() first.")
    
    bmu_first_hit_indices, bmu_first_hit_vectors = find_closest_hit_per_bmu(sm)
    bmu_hit_vectors_with_indices = [(i, v) for i, v in enumerate(bmu_first_hit_vectors)]

    context, scores, top_ONE_chunk_index, quantization_error = get_query_context_som_with_scores(question_embedding, sm, bmu_hit_vectors_with_indices, docs, top_k)
    return context, scores, top_ONE_chunk_index, quantization_error

def retrieve_final_contexts_cosine(question_embedding, top_k=NUM_CONTEXTS):
    """Retrieve cosine similarity contexts for a question"""
    global docs
    
    context, scores = get_query_context_cosine_with_scores(question_embedding.reshape(1, -1), docs, top_k)
    return context, scores

def find_contexts_all_questions(
        input_path: str, 
        output_path: str = "contexts/retrieved_contexts", 
        max_chunks: int = MAX_CHUNKS, 
        max_questions: int = MAX_QUESTIONS,
        map_size: tuple = (10,10),
        rough_len_method: str = "embedding_formula",
        finetune_len_method: str = "embedding_formula",
        lattice: str = "rect",
        top_k: int = NUM_CONTEXTS
    ):
    """Main function to generate contexts for all questions"""

    output_path_pkl = output_path + ".pkl"
    output_path_csv = output_path + ".csv"
    
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
    if len(questions) > max_questions:
        questions = questions[:max_questions]
        answers = answers[:max_questions]
        print(f"Limited to first {max_questions} questions for testing")

    print(f"Processing {len(questions)} questions...")
    
    # Setup
    # setup_ollama()
    np_embeddings = load_wikipedia_data(max_chunks=max_chunks)

    # Calculate rough_len and finetune_len based on selected methods
    if rough_len_method == 'embedding_formula':
        rough_len = int(5 * sqrt(len(np_embeddings)))
    elif rough_len_method == 'neurons_formula':
        rough_len = int(500 * map_size[0] * map_size[1])  
    else:
        raise ValueError(f"Unknown rough_len_method: {rough_len_method}")
    
    if finetune_len_method == 'embedding_formula':
        finetune_len = int(20 * sqrt(len(np_embeddings)))
    elif finetune_len_method == 'error_convergence':
        finetune_len = int(10 * sqrt(len(np_embeddings)))  # Example formula, adjust as needed
    else:
        raise ValueError(f"Unknown finetune_len_method: {finetune_len_method}")

    print(f"Training SOM with map size {map_size}, rough_len={rough_len}, finetune_len={finetune_len}...")
    rough_len, finetune_len, topographic_error, quantization_error = train_som_model(np_embeddings, mapsize=map_size, rough_len=rough_len, finetune_len=finetune_len, lattice=lattice)

    log_som_training(
        number_of_chunks=max_chunks,
        number_of_questions=max_questions,
        map_size_tuple=map_size,
        lattice=lattice,
        rough_len=rough_len,
        finetune_len=finetune_len,
        top_k=top_k,
        topographic_error=topographic_error,
        quantization_error=quantization_error
    )
    
    # Embed all questions
    question_embeddings = embed_questions(questions)
    
    # Generate contexts
    results = []
    dataset_2 = []
    for i, (question, answer, embedding) in enumerate(zip(questions, answers, question_embeddings)):
        print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        
        try:
            som_contexts, som_scores, top_ONE_chunk_index, som_quantization_error = retrieve_final_contexts_som(embedding, top_k)
            cosine_contexts, cosine_scores = retrieve_final_contexts_cosine(embedding, top_k)
            
            # results.append({
            #     "question": question,
            #     "answer": answer,
            #     "som_contexts": som_contexts,
            #     "som_scores": som_scores,
            #     "cosine_contexts": cosine_contexts,
            #     "cosine_scores": cosine_scores
            # })

            dataset_2.append({
                "question": question,
                "answer": answer,
                "som_contexts": som_contexts,
                "som_scores": som_scores,
                "som_quantization_error": som_quantization_error,
                "top_ONE_chunk_index": top_ONE_chunk_index,
                "cosine_contexts": cosine_contexts,
                "cosine_scores": cosine_scores
            })
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            # Add empty results for failed questions
            # results.append({
            #     "question": question,
            #     "answer": answer,
            #     "som_contexts": [],
            #     "som_scores": [],
            #     "cosine_contexts": [],
            #     "cosine_scores": []
            # })

            dataset_2.append({
                "question": question,
                "answer": answer,
                "som_contexts": [],
                "som_scores": [],
                "som_quantization_error": None,
                "top_ONE_chunk_index": [],
                "cosine_contexts": [],
                "cosine_scores": []
            })

        # -------------------------------
    # Select top 30% closest questions
    # -------------------------------

    # Attach original question index to each row
    dataset_2_with_idx = [
        {**row, "question_index": idx}
        for idx, row in enumerate(dataset_2)
        if row["som_quantization_error"] is not None
    ]

    # Sort by quantization error (ascending = closer = better)
    dataset_2_sorted = sorted(
        dataset_2_with_idx,
        key=lambda x: x["som_quantization_error"]
    )

    # Compute top 30% cutoff
    top_30_percent = int(0.3 * len(dataset_2_sorted))

    # Select top 30%
    results = dataset_2_sorted[:top_30_percent]

    print(f"Selected {len(results)} questions out of {len(dataset_2)} (top 30% closest to BMUs)") 
        
    # Save results
    print("Saving results...")
    
    # Create results directory if it doesn't exist with proper permissions
    results_dir = 'contexts'
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, mode=0o777, exist_ok=True)
    
    # Ensure we can write to the current directory
    if not output_path_csv.startswith('/'):
        # For relative paths, write to current directory
        output_path_csv = os.path.join(os.getcwd(), output_path_csv)
    
    print(f"Writing to: {output_path_csv}")
    
    # Save as CSV
    with open(output_path_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer", "som_context", "som_scores", "cosine_context", "cosine_scores"])
        
        for result in results:
            writer.writerow([
                result["question"],
                result["answer"],
                str(result["som_contexts"]),
                str(result["som_scores"]),
                str(result["cosine_contexts"]),
                str(result["cosine_scores"])
            ])

    with open(f"{output_path}_dataset2.csv", "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer", "som_contexts", "som_scores", "som_quantization_error", "top_ONE_chunk_index", "cosine_contexts", "cosine_scores"])
        
        for data in dataset_2:
            writer.writerow([
                data["question"],
                data["answer"],
                str(data["som_contexts"]),
                str(data["som_scores"]),
                str(data["som_quantization_error"]),
                str(data["top_ONE_chunk_index"]),
                str(data["cosine_contexts"]),
                str(data["cosine_scores"])
            ])
    
    # Save as pickle for the evaluation script
    
    # Save questions, answers and retrieved contexts into a single pickle
    retrieved = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "som_contexts": r["som_contexts"],
            "som_scores": r["som_scores"],
            "cosine_contexts": r["cosine_contexts"],
            "cosine_scores": r["cosine_scores"]
        }
        for r in results
    ]
    with open(f"{output_path_pkl}", "wb") as f:
        pickle.dump(retrieved, f)
    
    # print(f"✅ Saved results to {output_path}")
    print(f"✅ Saved pickle files to {output_path_pkl}")
    print(f"✅ Processed {len(results)} questions successfully")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate contexts from Wikipedia for Q&A")
    parser.add_argument("--input", "-i", type=str, default=INPUT_FILE,
                        help="Path to questions Excel file (default: %(default)s)")
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_FILE,
                        help="Path for output CSV (default: %(default)s)")
    parser.add_argument("--max_chunks", "-m", type=int, default=MAX_CHUNKS,
                        help="Number of Wikipedia chunks to load for SOM training (default: %(default)s)")
    parser.add_argument("--max_questions", "-q", type=int, default=MAX_QUESTIONS,
                        help="Limit number of questions to process (default: %(default)s)")
    # Get arguments for map size as a tuple for example (10,10) and also get rough_len and finetune_len calculation methods (options are rough_len : 'embedding_formula' or 'neurons formuls' | finetune_len : 'embedding_formula' or 'error_convergence')
    parser.add_argument("--map_size", "-s", type=str, default="10,10",
                        help="SOM map size as 'rows,cols' (default: %(default)s)")
    parser.add_argument("--rough_len_method", type=str, choices=['embedding_formula', 'neurons_formula'], default='embedding_formula',
                        help="Method to calculate rough_len (default: %(default)s)")
    parser.add_argument("--finetune_len_method", type=str, choices=['embedding_formula', 'error_convergence'], default='embedding_formula',
                        help="Method to calculate finetune_len (default: %(default)s)")
    parser.add_argument("--lattice", type=str, choices=['rect','hexa'], default='rect',
                        help="The shape of the map nodes (default: %(default)s)")
    parser.add_argument("--num_contexts", type=int, default=NUM_CONTEXTS,
                        help="Number of top contexts to retrieve (default: %(default)s)")
    args = parser.parse_args()

    # Use parsed args to call main function
    # Generate output path if not explicitly provided
    if args.output == OUTPUT_FILE:
        map_size_str = f"{args.map_size.split(',')[0]}x{args.map_size.split(',')[1]}"
        args.output = f"contexts/retrieved_contexts_c{args.max_chunks}_q{args.max_questions}_map_{map_size_str}_{args.lattice}_k{args.num_contexts}"
    
    find_contexts_all_questions(
        input_path=args.input,
        output_path=args.output,
        max_chunks=args.max_chunks,
        max_questions=args.max_questions,
        map_size=tuple(map(int, args.map_size.split(','))),
        rough_len_method=args.rough_len_method,
        finetune_len_method=args.finetune_len_method,
        lattice=args.lattice,
        top_k=args.num_contexts
    )