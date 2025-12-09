#!/usr/bin/env python3
"""Prepare embeddings using Cohere API for Wikipedia chunks and questions.

Usage:
  python3 prepare_embeddings_cohere.py wiki --max_chunks 500
  python3 prepare_embeddings_cohere.py questions --input questions_answers.xlsx

The script saves outputs under `stored_embeddings/`:
  - stored_embeddings/wiki_docs_cohere.pkl            (list of dicts with 'text' and 'meta')
  - stored_embeddings/wiki_embeddings_cohere.npy      (numpy array of shape [N, D])
  - stored_embeddings/questions_cohere.pkl            (list of question strings)
  - stored_embeddings/question_embeddings_cohere.npy  (numpy array of shape [M, D])

Uses Cohere API for embeddings:
  - Wikipedia: loads pre-embedded chunks from Cohere dataset
  - Questions: embeds questions using Cohere's multilingual-22-12 model
"""

import os
import pickle
import argparse
import time
from getpass import getpass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    import cohere
except Exception:
    cohere = None

try:
    from cohere.errors import TooManyRequestsError
except Exception:
    TooManyRequestsError = Exception


# ============================================================
# Helper Functions
# ============================================================

def save_pickle(obj, path: str):
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_wikipedia_and_embeddings_cohere(max_chunks: int = 500, streaming: bool = False) -> None:
    """
    Load Wikipedia chunks from Cohere dataset with pre-computed embeddings.
    
    Args:
        max_chunks: Maximum number of chunks to load
        streaming: If True, use streaming API (may trigger background threads). Default False.
    """
    if load_dataset is None:
        raise RuntimeError("datasets package is required to stream the wiki dataset")

    print(f"Loading up to {max_chunks} wiki chunks from Cohere dataset...")

    if streaming:
        stream = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings",
            split="train",
            streaming=True
        )
    else:
        # Non-streaming avoids lingering background threads that can crash on interpreter shutdown
        stream = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings",
            split=f"train[:{max_chunks}]",
            streaming=False
        )

    docs = []
    embeddings = []
    
    iterable = tqdm(stream, total=max_chunks)
    for doc in iterable:
        docs.append({
            "text": doc.get("text", ""),
            "meta": {"title": doc.get("title", "")}
        })
        emb = doc.get("emb")
        if emb is not None:
            embeddings.append(np.array(emb, dtype=float))
        else:
            raise ValueError("Cohere dataset embedding is missing; cannot proceed")
        
        if len(docs) >= max_chunks:
            break

    embeddings_array = np.vstack(embeddings)

    # Save
    os.makedirs("stored_embeddings", exist_ok=True)
    save_pickle(docs, "stored_embeddings/wiki_docs_cohere.pkl")
    np.save("stored_embeddings/wiki_embeddings_cohere.npy", embeddings_array)
    
    print(f"Saved {embeddings_array.shape[0]} wiki chunks to stored_embeddings/wiki_embeddings_cohere.npy")
    print(f"Wiki embeddings shape: {embeddings_array.shape}")


def load_questions_and_embed_cohere(input_excel: str = "questions_answers.xlsx",
                                    cohere_api_key: str = None,
                                    output_dimension: int = 768,
                                    model: str = "multilingual-22-12",
                                    batch_size: int = 96,
                                    rate_limit_sleep: float = 2.0) -> None:
    """
    Load questions from Excel and embed using Cohere API.
    
    Args:
        input_excel: Path to Excel file with 'question' column
        cohere_api_key: Cohere API key (if None, will prompt user)
        output_dimension: Output embedding dimension
        model: Cohere model to use
    """
    if cohere is None:
        raise RuntimeError("cohere package is required")
    
    if not os.path.exists(input_excel):
        raise FileNotFoundError(f"Questions Excel file not found: {input_excel}")

    df = pd.read_excel(input_excel)
    if "question" not in df.columns:
        raise ValueError(f"Excel file must contain a 'question' column. Found columns: {df.columns.tolist()}")

    questions = df["question"].astype(str).str.strip().tolist()
    print(f"Loaded {len(questions)} questions from {input_excel}")

    # Get API key if not provided
    if not cohere_api_key:
        cohere_api_key = getpass("Enter your Cohere API key: ")

    co = cohere.ClientV2(api_key=cohere_api_key)
    print("Cohere API client initialized.")

    # Embed questions using Cohere API
    if batch_size <= 0:
        batch_size = 96

    all_embs = []
    for start in tqdm(range(0, len(questions), batch_size)):
        end = min(start + batch_size, len(questions))
        batch = questions[start:end]

        while True:
            try:
                res = co.embed(
                    texts=batch,
                    model=model,
                    input_type="search_query",
                    output_dimension=output_dimension,
                    embedding_types=["float"],
                )
                break
            except TooManyRequestsError:
                sleep_for = max(rate_limit_sleep, 1.0)
                print(f"Rate limited by Cohere; sleeping {sleep_for:.1f}s then retrying...", flush=True)
                time.sleep(sleep_for)

        if res.embeddings and res.embeddings.float:
            all_embs.append(np.array(res.embeddings.float, dtype=float))
        else:
            raise ValueError("Cohere embed API did not return expected embeddings")

    emb_array = np.vstack(all_embs)

    # Save
    os.makedirs("stored_embeddings", exist_ok=True)
    save_pickle(questions, "stored_embeddings/questions_cohere.pkl")
    np.save("stored_embeddings/question_embeddings_cohere.npy", emb_array)
    
    print(f"Saved {emb_array.shape[0]} question embeddings to stored_embeddings/question_embeddings_cohere.npy")
    print(f"Question embeddings shape: {emb_array.shape}")


# ============================================================
# Main Function
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare embeddings for wiki and questions using Cohere API")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Wiki subcommand
    p_wiki = sub.add_parser("wiki", help="Load wiki chunks from Cohere dataset and save embeddings")
    p_wiki.add_argument("--max_chunks", type=int, default=500, help="Maximum number of chunks to load (default: 500)")
    p_wiki.add_argument("--streaming", action="store_true", help="Use streaming dataset API (may crash on exit); default off")

    # Questions subcommand
    p_q = sub.add_parser("questions", help="Load questions from Excel and embed using Cohere API")
    p_q.add_argument("--input", type=str, default="questions_answers.xlsx", help="Path to Excel file with questions")
    p_q.add_argument("--cohere_api_key", type=str, default=None, help="Cohere API key (if not provided, will prompt)")
    p_q.add_argument("--output_dimension", type=int, default=768, help="Output embedding dimension (default: 768)")
    p_q.add_argument("--model", type=str, default="multilingual-22-12", help="Cohere model to use (default: multilingual-22-12)")
    p_q.add_argument("--batch_size", type=int, default=96, help="Batch size for Cohere embed calls (max 96)")
    p_q.add_argument("--rate_limit_sleep", type=float, default=2.0, help="Seconds to sleep when rate limited (default: 2.0)")

    args = parser.parse_args()

    if args.cmd == "wiki":
        load_wikipedia_and_embeddings_cohere(max_chunks=args.max_chunks, streaming=args.streaming)
    elif args.cmd == "questions":
        load_questions_and_embed_cohere(
            input_excel=args.input,
            cohere_api_key=args.cohere_api_key,
            output_dimension=args.output_dimension,
            model=args.model,
            batch_size=args.batch_size,
            rate_limit_sleep=args.rate_limit_sleep
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()