#!/usr/bin/env python3
"""Prepare embeddings for Wikipedia chunks and questions using OpenAI.

Usage:
  python3 prepare_embeddings_openai.py wiki --max_chunks 500
  python3 prepare_embeddings_openai.py wiki --max_chunks 500 --use_openai
  python3 prepare_embeddings_openai.py questions --input questions_answers.xlsx

The script saves outputs under `stored_embeddings/`:
  - stored_embeddings/wiki_docs_openai.pkl            (list of dicts with 'text' and 'meta')
  - stored_embeddings/wiki_embeddings_openai.npy      (numpy array of shape [N, D])
  - stored_embeddings/questions_openai.pkl            (list of question strings)
  - stored_embeddings/question_embeddings_openai.npy  (numpy array of shape [M, D])

It can either use the precomputed embeddings from the Cohere wiki dataset
or call OpenAI embeddings API to (re)compute embeddings.
"""

import argparse
import os
import pickle
import time
from getpass import getpass
from typing import List

import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from openai import OpenAI, RateLimitError, APIError
except Exception:
    OpenAI = None
    RateLimitError = Exception
    APIError = Exception


DEFAULT_EMBED_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def robust_openai_embedding(
    client: "OpenAI",
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> List[float]:
    """
    Get embedding from OpenAI API with retry logic.
    
    Args:
        client: OpenAI client instance
        text: Text to embed
        model: OpenAI embedding model to use
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        List of floats representing the embedding
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text,
                model=model
            )
            
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                if embedding and len(embedding) > 0:
                    return embedding
            
            raise ValueError("OpenAI returned empty embedding")
            
        except RateLimitError as e:
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limited by OpenAI; sleeping {sleep_time:.1f}s then retrying...", flush=True)
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts due to rate limiting: {e}")
                
        except APIError as e:
            if attempt < max_retries - 1:
                sleep_time = retry_delay
                print(f"API error from OpenAI: {e}; retrying in {sleep_time:.1f}s...", flush=True)
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts due to API error: {e}")
                
        except Exception as e:
            raise RuntimeError(f"Unexpected error getting OpenAI embedding: {e}")
    
    raise RuntimeError("Failed to obtain embedding from OpenAI")


def save_pickle(obj, path: str):
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_wikipedia_and_embeddings_openai(
    max_chunks: int = 500,
    use_openai: bool = False,
    openai_api_key: str = None,
    model: str = DEFAULT_EMBED_MODEL,
    streaming: bool = False
) -> None:
    """
    Load Wikipedia chunks from Cohere dataset and optionally re-embed with OpenAI.
    
    Args:
        max_chunks: Maximum number of chunks to load
        use_openai: If True, re-compute embeddings using OpenAI API
        openai_api_key: OpenAI API key (if None, will prompt or use env var)
        model: OpenAI embedding model to use
        streaming: Whether to use streaming dataset API
    """
    if load_dataset is None:
        raise RuntimeError("datasets package is required to load the wiki dataset")

    print(f"Loading up to {max_chunks} wiki chunks from Cohere dataset...")
    
    if streaming:
        stream = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings",
            split="train",
            streaming=True
        )
    else:
        stream = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings",
            split=f"train[:{max_chunks}]",
            streaming=False
        )

    docs = []
    embeddings = []
    
    for doc in tqdm(stream, total=max_chunks):
        docs.append({"text": doc.get("text", ""), "meta": {"title": doc.get("title", "")}})
        # Use dataset embedding if present and not re-embedding
        emb = doc.get("emb")
        if emb is not None and not use_openai:
            embeddings.append(np.array(emb, dtype=float))
        else:
            # Placeholder, will fill later
            embeddings.append(None)
        
        if len(docs) >= max_chunks:
            break

    # If use_openai, or some embeddings are None, compute them via OpenAI
    need_recompute = use_openai or any(e is None for e in embeddings)
    if need_recompute:
        if OpenAI is None:
            raise RuntimeError("openai package is required for embedding computation")
        
        # Get API key
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            openai_api_key = getpass("Enter your OpenAI API key: ")
        
        client = OpenAI(api_key=openai_api_key)
        print(f"Computing embeddings via OpenAI API (model: {model})...")
        
        for idx in tqdm(range(len(docs))):
            text = docs[idx]["text"]
            emb = robust_openai_embedding(client, text, model=model)
            embeddings[idx] = np.array(emb, dtype=float)
            time.sleep(0.05)  # Small delay to avoid rate limits

    embeddings_array = np.vstack(embeddings)

    # Save
    os.makedirs("stored_embeddings", exist_ok=True)
    save_pickle(docs, "stored_embeddings/wiki_docs_openai.pkl")
    np.save("stored_embeddings/wiki_embeddings_openai.npy", embeddings_array)
    
    print(f"Saved {embeddings_array.shape[0]} wiki embeddings to stored_embeddings/wiki_embeddings_openai.npy")
    print(f"Wiki embeddings shape: {embeddings_array.shape}")


def load_questions_and_embed_openai(
    input_excel: str = "questions_answers.xlsx",
    openai_api_key: str = None,
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 100,
    rate_limit_sleep: float = 1.0
) -> None:
    """
    Load questions from Excel and embed using OpenAI API.
    
    Args:
        input_excel: Path to Excel file with 'question' column
        openai_api_key: OpenAI API key (if None, will prompt or use env var)
        model: OpenAI embedding model to use
        batch_size: Number of questions to process before sleeping
        rate_limit_sleep: Seconds to sleep between batches
    """
    import pandas as pd
    
    if OpenAI is None:
        raise RuntimeError("openai package is required")
    
    if not os.path.exists(input_excel):
        raise FileNotFoundError(f"Questions Excel file not found: {input_excel}")

    df = pd.read_excel(input_excel)
    if "question" not in df.columns:
        raise ValueError("Excel file must contain a 'question' column")

    questions = df["question"].astype(str).str.strip().tolist()
    print(f"Loaded {len(questions)} questions from {input_excel}")

    # Get API key
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = getpass("Enter your OpenAI API key: ")
    
    client = OpenAI(api_key=openai_api_key)
    print(f"OpenAI API client initialized (model: {model}).")

    # Embed questions using OpenAI API
    print(f"Embedding {len(questions)} questions using OpenAI API...")
    embeddings = []
    
    for idx, q in enumerate(tqdm(questions)):
        emb = robust_openai_embedding(client, q, model=model)
        embeddings.append(np.array(emb, dtype=float))
        
        # Sleep periodically to avoid rate limits
        if (idx + 1) % batch_size == 0:
            time.sleep(rate_limit_sleep)

    emb_array = np.vstack(embeddings)
    
    # Save
    os.makedirs("stored_embeddings", exist_ok=True)
    save_pickle(questions, "stored_embeddings/questions_openai.pkl")
    np.save("stored_embeddings/question_embeddings_openai.npy", emb_array)
    
    print(f"Saved {emb_array.shape[0]} question embeddings to stored_embeddings/question_embeddings_openai.npy")
    print(f"Question embeddings shape: {emb_array.shape}")


# ============================================================
# Main Function
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare embeddings for wiki and questions using OpenAI API")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Wiki subcommand
    p_wiki = sub.add_parser("wiki", help="Load wiki chunks and save embeddings")
    p_wiki.add_argument("--max_chunks", type=int, default=500, help="Maximum number of chunks to load (default: 500)")
    p_wiki.add_argument("--use_openai", action="store_true", help="Force computing embeddings with OpenAI API")
    p_wiki.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (if not provided, will prompt or use OPENAI_API_KEY env var)")
    p_wiki.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL, help=f"OpenAI embedding model (default: {DEFAULT_EMBED_MODEL})")
    p_wiki.add_argument("--streaming", action="store_true", help="Use streaming dataset API (may crash on exit); default off")

    # Questions subcommand
    p_q = sub.add_parser("questions", help="Load questions from Excel and embed using OpenAI API")
    p_q.add_argument("--input", type=str, default="questions_answers.xlsx", help="Path to Excel file with questions")
    p_q.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key (if not provided, will prompt or use OPENAI_API_KEY env var)")
    p_q.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL, help=f"OpenAI embedding model (default: {DEFAULT_EMBED_MODEL})")
    p_q.add_argument("--batch_size", type=int, default=100, help="Process this many questions before sleeping (default: 100)")
    p_q.add_argument("--rate_limit_sleep", type=float, default=1.0, help="Seconds to sleep between batches (default: 1.0)")

    args = parser.parse_args()

    if args.cmd == "wiki":
        load_wikipedia_and_embeddings_openai(
            max_chunks=args.max_chunks,
            use_openai=args.use_openai,
            openai_api_key=args.openai_api_key,
            model=args.model,
            streaming=args.streaming
        )
    elif args.cmd == "questions":
        load_questions_and_embed_openai(
            input_excel=args.input,
            openai_api_key=args.openai_api_key,
            model=args.model,
            batch_size=args.batch_size,
            rate_limit_sleep=args.rate_limit_sleep
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
