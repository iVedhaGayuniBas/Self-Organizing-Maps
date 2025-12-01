#!/usr/bin/env python3
"""Prepare embeddings for Wikipedia chunks and questions.

Usage:
  python3 prepare_embeddings.py wiki   --max_chunks 500
  python3 prepare_embeddings.py wiki --max_chunks 500 --use_ollama
  python3 prepare_embeddings.py questions --input questions_answers.xlsx

The script saves outputs under `stored_embeddings/`:
  - stored_embeddings/wiki_docs.pkl            (list of dicts with 'text' and 'emb')
  - stored_embeddings/wiki_embeddings.npy      (numpy array of shape [N, D])
  - stored_embeddings/questions.pkl            (list of question strings)
  - stored_embeddings/question_embeddings.npy  (numpy array of shape [M, D])
It can either use the precomputed embeddings from the Cohere wiki dataset
or call a local Ollama embeddings endpoint to (re)compute embeddings.
"""

import argparse
import os
import pickle
import time
from typing import List

import numpy as np
import requests
from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large:335m")


def robust_ollama_embedding(text: str, model: str = DEFAULT_EMBED_MODEL, timeout: int = 30, tries_per_variant: int = 1):
    # Try payloads in an order that works for common Ollama embedding models.
    # Some embedding models expect `prompt` while others expect `input` (string or list).
    payload_variants = [
        {"model": model, "prompt": text},
        {"model": model, "input": [text]},
        {"model": model, "input": text},
    ]

    last_response = None
    # small backoff parameters
    for payload in payload_variants:
        for attempt in range(max(1, tries_per_variant)):
            try:
                r = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=timeout)
                last_response = r
                r.raise_for_status()
                # try to parse JSON, but be defensive
                try:
                    j = r.json()
                except Exception:
                    j = None

                # Try common response shapes for embeddings
                cand = None
                if isinstance(j, dict):
                    if "embedding" in j and j["embedding"] is not None:
                        cand = j["embedding"]
                    elif "data" in j and isinstance(j["data"], list) and j["data"]:
                        d0 = j["data"][0]
                        if isinstance(d0, dict) and "embedding" in d0 and d0["embedding"] is not None:
                            cand = d0["embedding"]
                    elif "embeddings" in j and isinstance(j["embeddings"], list) and j["embeddings"]:
                        # sometimes embeddings is a list-of-lists
                        cand = j["embeddings"][0]

                # If candidate exists and non-empty, return it
                if cand is not None and hasattr(cand, "__len__") and len(cand) > 0:
                    return cand

                # If candidate exists but empty, log full response text for debugging and continue
                if j is not None and isinstance(j, dict) and (
                    ("embedding" in j and (j["embedding"] == [] or j["embedding"] is None)) or
                    ("data" in j and isinstance(j["data"], list) and j["data"] and (j["data"][0].get("embedding") == [] or j["data"][0].get("embedding") is None))
                ):
                    print("Warning: Ollama returned an empty embedding. Full response below:")
                    try:
                        print(json.dumps(j, indent=2)[:4000])
                    except Exception:
                        print(str(j)[:4000])

            except requests.RequestException as e:
                print(f"Network/error from Ollama: {e}; payload: {payload}")
            except Exception as e:
                print(f"Unexpected parsing error: {e}; payload: {payload}")

    # If no payload returned a valid embedding, raise with helpful context (include last response text)
    body = None
    if last_response is not None:
        try:
            body = last_response.text
        except Exception:
            body = '<unreadable response body>'
    raise RuntimeError(f"Failed to obtain non-empty embedding from Ollama (model={model}). Last response: {body}")


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_wikipedia_and_embeddings(max_chunks: int = 500, use_ollama: bool = False) -> None:
    if load_dataset is None:
        raise RuntimeError("datasets package is required to stream the wiki dataset")

    print(f"Loading up to {max_chunks} wiki chunks from dataset...")
    stream = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)

    docs = []
    embeddings = []
    for i, doc in enumerate(tqdm(stream, total=max_chunks)):
        docs.append({"text": doc.get("text", ""), "meta": {}})
        # prefer dataset embedding if present and not re-embedding
        emb = doc.get("emb")
        if emb is not None and not use_ollama:
            embeddings.append(np.array(emb, dtype=float))
        else:
            # placeholder, will fill later
            embeddings.append(None)
        if len(docs) >= max_chunks:
            break

    # If use_ollama, or some embeddings are None, compute them via Ollama
    need_recompute = use_ollama or any(e is None for e in embeddings)
    if need_recompute:
        print("Computing embeddings via Ollama (this can take a while)...")
        for idx in tqdm(range(len(docs))):
            text = docs[idx]["text"]
            emb = robust_ollama_embedding(text)
            embeddings[idx] = np.array(emb, dtype=float)
            time.sleep(0.01)

    embeddings_array = np.vstack(embeddings)

    # Save
    os.makedirs("stored_embeddings", exist_ok=True)
    save_pickle(docs, "stored_embeddings/wiki_docs.pkl")
    np.save("stored_embeddings/wiki_embeddings.npy", embeddings_array)
    print(f"Saved {embeddings_array.shape[0]} wiki embeddings to stored_embeddings/wiki_embeddings.npy")
    print(f"wiki embeddings shape: {embeddings_array.shape}")

def load_questions_and_embed(input_excel: str = "questions_answers.xlsx", use_ollama: bool = True) -> None:
    import pandas as pd

    if not os.path.exists(input_excel):
        raise FileNotFoundError(f"Questions Excel file not found: {input_excel}")

    df = pd.read_excel(input_excel)
    if "question" not in df.columns:
        raise ValueError("Excel file must contain a 'question' column")

    questions = df["question"].astype(str).str.strip().tolist()
    print(f"Loaded {len(questions)} questions from {input_excel}")

    embeddings = []
    if use_ollama:
        print("Embedding questions using Ollama...")
        for q in tqdm(questions):
            emb = robust_ollama_embedding(q)
            embeddings.append(np.array(emb, dtype=float))
            time.sleep(0.005)
    else:
        raise RuntimeError("Only Ollama embedding is implemented for questions currently")

    emb_array = np.vstack(embeddings)
    os.makedirs("stored_embeddings", exist_ok=True)
    save_pickle(questions, "stored_embeddings/questions.pkl")
    np.save("stored_embeddings/question_embeddings.npy", emb_array)
    print(f"Saved {emb_array.shape[0]} question embeddings to stored_embeddings/question_embeddings.npy")
    print(f"question embeddings shape: {emb_array.shape}")

def main():
    parser = argparse.ArgumentParser(description="Prepare embeddings for wiki and questions")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wiki = sub.add_parser("wiki", help="Load wiki chunks and save embeddings")
    p_wiki.add_argument("--max_chunks", type=int, default=500)
    p_wiki.add_argument("--use_ollama", action="store_true", help="Force computing embeddings with Ollama")

    p_q = sub.add_parser("questions", help="Load questions from Excel and embed them")
    p_q.add_argument("--input", type=str, default="questions_answers.xlsx")
    p_q.add_argument("--no_ollama", dest="use_ollama", action="store_false")

    args = parser.parse_args()

    if args.cmd == "wiki":
        load_wikipedia_and_embeddings(max_chunks=args.max_chunks, use_ollama=args.use_ollama)
    elif args.cmd == "questions":
        load_questions_and_embed(input_excel=args.input, use_ollama=args.use_ollama)


if __name__ == "__main__":
    main()
