# 🔍 SOM-based Dimensionality Reduction for Efficient Context Retrieval

This project explores and compares two approaches for retrieving relevant context from a document corpus to support **Retrieval-Augmented Generation (RAG)**:

1. **Standard Cosine Similarity Search**  
2. **SOM-based Search using Reduced Vector Embeddings**

The goal is to assess both the **accuracy** (precision, recall, F1) and **efficiency** (time, memory) of context retrieval using Self-Organizing Maps (SOM) vs standard embedding-based cosine similarity.

---

## 🚀 Project Motivation

Large Language Models (LLMs) in RAG pipelines rely on finding the most relevant context to answer queries. However, when the knowledge base grows, performing cosine similarity search over thousands of vectors becomes expensive.

By using **SOM** to reduce the embedding space and pre-group similar vectors, we can:
- Drastically **reduce the number of similarity computations**
- Maintain comparable **retrieval quality**
- Enable **faster and more scalable** RAG deployments

---

## 🧪 Main Experiments

The core experiment compares:

| Method                        | Search Base         | Retrieval Size | Goal                                |
|------------------------------|---------------------|----------------|-------------------------------------|
| **Cosine Similarity**        | All chunks (>1000)  | Top-k (e.g., 5) | High accuracy, High speed, but costly           |
| **SOM (Reduced)**            | BMU hits (~<100)    | Top-k (e.g., 5) | High accuracy, Higher speed, less costly     |

Metrics computed:
- 🧠 Accuracy, Precision, Recall
- ✅ MCC, ROC
- ⏱ Execution time & memory usage (via `tracemalloc`)
  
---

## 📊 Results 

[]

---
