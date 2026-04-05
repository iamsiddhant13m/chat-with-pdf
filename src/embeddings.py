"""
embeddings.py
Handles text embedding using Google's embedding model
and cosine similarity search (no external vector DB needed).
"""

import os
import numpy as np
from typing import List, Tuple
import google.generativeai as genai


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Please enter your API key in the sidebar.")
    genai.configure(api_key=api_key)
    return genai


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings using Google's text-embedding model.
    Returns a list of embedding vectors.
    """
    client = _get_client()
    embeddings = []

    # Batch in groups of 100 to stay within API limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=batch,
            task_type="retrieval_document",
        )
        embeddings.extend(result["embedding"])

    return embeddings


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    Uses retrieval_query task type for better search performance.
    """
    client = _get_client()
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]


def build_vector_store(chunks: List[str]) -> np.ndarray:
    """
    Build an in-memory vector store from text chunks.

    Args:
        chunks: List of text chunks to embed

    Returns:
        numpy array of shape (num_chunks, embedding_dim)
    """
    embeddings = embed_texts(chunks)
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def retrieve_relevant_chunks(
    query: str,
    vector_store: np.ndarray,
    chunks: List[str],
    top_k: int = 4
) -> List[str]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    This is the core of the RAG pipeline:
    1. Embed the query
    2. Compute cosine similarity against all chunk embeddings
    3. Return the top-k chunks by similarity score

    Args:
        query: User's question
        vector_store: Pre-built numpy array of chunk embeddings
        chunks: Original text chunks (parallel to vector_store rows)
        top_k: Number of chunks to retrieve

    Returns:
        List of the most relevant text chunks
    """
    query_embedding = np.array(embed_query(query), dtype=np.float32)

    # Compute similarity between query and every chunk
    similarities = []
    for i, chunk_embedding in enumerate(vector_store):
        score = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((score, i))

    # Sort descending by similarity score
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Return top-k chunk texts
    top_chunks = [chunks[idx] for _, idx in similarities[:top_k]]
    return top_chunks
