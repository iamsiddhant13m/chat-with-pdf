"""
embeddings.py
Handles text embedding using Google's Gemini embedding model
and cosine similarity search.
"""

import os
import numpy as np
from typing import List
import google.generativeai as genai


def _get_client():
    """Configure Gemini API client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Please enter your API key in the sidebar.")
    genai.configure(api_key=api_key)
    return genai


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of text strings."""
    _get_client()
    embeddings = []
    
    for i, text in enumerate(texts):
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
            
            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                print(f"Embedded {i+1}/{len(texts)} chunks")
                
        except Exception as e:
            print(f"Error embedding chunk {i}: {str(e)}")
            raise
    
    return embeddings


def embed_query(query: str) -> List[float]:
    """Embed a single user query."""
    _get_client()
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return result["embedding"]


def build_vector_store(chunks: List[str]) -> np.ndarray:
    """Build vector store from chunks."""
    print(f"🔄 Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    vector_store = np.array(embeddings, dtype=np.float32)
    print(f"✅ Vector store built! Shape: {vector_store.shape}")
    return vector_store


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity."""
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
    """Retrieve top-k relevant chunks."""
    query_embedding = np.array(embed_query(query), dtype=np.float32)

    similarities = []
    for i, chunk_embedding in enumerate(vector_store):
        score = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((score, i))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunks[idx] for _, idx in similarities[:top_k]]
    return top_chunks