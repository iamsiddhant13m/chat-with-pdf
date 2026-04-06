"""
embeddings.py - Simple & Stable version for Gemini Embedding
"""

import os
import numpy as np
from typing import List
import google.generativeai as genai


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Please enter your API key in the sidebar.")
    genai.configure(api_key=api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed document chunks - simple and stable"""
    _get_client()
    embeddings = []
    
    for i, text in enumerate(texts):
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text
                # task_type removed - causing issues on Streamlit Cloud
            )
            embeddings.append(result["embedding"])
            
            if (i + 1) % 8 == 0 or i == len(texts) - 1:
                print(f"✅ Embedded {i+1}/{len(texts)} chunks")
                
        except Exception as e:
            print(f"❌ Embedding error on chunk {i}: {str(e)[:200]}")
            raise
    
    return embeddings


def embed_query(query: str) -> List[float]:
    """Embed user query"""
    _get_client()
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query
    )
    return result["embedding"]


def build_vector_store(chunks: List[str]) -> np.ndarray:
    print(f"🔄 Starting to embed {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    vector_store = np.array(embeddings, dtype=np.float32)
    print(f"✅ Vector store created successfully! Shape: {vector_store.shape}")
    return vector_store


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
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
    query_embedding = np.array(embed_query(query), dtype=np.float32)

    similarities = []
    for i, chunk_embedding in enumerate(vector_store):
        score = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((score, i))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunks[idx] for _, idx in similarities[:top_k]]
    return top_chunks