"""
gemini_chat.py
Handles RAG prompt and response generation using Gemini.
"""

import google.generativeai as genai
from typing import List

def get_gemini_response(query: str, relevant_chunks: List[str]) -> str:
    """
    Generate response using Gemini with retrieved context.
    """
    try:
        # Use a currently supported stable model (April 2026)
        model = genai.GenerativeModel('gemini-2.0-flash')   # ← Updated model

        # Create context from retrieved chunks
        context = "\n\n".join(relevant_chunks)

        prompt = f"""
You are a helpful assistant that answers questions based ONLY on the provided context from a PDF document.
If the answer is not in the context, say "I don't have enough information from the document to answer this."

Context from PDF:
{context}

Question: {query}

Answer:
"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            return "❌ Model error: The Gemini model is temporarily unavailable. Please try again in a minute."
        else:
            return f"❌ Error generating response: {error_msg[:200]}"