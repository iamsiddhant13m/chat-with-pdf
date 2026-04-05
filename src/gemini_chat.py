"""
gemini_chat.py
Handles generating answers using Google Gemini,
grounded in the retrieved context chunks (RAG).
"""

import os
from typing import List
import google.generativeai as genai


SYSTEM_PROMPT = """You are a helpful document assistant. Your job is to answer questions 
based ONLY on the provided context extracted from the user's PDF document.

Rules:
- Answer only from the given context. Do not use outside knowledge.
- If the answer is not in the context, say: "I couldn't find information about that in the document."
- Be clear, concise, and accurate.
- If quoting from the document, keep quotes short and relevant.
- If asked to summarize, use bullet points for clarity.
"""


def get_gemini_response(question: str, context_chunks: List[str]) -> str:
    """
    Generate an answer to the question using Gemini,
    grounded in the provided context chunks.

    Args:
        question: User's question
        context_chunks: Relevant text chunks retrieved from the PDF

    Returns:
        Gemini's answer as a string
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "❌ API key not found. Please enter your Gemini API key in the sidebar."

    genai.configure(api_key=api_key)

    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    # Construct the prompt
    prompt = f"""{SYSTEM_PROMPT}

---
CONTEXT FROM DOCUMENT:
{context}

---
USER QUESTION:
{question}

---
YOUR ANSWER:"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,        # Low temperature = more factual, less creative
                max_output_tokens=1024,
            )
        )
        return response.text

    except Exception as e:
        return f"❌ Gemini API error: {str(e)}\n\nPlease check your API key and try again."
