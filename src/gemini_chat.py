"""
gemini_chat.py
Handles RAG prompt and response generation using Gemini (updated April 2026)
"""

import google.generativeai as genai
from typing import List

def get_gemini_response(query: str, relevant_chunks: List[str]) -> str:
    """
    Generate response using Gemini with retrieved context.
    """
    try:
        # Use a currently supported stable model
        model = genai.GenerativeModel('gemini-2.5-flash')   # ← This is the recommended stable model right now

        context = "\n\n".join(relevant_chunks)

        prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the following context from the PDF.
If the answer is not in the context, clearly say "I don't have enough information in the document to answer this."

Context:
{context}

Question: {query}

Answer in a clear and concise way:
"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str:
            return "❌ The Gemini model is temporarily unavailable. Please wait 30 seconds and try asking again."
        else:
            return f"❌ Sorry, I encountered an error while generating the answer: {str(e)[:150]}"