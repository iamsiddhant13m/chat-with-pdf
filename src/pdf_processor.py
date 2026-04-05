"""
pdf_processor.py
Handles PDF text extraction and text chunking for RAG pipeline.
"""

import io
import re
from typing import List
import PyPDF2


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded PDF file.
    Works with Streamlit's UploadedFile object.
    """
    text = ""
    try:
        pdf_bytes = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Add a page marker so we can reference it later
                text += f"\n[Page {page_num + 1}]\n{page_text}\n"

    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks for embedding.

    Args:
        text: Full document text
        chunk_size: Target characters per chunk
        overlap: Characters to overlap between consecutive chunks
                 (preserves context across chunk boundaries)

    Returns:
        List of text chunks
    """
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            # Last chunk — take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to break at a sentence boundary (. ! ?) or newline
        # so chunks don't cut mid-sentence
        boundary = -1
        for punct in ['\n\n', '.\n', '. ', '!\n', '! ', '?\n', '? ', '\n']:
            idx = text.rfind(punct, start + chunk_size // 2, end)
            if idx != -1:
                boundary = idx + len(punct)
                break

        if boundary == -1:
            # No good boundary found — just cut at chunk_size
            boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward with overlap
        start = boundary - overlap

    return chunks
