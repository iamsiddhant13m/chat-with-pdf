import streamlit as st
import os
from src.pdf_processor import extract_text_from_pdf, chunk_text
from src.embeddings import build_vector_store, retrieve_relevant_chunks
from src.gemini_chat import get_gemini_response

# Page Config
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

# Safe Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    st.markdown("---")
    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and api_key:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                try:
                    raw_text = extract_text_from_pdf(uploaded_file)
                    chunks = chunk_text(raw_text)
                    vector_store = build_vector_store(chunks)

                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []  # Reset chat

                    st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    if st.session_state.pdf_name:
        st.info(f"**Active PDF:** {st.session_state.pdf_name}")
        st.info(f"**Chunks indexed:** {len(st.session_state.chunks)}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Area
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask anything about it — powered by Google Gemini + RAG")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if api_key and st.session_state.vector_store is not None:
    user_input = st.chat_input("Ask a question about your PDF...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            relevant_chunks = retrieve_relevant_chunks(
                user_input, 
                st.session_state.vector_store, 
                st.session_state.chunks
            )
            answer = get_gemini_response(user_input, relevant_chunks)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

elif not api_key:
    st.info("👈 Please enter your Gemini API Key in the sidebar.")
else:
    st.info("👈 Please upload a PDF to start chatting.")