import streamlit as st
import os

from src.pdf_processor import extract_text_from_pdf, chunk_text
from src.embeddings import build_vector_store, retrieve_relevant_chunks
from src.gemini_chat import get_gemini_response

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="centered")

# Safe Session State Initialization
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

    st.divider()
    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and api_key:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("🔄 Processing PDF..."):
                try:
                    raw_text = extract_text_from_pdf(uploaded_file)
                    chunks = chunk_text(raw_text)
                    vector_store = build_vector_store(chunks)

                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []   # Clear old chat

                    st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    if st.session_state.pdf_name:
        st.info(f"**Active PDF:** {st.session_state.pdf_name}")
        st.info(f"**Chunks:** {len(st.session_state.chunks)}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Area
st.title("📄 Chat with your PDF")
st.caption("Ask anything about the uploaded PDF")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat Input
if not api_key:
    st.info("👈 Enter your Gemini API Key in the sidebar to continue.")
elif st.session_state.vector_store is None:
    st.info("👈 Upload a PDF in the sidebar to start chatting.")
else:
    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🤖 Thinking..."):
            relevant_chunks = retrieve_relevant_chunks(
                prompt, 
                st.session_state.vector_store, 
                st.session_state.chunks, 
                top_k=4
            )
            response = get_gemini_response(prompt, relevant_chunks)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()