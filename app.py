import streamlit as st
import os
from src.pdf_processor import extract_text_from_pdf, chunk_text
from src.embeddings import build_vector_store, retrieve_relevant_chunks
from src.gemini_chat import get_gemini_response

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1A3557; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .chat-user { background: #EFF6FF; border-left: 4px solid #2563EB; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }
    .chat-bot { background: #F0FDF4; border-left: 4px solid #16A34A; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }
    .source-box { background: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 6px; padding: 8px 12px; font-size: 0.82rem; color: #6B7280; margin-top: 4px; }
    .stButton > button { background-color: #2563EB; color: white; border: none; border-radius: 6px; padding: 0.5rem 1.2rem; font-weight: 600; }
    .stButton > button:hover { background-color: #1D4ED8; }
</style>
""", unsafe_allow_html=True)

# ── Safe Session State Initialization ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at https://aistudio.google.com/app/apikey"
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — report, textbook, resume, or document"
    )

    if uploaded_file and api_key:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("🔍 Reading and indexing your PDF..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                
                if not raw_text.strip():
                    st.error("Could not extract text from PDF.")
                else:
                    chunks = chunk_text(raw_text)
                    vector_store = build_vector_store(chunks)

                    # Safe assignment
                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []   # Reset chat for new PDF

                    st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

    elif uploaded_file and not api_key:
        st.warning("⚠️ Please enter your Gemini API key first.")

    # Show active PDF info
    if st.session_state.pdf_name:
        st.markdown(f"**Active PDF:** `{st.session_state.pdf_name}`")
        st.markdown(f"**Chunks indexed:** `{len(st.session_state.chunks)}`")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main Chat Area ───────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 Chat with your PDF</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF and ask anything about it — powered by Google Gemini + RAG</div>', unsafe_allow_html=True)

# Display previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🧑 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">🤖 <strong>Gemini:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📎 Source chunks used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f'<div class="source-box"><strong>Chunk {i}:</strong> {src[:300]}{"..." if len(src) > 300 else ""}</div>', unsafe_allow_html=True)

# Chat Input
st.markdown("---")

if not api_key:
    st.info("👈 Enter your **Gemini API key** in the sidebar.")
elif not st.session_state.vector_store:
    st.info("👈 **Upload a PDF** in the sidebar to start chatting.")
else:
    user_input = st.text_input(
        "Ask a question about your PDF",
        placeholder="e.g. Summarize this document... What are the key findings?",
        key="user_input"
    )

    if st.button("Send ➤") and user_input.strip():
        question = user_input.strip()

        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🤔 Thinking..."):
            relevant_chunks = retrieve_relevant_chunks(
                question, 
                st.session_state.vector_store, 
                st.session_state.chunks, 
                top_k=4
            )
            answer = get_gemini_response(question, relevant_chunks)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": relevant_chunks
        })

        st.rerun()