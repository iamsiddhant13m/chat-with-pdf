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
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .chat-user {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .chat-bot {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .source-box {
        background: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.82rem;
        color: #6B7280;
        margin-top: 4px;
    }
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

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
                # Extract text
                raw_text = extract_text_from_pdf(uploaded_file)
                if not raw_text.strip():
                    st.error("Could not extract text. Make sure the PDF is not scanned/image-only.")
                else:
                    # Chunk and embed
                    chunks = chunk_text(raw_text)
                    vector_store = build_vector_store(chunks)

                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []  # reset chat for new PDF

                    st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

    elif uploaded_file and not api_key:
        st.warning("⚠️ Please enter your Gemini API key above first.")

    if st.session_state.pdf_name:
        st.markdown(f"**Active PDF:** `{st.session_state.pdf_name}`")
        st.markdown(f"**Chunks indexed:** `{len(st.session_state.chunks)}`")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Upload any PDF
    2. App splits it into chunks
    3. Chunks are embedded using Google's embedding model
    4. Your question retrieves the most relevant chunks (RAG)
    5. Gemini generates an answer grounded in your document
    
    *Built by Siddhant Mishra*  
    *Stack: Python · Streamlit · Gemini API · RAG*
    """)

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 Chat with your PDF</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF and ask anything about it — powered by Google Gemini + RAG</div>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🧑 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">🤖 <strong>Gemini:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📎 Source chunks used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f'<div class="source-box"><strong>Chunk {i}:</strong> {src[:300]}{"..." if len(src) > 300 else ""}</div>', unsafe_allow_html=True)

# Input area
st.markdown("---")

if not api_key:
    st.info("👈 Enter your **Gemini API key** in the sidebar to get started.")
elif not st.session_state.vector_store:
    st.info("👈 **Upload a PDF** in the sidebar to begin chatting.")
else:
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question about your PDF",
            placeholder="e.g. What are the key findings? Summarize chapter 2. What does it say about X?",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    if send and user_input.strip():
        question = user_input.strip()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🤔 Thinking..."):
            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(
                question,
                st.session_state.vector_store,
                st.session_state.chunks,
                top_k=4
            )

            # Get Gemini response
            answer = get_gemini_response(question, relevant_chunks)

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": relevant_chunks
        })

        st.rerun()

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "Summarize this document in 5 bullet points",
            "What are the key topics covered?",
            "What conclusions or recommendations are mentioned?",
            "List any numbers, dates, or statistics mentioned",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {s}", key=f"sug_{i}"):
                    st.session_state.messages.append({"role": "user", "content": s})
                    with st.spinner("🤔 Thinking..."):
                        relevant_chunks = retrieve_relevant_chunks(
                            s, st.session_state.vector_store,
                            st.session_state.chunks, top_k=4
                        )
                        answer = get_gemini_response(s, relevant_chunks)
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": relevant_chunks
                    })
                    st.rerun()
