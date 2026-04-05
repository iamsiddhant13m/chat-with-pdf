# 📄 Chat with PDF — Gemini RAG App

A beginner-friendly AI project that lets you **upload any PDF and chat with it** using Google Gemini and a Retrieval-Augmented Generation (RAG) pipeline. Built with Python and Streamlit.

**Built by:** Siddhant Mishra  
**Stack:** Python · Streamlit · Google Gemini API · RAG · Numpy

---

## 🚀 Live Demo

> Deploy your own on [Streamlit Cloud](https://streamlit.io/cloud) for free — see deployment steps below.

---

## 🧠 How It Works (RAG Pipeline)

```
PDF Upload
    ↓
Extract Text (PyPDF2)
    ↓
Split into Chunks (~800 chars with overlap)
    ↓
Embed each chunk (Google text-embedding-004)
    ↓  ← stored in memory as numpy array
User asks a question
    ↓
Embed the question
    ↓
Cosine Similarity Search → Top 4 relevant chunks
    ↓
Send [question + chunks] to Gemini 1.5 Flash
    ↓
Grounded answer displayed to user
```

This is a **from-scratch RAG implementation** — no LangChain or vector DB required. Just Google's embedding model + numpy cosine similarity + Gemini for generation.

---

## 📁 Project Structure

```
chat-with-pdf/
│
├── app.py                  # Main Streamlit UI
├── requirements.txt        # Dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml         # UI theme config
│
└── src/
    ├── __init__.py
    ├── pdf_processor.py    # PDF text extraction + chunking
    ├── embeddings.py       # Google embeddings + cosine similarity search
    └── gemini_chat.py      # Gemini answer generation with RAG prompt
```

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get your free Gemini API key
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **Create API Key**
3. Copy it — you'll paste it into the app's sidebar

### 5. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app** → select your repo → set main file as `app.py`
4. Click **Deploy**
5. Enter your Gemini API key in the sidebar when the app loads

> ⚠️ **Do NOT** commit your API key to GitHub. Always enter it through the sidebar or use Streamlit secrets.

---

## 💡 What You Can Ask

- *"Summarize this document in 5 bullet points"*
- *"What are the key findings or conclusions?"*
- *"What does it say about [specific topic]?"*
- *"List all dates or numbers mentioned"*
- *"What recommendations are made?"*

---

## 🔧 Key Concepts Used

| Concept | Where | Description |
|---|---|---|
| **Text Chunking** | `pdf_processor.py` | Splits PDF text into overlapping chunks to preserve context |
| **Text Embeddings** | `embeddings.py` | Converts text to vectors using `text-embedding-004` |
| **Cosine Similarity** | `embeddings.py` | Finds most relevant chunks for the user's question |
| **RAG Prompt** | `gemini_chat.py` | Sends question + context chunks to Gemini together |
| **Gemini 1.5 Flash** | `gemini_chat.py` | Fast, free-tier LLM for generating grounded answers |

---

## 📌 Future Improvements

- [ ] Support for multiple PDFs at once
- [ ] Chat history export
- [ ] Better handling of scanned/image PDFs (OCR with Tesseract)
- [ ] Persistent vector store (save embeddings to disk)
- [ ] Switch to a proper vector DB (ChromaDB, FAISS) for large documents

---

## 📜 License

MIT License — free to use, modify, and share.
