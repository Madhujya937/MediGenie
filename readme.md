# 🩺 MediGenie: Medical Q&A Chatbot

MediGenie is a privacy-first, local medical Q&A chatbot that answers your questions using trusted medical PDFs. It features a modern chat UI, fast PDF search, and optional LLM-powered summarization—never hallucinating or inventing answers.

---

## 🚀 Features

- **PDF-based Medical Knowledge:** Answers are always grounded in your own uploaded medical PDFs (e.g., medical encyclopedias, textbooks).
- **Modern Chat UI:** Clean, ChatGPT-like interface built with Streamlit.
- **Context Highlighting:** Shows the most relevant paragraphs and sections from your PDFs for every answer.
- **LLM Summarization (Optional):** Uses a local or cloud LLM to summarize the retrieved context, but never invents information.
- **Multi-PDF Support:** Index and search across multiple PDFs at once.
- **Fast Startup:** Preprocesses and caches all PDF data for instant Q&A.
- **Privacy-First:** All processing can be done locally—no data leaves your machine.

---

## 🖥️ How to Use

1. **Clone this repo and add your PDFs to the `data/` folder.**
2. **Preprocess PDFs:**
   ```bash
   python preprocess_pdfs.py
   ```
3. **Run the app:**
   ```bash
   streamlit run medigenie_app.py
   ```
4. **Ask questions in your browser!**

---

## 📁 Project Structure

```
medigenie_app.py           # Streamlit chat UI
create_memory_for_llm.py   # PDF processing and search backend
preprocess_pdfs.py         # One-time PDF preprocessing script
data/                      # Your medical PDFs
models/                    # Local LLM model files (optional)
pdf_text_cache.pkl         # Cached PDF data for fast startup
requirements.txt / Pipfile # Python dependencies
```

---

## 📝 Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

---

## ⚠️ What Not to Commit

Add a `.gitignore` file with:
```

venv/
__pycache__/
*.pkl
*.faiss
*.gguf
models/
data/
pdf_text_cache.pkl
.DS_Store
Thumbs.db
.env

```

---


 
