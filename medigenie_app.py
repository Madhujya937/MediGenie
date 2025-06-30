import streamlit as st
import os
import pickle
from create_memory_for_llm import (
    extract_term_from_question,
    multi_pdf_extract_definition_section,
    summarize_with_llm
)

st.set_option('client.showErrorDetails', True)
st.set_page_config(page_title="MediGenie: Medical Q&A Chatbot", layout="centered")

# Custom CSS for improved chat UI
st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .chat-message {
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
        word-break: break-word;
        font-size: 1.05rem;
    }
    .user-message {
        background-color: #DCF8C6;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 0.2rem;
    }
    .bot-message {
        background-color: #F1F0F0;
        align-self: flex-start;
        margin-right: auto;
        border-bottom-left-radius: 0.2rem;
    }
    .chat-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1976d2;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for PDF status
with st.sidebar:
    cache_file = "pdf_text_cache.pkl"
    if not os.path.exists(cache_file):
        st.warning("No cached PDF data found. Please run preprocess_pdfs.py first.")
        st.stop()
    with open(cache_file, "rb") as f:
        all_paragraphs = pickle.load(f)

# Build a simple index for fast search (in-memory)
def search_paragraphs(term, top_n=3):
    term_lower = term.lower()
    results = [p for p in all_paragraphs if term_lower in p['paragraph'].lower()]
    results.sort(key=lambda x: x['paragraph'].lower().count(term_lower), reverse=True)
    return results[:top_n]

st.markdown('<div class="chat-header">🩺 MediGenie: Medical Q&A Chatbot</div>', unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input area at the bottom using a form
st.markdown("---")
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5,1])
    user_input = col1.text_input("Type your question and press Enter...", label_visibility="collapsed")
    send_clicked = col2.form_submit_button("Send", use_container_width=True)

    if send_clicked and user_input:
        with st.spinner("Thinking..."):
            term = extract_term_from_question(user_input)
            results = search_paragraphs(term, top_n=3)
            context_text = "\n\n".join([f"[PDF: {r.get('pdf','?')}] [Page: {r.get('page','?')}]\n{r['paragraph']}" for r in results])
            answer = context_text if context_text else "Not found in context."
            # Only call LLM if there is real context
            if results:
                llm = None
                try:
                    from llama_cpp import Llama
                    llm = Llama(model_path="models/Mistral-7B-Instruct-v0.3.Q3_K_L.gguf")
                except Exception:
                    pass
                llm_summary = None
                if llm:
                    def strict_summarize_with_llm(llm, context, question, max_tokens=48):
                        prompt_template = (
                            "Answer only using the provided context. "
                            "If the answer is not in the context, say 'Not found in context.'\n"
                            "Context:\n{context}\nQuestion: {question}"
                        )
                        # Use only the single most relevant paragraph, trimmed to 500 chars
                        if isinstance(context, list):
                            context = context[0] if context else ""
                        if len(context) > 500:
                            context = context[:500] + "..."
                        prompt = prompt_template.format(context=context, question=question)
                        response = llm(prompt, max_tokens=max_tokens)
                        return response['choices'][0]['text'].strip()
                    # Use only the most relevant paragraph for LLM
                    most_relevant = results[0]['paragraph'] if results else ""
                    llm_summary = strict_summarize_with_llm(llm, most_relevant, user_input, max_tokens=48)
                    if llm_summary and '<' not in llm_summary and '>' not in llm_summary:
                        answer = llm_summary + "\n\n" + context_text
            st.session_state['chat_history'].append((user_input, answer))

# Improved chat message container (move below the form)
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for q, a in st.session_state['chat_history']:
        st.markdown(f'<div class="chat-message user-message"><b>You:</b> {q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message bot-message"><b>MediGenie:</b> {a}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
