import warnings
warnings.filterwarnings("ignore")
print("=== MediGenie vectorstore script started ===")
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from llama_cpp import Llama
import re
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)

load_dotenv()

# Path to your PDF
DOC_PATH = os.path.join("data", "The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf")
if not os.path.exists(DOC_PATH):
    print(f"ERROR: PDF file not found at {DOC_PATH}")
    exit(1)

# --- Direct text extraction only ---
all_text = []
with pdfplumber.open(DOC_PATH) as pdf:
    for i, page in enumerate(pdf.pages):  # Process all pages
        text = page.extract_text()
        if text and text.strip():
            all_text.append(text)
        else:
            all_text.append("")

full_text = "\n".join(all_text)
if full_text.strip():
    print(f"[PDF] Extracted {len(full_text)} characters from PDF using direct text extraction. {len(all_text)} pages with text.")
else:
    print("ERROR: No text found in PDF. Please provide a text-based PDF (not scanned images). Exiting.")
    exit(1)

# Section-based chunking (by headings)
def split_by_headings_with_pages(all_text):
    section_chunks = []
    heading = None
    buffer = []
    page_num = 1
    for page_text in all_text:
        lines = page_text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if (line_stripped.isupper() or (line_stripped.istitle() and len(line_stripped.split()) <= 6)) and len(line_stripped) > 2:
                if buffer and heading:
                    section_chunks.append({
                        'heading': heading,
                        'text': '\n'.join(buffer).strip(),
                        'page': page_num
                    })
                    buffer = []
                heading = line_stripped
            else:
                buffer.append(line)
        if buffer and heading:
            section_chunks.append({
                'heading': heading,
                'text': '\n'.join(buffer).strip(),
                'page': page_num
            })
            buffer = []
        page_num += 1
    return section_chunks

section_chunks = split_by_headings_with_pages(all_text)
chunks = [c['text'] for c in section_chunks]
pages = [c['page'] for c in section_chunks]
headings = [c['heading'] for c in section_chunks]
print(f"Split into {len(chunks)} section-based chunks (with headings and page numbers). Example heading: {headings[0] if headings else 'None'}")
if chunks:
    print(f"First chunk (preview): {chunks[0][:300]}")
else:
    print("ERROR: No chunks created. Check PDF extraction and chunking logic.")

# 3. Embed and add to FAISS vector store using Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=[{"page": p, "heading": h} for p, h in zip(pages, headings)]
)
vectorstore.save_local("vectorstore/db_faiss/")
if os.path.exists("vectorstore/db_faiss/index.faiss") and os.path.exists("vectorstore/db_faiss/index.pkl"):
    print("SUCCESS: MediGenie vector store created and saved with Gale Encyclopedia data using Hugging Face embeddings!")
else:
    print("ERROR: Vectorstore files not found after saving. Check for errors above.")

LOGICAL_PAGE_OFFSET = -12  # Adjust this value based on your PDF (e.g., if PDF page 13 is printed as 1, use -12)

def get_logical_page_number(text):
    # Try to find a page number at the bottom of the page (last line)
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        match = re.match(r'^(\d{1,4})$', last_line)
        if match:
            return int(match.group(1))
    return None

def extract_printed_page_number(paragraph):
    # Looks for "GALE ENCYCLOPEDIA OF MEDICINE <number>"
    match = re.search(r'GALE ENCYCLOPEDIA OF MEDICINE\s+(\d+)', paragraph)
    if match:
        return match.group(1)
    return None

def extract_top_pdf_paragraphs(pdf_path, term, top_n=5):
    reader = PdfReader(pdf_path)
    term_lower = term.lower()
    results = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        printed_page = extract_printed_page_number(text)
        page_label = f"Book Page {printed_page}" if printed_page else f"PDF Page {page_num}"
        paragraphs = re.split(r'\\n\\s*\\n|\\n{2,}', text)
        for para in paragraphs:
            para_clean = para.replace('\\n', ' ').strip()
            if (term_lower in para_clean.lower() or
                para_clean.lower().startswith(term_lower + ':') or
                para_clean.lower().startswith(term_lower + ' ')):
                score = para_clean.lower().count(term_lower)
                if para_clean.lower().startswith(term_lower):
                    score += 2
                results.append({
                    'page': page_label,
                    'paragraph': para_clean,
                    'score': score
                })
    if not results and term_lower == 'prognosis':
        for page_num, page in enumerate(reader.pages, start=1):
            logical_page = page_num + LOGICAL_PAGE_OFFSET
            page_label = f"Page {logical_page}" if logical_page > 0 else f"PDF Page {page_num}"
            text = page.extract_text() or ""
            paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
            for para in paragraphs:
                para_clean = para.replace('\n', ' ').strip()
                if 'prognosis' in para_clean.lower():
                    score = para_clean.lower().count('prognosis')
                    results.append({
                        'page': page_label,
                        'paragraph': para_clean,
                        'score': score
                    })
    results.sort(key=lambda x: (x['score'], len(x['paragraph'])), reverse=True)
    return results[:top_n]

def extract_main_definition(pdf_path, term):
    reader = PdfReader(pdf_path)
    term_lower = term.lower()
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if term_lower in text.lower():
            # Look for the 'Definition' section after the term
            match = re.search(r'Definition\s+(.*?)(?=\n[A-Z][a-z]+:|\n[A-Z][a-z]+\s|Purpose|Description|$)', text, re.DOTALL)
            if match:
                definition = match.group(1).replace('\n', ' ').strip()
                # Try to extract printed page number
                page_match = re.search(r'GALE ENCYCLOPEDIA OF MEDICINE\s+(\d+)', text)
                page_label = f"Book Page {page_match.group(1)}" if page_match else f"PDF Page {page_num}"
                return f"{page_label}: {definition}"
    return "Not found in context."

def search_with_section_filter(vectorstore, query, section_keyword=None, top_k=5):
    # Embed the query
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_emb = embeddings.embed_query(query)
    # Search vectorstore
    results = vectorstore.similarity_search_by_vector(query_emb, k=top_k)
    # Filter by section if requested
    if section_keyword:
        filtered = [r for r in results if section_keyword.lower() in (r.metadata.get('heading') or '').lower()]
        if filtered:
            results = filtered
    for i, r in enumerate(results, 1):
        heading = r.metadata.get('heading', '?')
        page = r.metadata.get('page', '?')
        print(f"ans {i}: [Section: {heading}] [PDF Page: {page}]\n{r.page_content}\n")

def hybrid_search(vectorstore, query, section_keyword=None, top_k=5):
    # Semantic search
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_emb = embeddings.embed_query(query)
    semantic_results = vectorstore.similarity_search_by_vector(query_emb, k=top_k)
    # Keyword search (in text and heading)
    keyword_results = []
    for doc in vectorstore.similarity_search(query, k=top_k*2):
        if (query.lower() in doc.page_content.lower() or
            (doc.metadata.get('heading') and query.lower() in doc.metadata['heading'].lower())):
            keyword_results.append(doc)
    # Combine and deduplicate (by page_content)
    seen = set()
    combined = []
    for r in keyword_results + semantic_results:
        if r.page_content not in seen:
            if not section_keyword or (section_keyword.lower() in (r.metadata.get('heading') or '').lower()):
                combined.append(r)
                seen.add(r.page_content)
        if len(combined) >= top_k:
            break
    for i, r in enumerate(combined, 1):
        heading = r.metadata.get('heading', '?')
        page = r.metadata.get('page', '?')
        print(f"ans {i}: [Section: {heading}] [PDF Page: {page}]\n{r.page_content}\n")

def summarize_with_llama(llm, context, question, max_tokens=24):
    prompt_template = (
        "Answer only using the provided context. "
        "If the answer is not in the context, say 'Not found in context.'\n"
        "Context:\n{context}\nQuestion: {question}"
    )
    max_context_chars = 200
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."
    prompt = prompt_template.format(context=context, question=question)
    response = llm(prompt, max_tokens=max_tokens)
    return response['choices'][0]['text'].strip()

def main():
    print("[DEBUG] main() function entered.")
    pdf_path = r"d:\MEDIGenie\data\The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
    model_path = r"models/Mistral-7B-Instruct-v0.3.Q3_K_L.gguf"
    print("[INFO] Starting MediGenie setup...")
    # Load vectorstore
    try:
        print("[INFO] Loading vectorstore from vectorstore/db_faiss/ ...")
        vectorstore = FAISS.load_local(
            "vectorstore/db_faiss/",
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
        print("[INFO] Vectorstore loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load vectorstore: {e}")
        return
    # Load Llama model
    try:
        print(f"[INFO] Loading Llama model from {model_path} ... (this may take a while)")
        llm = Llama(model_path=model_path)
        print("[INFO] Llama model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Llama model: {e}")
        return
    print("[INFO] MediGenie (local Llama + hybrid search) is ready. Type your question or 'exit' to quit.")
    while True:
        user_question = input("You: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        # Optional: let user specify section, e.g., 'Definition' or 'Treatment'
        section_keyword = None
        if any(word in user_question.lower() for word in ["definition", "define"]):
            section_keyword = "Definition"
        elif any(word in user_question.lower() for word in ["treatment", "treat"]):
            section_keyword = "Treatment"
        # Hybrid search with section filtering
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_emb = embeddings.embed_query(user_question)
        semantic_results = vectorstore.similarity_search_by_vector(query_emb, k=5)
        keyword_results = []
        for doc in vectorstore.similarity_search(user_question, k=10):
            if (user_question.lower() in doc.page_content.lower() or
                (doc.metadata.get('heading') and user_question.lower() in doc.metadata['heading'].lower())):
                keyword_results.append(doc)
        seen = set()
        combined = []
        for r in keyword_results + semantic_results:
            if r.page_content not in seen:
                if not section_keyword or (section_keyword.lower() in (r.metadata.get('heading') or '').lower()):
                    combined.append(r)
                    seen.add(r.page_content)
            if len(combined) >= 5:
                break
        if not combined:
            print("ans  Not found in context.\n")
            continue
        context = "\n\n".join([f"[Section: {r.metadata.get('heading', '?')}] [PDF Page: {r.metadata.get('page', '?')}]\n{r.page_content}" for r in combined])
        answer = summarize_with_llama(llm, context, user_question)
        print("MediGenie says:", answer)

if __name__ == "__main__":
    main()