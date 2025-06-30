from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from pypdf import PdfReader
import re
from collections import defaultdict
import pickle
import hashlib
import glob

# Remove sample medical knowledge and use real PDF content for all context and vectorstore

def extract_all_pdf_paragraphs(pdf_paths):
    """Extract all paragraphs from all PDFs for indexing/vectorstore."""
    all_paragraphs = []
    for pdf_path in pdf_paths:
        texts = pdf_text_cache.get(pdf_path)
        if not texts:
            reader = PdfReader(pdf_path)
            texts = [page.extract_text() or "" for page in reader.pages]
            pdf_text_cache[pdf_path] = texts
        for page_num, text in enumerate(texts, start=1):
            paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
            for para in paragraphs:
                para_clean = para.replace('\n', ' ').strip()
                if para_clean:
                    all_paragraphs.append({
                        'pdf': os.path.basename(pdf_path),
                        'page': page_num,
                        'paragraph': para_clean
                    })
    return all_paragraphs

vectorstore = None  # Global variable for vectorstore

def retrieve_from_vectorstore(question, k=1):
    global vectorstore
    if vectorstore is None:
        raise RuntimeError("Vectorstore not loaded.")
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=k)
    return "\n".join([doc.page_content for doc, _ in docs_and_scores])

def get_most_relevant_context(question):
    # Retrieve the top document or passage as before
    long_context = retrieve_from_vectorstore(question)
    # Split into lines or Q&A pairs and find the most relevant one
    for line in long_context.split('\n'):
        if question.lower() in line.lower():
            return line
    # Fallback: return the first relevant sentence
    return long_context.split('\n')[0]

def highlight_keywords(text, keywords):
    # Highlight all keywords in the text (case-insensitive)
    def replacer(match):
        return f"**{match.group(0)}**"
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(replacer, text)
    return text

# Global cache for PDF texts
pdf_text_cache = {}

def cache_all_pdf_texts(pdf_paths):
    for pdf_path in pdf_paths:
        if pdf_path not in pdf_text_cache:
            reader = PdfReader(pdf_path)
            pdf_text_cache[pdf_path] = [page.extract_text() or "" for page in reader.pages]

def extract_top_pdf_paragraphs(pdf_path, term, top_n=10, filter_term=None):
    texts = pdf_text_cache.get(pdf_path)
    if not texts:
        reader = PdfReader(pdf_path)
        texts = [page.extract_text() or "" for page in reader.pages]
        pdf_text_cache[pdf_path] = texts
    term_lower = term.lower()
    filter_lower = filter_term.lower() if filter_term else None
    results = []
    for page_num, text in enumerate(texts, start=1):
        heading = None
        for line in text.split('\n'):
            if line.isupper() or (line.istitle() and len(line.split()) <= 6):
                heading = line.strip()
                break
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        for para in paragraphs:
            para_clean = para.replace('\n', ' ').strip()
            # Only include paragraphs that contain the term (strict match)
            if term_lower in para_clean.lower():
                if filter_lower and filter_lower not in para_clean.lower():
                    continue
                # Score: count of term occurrences
                score = para_clean.lower().count(term_lower)
                if filter_lower:
                    score += para_clean.lower().count(filter_lower) * 2  # Boost score for filter term
                results.append({
                    'page': page_num,
                    'paragraph': para_clean,
                    'score': score,
                    'heading': heading or 'Unknown'
                })
    # Sort by score (descending), then by paragraph length (descending)
    results.sort(key=lambda x: (x['score'], len(x['paragraph'])), reverse=True)
    if not results:
        print(f"[DEBUG] No paragraphs found for '{term}' in {os.path.basename(pdf_path)}")
    return results[:top_n]

def build_index(pdf_path):
    reader = PdfReader(pdf_path)
    index = defaultdict(list)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # Index all words in headings and 'Definition' lines
        for line in text.split('\n'):
            # Index lines that look like headings or start with 'Definition'
            if line.isupper() or line.strip().startswith('Definition'):
                words = re.findall(r'\w+', line.lower())
                for word in words:
                    if len(word) > 2:  # skip very short words
                        index[word].append(page_num)
    return index

def extract_term_from_question(question):
    match = re.search(r'what is ([\w\s-]+)', question.lower())
    if match:
        return match.group(1).strip()
    match = re.search(r'how (?:to|2) ([\w\s-]+)', question.lower())
    if match:
        return match.group(1).strip()
    return question.strip().lower()

def extract_main_definition(pdf_path, question, index=None):
    term = extract_term_from_question(question)
    reader = PdfReader(pdf_path)
    # Use index to get candidate pages
    candidate_pages = set()
    if index and term:
        for word in term.split():
            if word in index:
                candidate_pages.update(index[word])
    if not candidate_pages:
        candidate_pages = range(1, len(reader.pages)+1)
    for page_num in candidate_pages:
        page = reader.pages[page_num-1]
        text = page.extract_text() or ""
        if term in text.lower():
            match = re.search(r'Definition\s+(.*?)(?=\n[A-Z][a-z]+:|\n[A-Z][a-z]+\s|Purpose|Description|$)', text, re.DOTALL)
            if match:
                definition = match.group(1).replace('\n', ' ').strip()
                page_match = re.search(r'GALE ENCYCLOPEDIA OF MEDICINE\s+(\d+)', text)
                page_label = f"Book Page {page_match.group(1)}" if page_match else f"PDF Page {page_num}"
                return f"{page_label}: {definition}"
            heading_match = re.search(rf'{re.escape(term)}\s*\n(.*?)(?=\n[A-Z][a-z]+:|\n[A-Z][a-z]+\s|$)', text, re.IGNORECASE | re.DOTALL)
            if heading_match:
                paragraph = heading_match.group(1).replace('\n', ' ').strip()
                page_match = re.search(r'GALE ENCYCLOPEDIA OF MEDICINE\s+(\d+)', text)
                page_label = f"Book Page {page_match.group(1)}" if page_match else f"PDF Page {page_num}"
                return f"{page_label}: {paragraph}"
    return "Not found in context."

def extract_definition_section(pdf_path, term):
    texts = pdf_text_cache.get(pdf_path)
    if not texts:
        reader = PdfReader(pdf_path)
        texts = [page.extract_text() or "" for page in reader.pages]
        pdf_text_cache[pdf_path] = texts
    term_lower = term.lower()
    for page_num, text in enumerate(texts, start=1):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            # Look for the term and 'definition' in the same or adjacent lines
            if (term_lower in line.lower() and 'definition' in line.lower()) or \
               (term_lower in line.lower() and i+1 < len(lines) and 'definition' in lines[i+1].lower()):
                # Start collecting lines after the definition heading
                def_start = i+1 if 'definition' in line.lower() else i+2
                definition_lines = []
                for j in range(def_start, len(lines)):
                    # Stop at next ALL CAPS heading, Title Case heading, or empty line
                    if lines[j].isupper() or (lines[j].istitle() and len(lines[j].split()) <= 6):
                        break
                    if not lines[j].strip():
                        break
                    definition_lines.append(lines[j])
                if definition_lines:
                    print(f"[DEBUG] Matched definition for '{term}' on page {page_num} in {os.path.basename(pdf_path)}")
                    return {'page': page_num, 'heading': 'Definition', 'paragraph': ' '.join(definition_lines)}
    print(f"[DEBUG] No definition found for '{term}' in {os.path.basename(pdf_path)}")
    return None

def summarize_with_llm(llm, context, question, max_tokens=24):
    prompt_template = (
        "[INST] You are a helpful medical assistant. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say 'Not found in context.'\n"
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Answer: [/INST]"
    )
    # Further reduce context to fit within LLM context window (e.g., 200 chars for prompt, 24 tokens output)
    max_context_chars = 200
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."
    prompt = prompt_template.format(context=context, question=question)
    response = llm(prompt, max_tokens=max_tokens)
    return response['choices'][0]['text'].strip()

def get_pdf_hash(pdf_path):
    h = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_or_build_index(pdf_path, cache_path="index_cache.pkl"):
    pdf_hash = get_pdf_hash(pdf_path)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if cache.get('pdf_hash') == pdf_hash:
                print("[Index] Loaded index from cache.")
                return cache['index']
            else:
                print("[Index] PDF changed, rebuilding index...")
        except Exception as e:
            print(f"[Index] Cache load failed: {e}. Rebuilding index...")
    index = build_index(pdf_path)
    with open(cache_path, 'wb') as f:
        pickle.dump({'pdf_hash': pdf_hash, 'index': index}, f)
    print("[Index] Index built and cached.")
    return index

def get_all_pdf_paths(data_dir="data"):
    return glob.glob(os.path.join(data_dir, "*.pdf"))

def load_or_build_all_indexes(pdf_paths):
    indexes = {}
    for pdf_path in pdf_paths:
        cache_path = os.path.splitext(pdf_path)[0] + "_index_cache.pkl"
        indexes[pdf_path] = load_or_build_index(pdf_path, cache_path)
    return indexes

def multi_pdf_extract_top_paragraphs(pdf_paths, term, top_n=10, filter_term=None):
    all_results = []
    for pdf_path in pdf_paths:
        results = extract_top_pdf_paragraphs(pdf_path, term, top_n=top_n, filter_term=filter_term)
        for r in results:
            r['pdf'] = os.path.basename(pdf_path)
        all_results.extend(results)
    all_results.sort(key=lambda x: (x['score'], len(x['paragraph'])), reverse=True)
    return all_results[:top_n]

def multi_pdf_extract_definition_section(pdf_paths, term):
    for pdf_path in pdf_paths:
        definition = extract_definition_section(pdf_path, term)
        if definition:
            definition['pdf'] = os.path.basename(pdf_path)
            return definition
    return None

def build_vectorstore_from_pdfs(pdf_paths, embeddings):
    all_paragraphs = extract_all_pdf_paragraphs(pdf_paths)
    texts = [p['paragraph'] for p in all_paragraphs]
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local("vectorstore/db_faiss/")
    print(f"MediGenie vector store created and saved from {len(texts)} PDF paragraphs!")
    return vectorstore

# Removed the CLI main() function and the if __name__ == '__main__': main() block