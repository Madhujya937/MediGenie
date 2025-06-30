from pypdf import PdfReader
import re
from collections import defaultdict
import difflib
import os
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import hashlib
import glob
import datetime

def extract_printed_page_number(text):
    # Look for a pattern like "GALE ENCYCLOPEDIA OF MEDICINE 37"
    match = re.search(r'GALE ENCYCLOPEDIA OF MEDICINE\s+(\d+)', text)
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
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        for para in paragraphs:
            para_clean = para.replace('\n', ' ').strip()
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
    results.sort(key=lambda x: (x['score'], len(x['paragraph'])), reverse=True)
    return results[:top_n]

def build_index(pdf_path):
    reader = PdfReader(pdf_path)
    index = defaultdict(list)
    total_pages = len(reader.pages)
    for page_num, page in enumerate(reader.pages, start=1):
        if page_num % 10 == 0 or page_num == 1 or page_num == total_pages:
            print(f"[Indexing] Processing page {page_num}/{total_pages}...")
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Index first non-empty line as possible heading
        if lines:
            first_line = lines[0].lower()
            if len(first_line) > 2:
                index[first_line].append(page_num)
        for i, line in enumerate(lines):
            # Index lines that look like headings:
            # - ALL CAPS
            # - Title Case (first letter uppercase, few words)
            # - Start with 'Definition'
            if (line.isupper() or
                (line.istitle() and len(line.split()) <= 6) or
                line.startswith('Definition')):
                key = line.lower()
                if len(key) > 2:
                    index[key].append(page_num)
            # Also index lines followed by a blank line (possible heading)
            if i+1 < len(lines) and lines[i+1] == '':
                key = line.lower()
                if len(key) > 2:
                    index[key].append(page_num)
    return index

def extract_term_from_question(question):
    match = re.search(r'what is ([\w\s-]+)', question.lower())
    if match:
        return match.group(1).strip()
    match = re.search(r'how (?:to|2) ([\w\s-]+)', question.lower())
    if match:
        return match.group(1).strip()
    return question.strip().lower()

def extract_top_pdf_paragraphs_fast(pdf_path, term, index, top_n=3):
    reader = PdfReader(pdf_path)
    term_lower = term.lower()
    candidate_pages = set()
    # Try direct match in index (case-insensitive, partial)
    for key in index:
        if term_lower in key:
            candidate_pages.update(index[key])
    # Try uppercase version (for headings in all caps)
    if not candidate_pages:
        term_upper = term.upper()
        for key in index:
            if term_upper in key.upper():
                candidate_pages.update(index[key])
    # Fallback: try last word if no match
    if not candidate_pages and " " in term_lower:
        fallback_term = term_lower.split()[-1]
        for key in index:
            if fallback_term in key:
                candidate_pages.update(index[key])
    # Fuzzy match: find closest heading if still not found
    if not candidate_pages:
        all_keys = list(index.keys())
        close_matches = difflib.get_close_matches(term_lower, all_keys, n=1, cutoff=0.7)
        if close_matches:
            candidate_pages.update(index[close_matches[0]])
    results = []
    for page_num in candidate_pages:
        text = reader.pages[page_num-1].extract_text() or ""
        printed_page = extract_printed_page_number(text) if 'extract_printed_page_number' in globals() else None
        page_label = f"Book Page {printed_page}" if printed_page else f"PDF Page {page_num}"
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        for para in paragraphs:
            para_clean = para.replace('\n', ' ').strip()
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
    results.sort(key=lambda x: (x['score'], len(x['paragraph'])), reverse=True)
    return results[:top_n]

def split_by_headings_with_pages(pdf_path):
    reader = PdfReader(pdf_path)
    sections = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        current_section = []
        heading = None
        for line in lines:
            if line.isupper() or (line.istitle() and len(line.split()) <= 6):
                if current_section:
                    sections.append({
                        'heading': heading or 'Unknown',
                        'page': page_num,
                        'content': '\n'.join(current_section)
                    })
                    current_section = []
                heading = line
            else:
                current_section.append(line)
        if current_section:
            sections.append({
                'heading': heading or 'Unknown',
                'page': page_num,
                'content': '\n'.join(current_section)
            })
    return sections

def hybrid_retrieve(pdf_path, question, vectorstore, index, top_n=5):
    # Keyword/heading search
    main_term = extract_term_from_question(question)
    keyword_results = extract_top_pdf_paragraphs_fast(pdf_path, main_term, index, top_n=top_n)
    # Vector search
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_emb = embeddings.embed_query(question)
    vector_results = vectorstore.similarity_search_by_vector(query_emb, k=top_n)
    # Combine, deduplicate by content
    seen = set()
    combined = []
    for para in keyword_results:
        if para['paragraph'] not in seen:
            combined.append({'source': 'keyword', **para})
            seen.add(para['paragraph'])
    for doc in vector_results:
        para = doc.page_content.strip()
        if para and para not in seen:
            combined.append({'source': 'vector', 'page': doc.metadata.get('page', '?'), 'paragraph': para, 'score': 0})
            seen.add(para)
    return combined[:top_n]

def summarize_with_llama(llm, context, question, max_tokens=24):
    prompt_template = (
        "[INST] You are a helpful medical assistant. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say 'Not found in context.'\n"
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Answer: [/INST]"
    )
    max_context_chars = 200
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."
    prompt = prompt_template.format(context=context, question=question)
    response = llm(prompt, max_tokens=max_tokens)
    return response['choices'][0]['text'].strip()

def get_pdf_hash(pdf_path):
    """Return a hash of the PDF file for cache validation."""
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
    # Build and cache index
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

def multi_pdf_hybrid_retrieve(pdf_paths, question, vectorstore, indexes, section_filter=None, top_n=5):
    all_results = []
    for pdf_path in pdf_paths:
        index = indexes[pdf_path]
        results = hybrid_retrieve(pdf_path, question, vectorstore, index, top_n=top_n*2)
        for r in results:
            r['pdf'] = os.path.basename(pdf_path)
            if section_filter:
                heading = r.get('paragraph', '').split('\n')[0]
                if section_filter.lower() not in heading.lower():
                    continue
            all_results.append(r)
    # Sort and deduplicate
    seen = set()
    deduped = []
    for r in sorted(all_results, key=lambda x: (x['score'], len(x['paragraph'])), reverse=True):
        if r['paragraph'] not in seen:
            deduped.append(r)
            seen.add(r['paragraph'])
        if len(deduped) >= top_n:
            break
    return deduped

def log_interaction(question, section, context, summary):
    with open("medigenie_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n---\nTime: {datetime.datetime.now()}\nQuestion: {question}\nSection: {section}\nContext: {context}\nSummary: {summary}\n")

def main():
    data_dir = r"d:\MEDIGenie\data"
    model_path = r"models/Mistral-7B-Instruct-v0.3.Q3_K_L.gguf"
    print("Scanning for PDFs...")
    pdf_paths = get_all_pdf_paths(data_dir)
    print(f"Found {len(pdf_paths)} PDFs. Building/loading indexes...")
    indexes = load_or_build_all_indexes(pdf_paths)
    print("[DEBUG] Loading local Llama model for summarization...")
    llm = Llama(model_path=model_path)
    print("[DEBUG] Loading vectorstore for semantic search...")
    vectorstore = FAISS.load_local(
        "vectorstore/db_faiss/",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    print("PDF Q&A is ready. Type your question or 'exit' to quit.")
    while True:
        user_question = input("question    ")
        if user_question.lower() in ["exit", "quit"]:
            break
        section = input("(Optional) Enter section to filter (e.g., Definition, Treatment), or press Enter to skip: ").strip()
        section_filter = section if section else None
        top_contexts = multi_pdf_hybrid_retrieve(pdf_paths, user_question, vectorstore, indexes, section_filter=section_filter, top_n=5)
        if not top_contexts:
            print("No relevant context found in the PDFs.\n")
            log_interaction(user_question, section_filter, "", "Not found in context.")
        else:
            combined_context = "\n".join([f"[PDF: {r['pdf']}] [Page: {r['page']}]: {r['paragraph']}" for r in top_contexts])
            print("\nPDF Context:")
            print(combined_context)
            try:
                summary = summarize_with_llama(llm, combined_context, user_question, max_tokens=64)
                print("\nLLM Summary:")
                print(summary)
                log_interaction(user_question, section_filter, combined_context, summary)
            except Exception as e:
                print(f"[LLM ERROR] {e}\nCould not generate LLM summary. Please check your model setup.")
                log_interaction(user_question, section_filter, combined_context, f"LLM ERROR: {e}")

if __name__ == "__main__":
    print("[DEBUG] Script started. Entering main().")
    try:
        main()
    except Exception as e:
        import traceback
        print("[ERROR] Exception in main():", e)
        traceback.print_exc()