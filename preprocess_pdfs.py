import os
import pickle
from create_memory_for_llm import get_all_pdf_paths, cache_all_pdf_texts, extract_all_pdf_paragraphs

def preprocess_and_cache(data_dir="data", cache_file="pdf_text_cache.pkl"):
    pdf_paths = get_all_pdf_paths(data_dir)
    print(f"Found {len(pdf_paths)} PDFs. Extracting and caching text...")
    cache_all_pdf_texts(pdf_paths)
    all_paragraphs = extract_all_pdf_paragraphs(pdf_paths)
    with open(cache_file, "wb") as f:
        pickle.dump(all_paragraphs, f)
    print(f"Cached {len(all_paragraphs)} paragraphs to {cache_file}.")

if __name__ == "__main__":
    preprocess_and_cache()
