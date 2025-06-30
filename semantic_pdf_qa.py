from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

PDF_PATH = "data/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
VECTORSTORE_DIR = "vectorstore/chroma"
COLLECTION_NAME = "pdf_paragraphs"

# Extract paragraphs from PDF

def extract_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    paragraphs = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for para in text.split('\n\n'):
            para = para.strip()
            if len(para) > 30:
                paragraphs.append(para)
    return paragraphs

# Build vectorstore if not exists

def build_vectorstore(pdf_path, persist_dir=VECTORSTORE_DIR):
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        print("Vectorstore already exists. Skipping embedding.")
        return
    print("Extracting paragraphs and building vectorstore...")
    paragraphs = extract_paragraphs(pdf_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(paragraphs, show_progress_bar=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    for i, (para, emb) in enumerate(zip(paragraphs, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[emb.tolist()],
            documents=[para]
        )
    print(f"Stored {len(paragraphs)} paragraphs in vectorstore.")

# Semantic search

def semantic_search(query, persist_dir=VECTORSTORE_DIR, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode([query])[0]
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k
    )
    for i, doc in enumerate(results['documents'][0]):
        print(f"ans {i+1}: {doc}\n")

if __name__ == "__main__":
    build_vectorstore(PDF_PATH)
    print("Semantic PDF Q&A ready. Type your question or 'exit' to quit.")
    while True:
        user_question = input("question    ")
        if user_question.lower() in ["exit", "quit"]:
            break
        semantic_search(user_question)
