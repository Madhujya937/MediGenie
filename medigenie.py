from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os

app = FastAPI(title="MediGenie AI Medical Chatbot")

class ChatRequest(BaseModel):
    query: str

# Load OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Load vector store and LLM once at startup
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.load_local(
    "vectorstore/db_faiss/",
    SentenceTransformer(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Initialize the language model (LLM)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key)

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    try:
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=2)
        context = " ".join([doc.page_content for doc, _ in docs_and_scores])
        prompt = f"Based on this context: {context}\nAnswer: {query}"
        response = llm(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("This script is deprecated. Use medigenie_app.py and create_vectorstore_from_pdf.py for all Q&A and vectorstore tasks.")