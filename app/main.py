from fastapi import FastAPI
from pydantic import BaseModel
from app.embeddings import Embeddings
from app.vectorstore import VectorStore
from app.retrieval import Retrieval
from app.llm import call_llm
from app.loader import DataLoader
import numpy as np

app = FastAPI(title="RAG FastAPI with Robust Loader")

# Load documents
pdf_docs = DataLoader.load_pdfs()
kb_docs = DataLoader.load_kb_articles()
all_docs = pdf_docs + kb_docs

if not all_docs:
    raise ValueError(
        "No documents loaded! Check your PDF folder and KB JSON files."
    )

print(f"Total documents loaded: {len(all_docs)}")

# Embeddings
emb_model = Embeddings()
embeddings = emb_model.encode(all_docs)

# Ensure embeddings are 2D numpy array
embeddings = np.array(embeddings)
if len(embeddings.shape) != 2:
    raise ValueError("Embeddings must be 2D. Check the documents loaded.")

print(f"Embeddings shape: {embeddings.shape}")

# Vector Store
vector_store = VectorStore(dimension=embeddings.shape[1])
vector_store.add_texts(all_docs, embeddings)

# Request model
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_rag(req: QueryRequest):
    query_emb = emb_model.encode([req.question])
    query_emb = np.array(query_emb)
    docs = vector_store.search(query_emb, top_k=3)
    prompt = Retrieval.build_prompt(docs, req.question)
    answer = call_llm(prompt)
    return {
        "question": req.question,
        "answer": answer,
        "sources": docs
    }
