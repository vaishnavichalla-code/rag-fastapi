import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.embeddings import Embeddings
from app.vectorstore import VectorStore
from app.retrieval import Retrieval
from app.llm import call_llm
from app.loader import DataLoader

app = FastAPI()

# Initialize embeddings
emb_model = Embeddings()
dimension = emb_model.model.get_sentence_embedding_dimension()

# Initialize vector store with persistence paths
vector_store = VectorStore(dimension=dimension)

# Load persistent index if exists, else build it
if os.path.exists(vector_store.persist_path) and os.path.exists(vector_store.meta_path):
    vector_store.load()
else:
    pdf_docs = DataLoader.load_pdfs()
    kb_docs = DataLoader.load_kb_articles()
    all_docs = pdf_docs + kb_docs
    embeddings = emb_model.encode([d["text"] for d in all_docs])
    vector_store.add_texts(all_docs, embeddings)
    vector_store.save()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_rag(req: QueryRequest):
    query_emb = emb_model.encode([req.question])
    docs = vector_store.search(query_emb, top_k=3)
    prompt = Retrieval.build_prompt(docs, req.question)
    answer = call_llm(prompt)
    return {
        "question": req.question,
        "answer": answer,
        "sources": docs
    }
