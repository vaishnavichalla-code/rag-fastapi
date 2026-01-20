import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.embeddings import Embeddings
from app.vectorstore import VectorStore
from app.retrieval import Retrieval
from app.llm import call_llm
from app.loader import DataLoader

app = FastAPI()

# ---- Init models once ----
emb_model = Embeddings()
vector_store = VectorStore(
    dimension=emb_model.model.get_sentence_embedding_dimension()
)

# ---- Load vector DB at startup ----
@app.on_event("startup")
def load_vector_db():
    if os.path.exists(vector_store.persist_path):
        print("Loading vector DB...")
        vector_store.load()
    else:
        print("Vector DB not found. Start empty.")

# ---------------- ASK API ----------------
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

# ---------------- INGEST API ----------------
class SNRequest(BaseModel):
    instance_url: str
    username: str
    password: str
    query: str = "workflow_state=published"

@app.post("/ingest/servicenow")
def ingest_servicenow(req: SNRequest):

    # 1. Load ServiceNow docs
    # docs = DataLoader.load_servicenow_kb(
    #     instance_url=req.instance_url,
    #     user=req.username,
    #     password=req.password,
    #     query=req.query
    # )
    docs = DataLoader.load_servicenow_kb(
    instance_url=req.instance_url.rstrip("/"),
    user=req.username,
    password=req.password,
    query=req.query.replace("%3D", "=")
    )

    print(f"instance_url: {req.instance_url}")
    print(f"user: {req.username}")
    print(f"password: {req.password}")
    print(f"query: {req.query}")
    print(f"Loaded {len(docs)} documents from ServiceNow.")
    # 2. Filter empty docs (MANDATORY)
    docs = [d for d in docs if d.get("text") and d["text"].strip()]

    if not docs:
        return {"status": "no_valid_docs"}

    # 3. Generate embeddings (ALWAYS list)
    texts = [d["text"] for d in docs]
    embeddings = emb_model.encode(texts)

    # 4. Ensure 2D shape for FAISS
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # 5. Store
    vector_store.add_texts(docs, embeddings)
    vector_store.save()

    return {
        "status": "success",
        "articles_loaded": len(docs)
    }
