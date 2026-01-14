import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.embeddings import Embeddings
from app.vectorstore import VectorStore
from app.retrieval import Retrieval
from app.llm import call_llm
from app.loader import DataLoader

app = FastAPI()

# ---- Configure ServiceNow credentials ----
SERVICENOW_INSTANCE = os.getenv("SERVICENOW_INSTANCE")
SERVICENOW_USER = os.getenv("SERVICENOW_USER")
SERVICENOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD")
SERVICENOW_QUERY = os.getenv("SERVICENOW_QUERY", "workflow_state=published")  # default if not set

# ---- Initialize embeddings & vector store ----
emb_model = Embeddings()
dimension = emb_model.model.get_sentence_embedding_dimension()
vector_store = VectorStore(dimension=dimension)

# ---- Load or build vector DB ----
if os.path.exists(vector_store.persist_path) and os.path.exists(vector_store.meta_path):
    print("Loading existing vector DB...")
    vector_store.load()
else:
    print("Building vector DB from PDFs + ServiceNow KB...")

    pdf_docs = DataLoader.load_pdfs()
    try:
        sn_docs = DataLoader.load_servicenow_kb(
            SERVICENOW_INSTANCE, SERVICENOW_USER, SERVICENOW_PASSWORD
        )
    except Exception as e:
        print(f"ServiceNow KB load failed: {e}")
        sn_docs = []

    all_docs = pdf_docs + sn_docs
    embeddings = emb_model.encode([d["text"] for d in all_docs])
    vector_store.add_texts(all_docs, embeddings)
    vector_store.save()
    print(f"Vector DB built and saved with {len(all_docs)} docs.")

# ---- FastAPI endpoint ----
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
