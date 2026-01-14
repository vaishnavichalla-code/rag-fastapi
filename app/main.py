from fastapi import FastAPI
from pydantic import BaseModel
from app.embeddings import Embeddings
from app.vectorstore import VectorStore
from app.retrieval import Retrieval
from app.llm import call_llm

# Initialize
app = FastAPI()
emb_model = Embeddings()
texts = [
    "FastAPI is a web framework",
    "Dogs are friendly animals",
    "LLMs can generate text"
]
embeddings = emb_model.encode(texts)
vector_store = VectorStore(dimension=embeddings.shape[1])
vector_store.add_texts(texts, embeddings)

# Request Model
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_rag(req: QueryRequest):
    query_emb = emb_model.encode([req.question])
    docs = vector_store.search(query_emb, top_k=2)
    prompt = Retrieval.build_prompt(docs, req.question)
    answer = call_llm(prompt)
    return {
        "question": req.question,
        "answer": answer,
        "sources": docs
    }
