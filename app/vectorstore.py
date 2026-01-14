import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dimension, persist_path=None, meta_path=None):
        self.dimension = dimension
        self.persist_path = persist_path or "vector_db/faiss.index"
        self.meta_path = meta_path or "vector_db/meta.pkl"
        self.index = faiss.IndexFlatL2(dimension)
        self.meta = []
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)

    def add_texts(self, docs, embeddings):
        self.meta.extend(docs)
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(np.array(query_embedding).astype("float32"), top_k)
        return [self.meta[i] for i in indices[0]]

    def save(self):
        faiss.write_index(self.index, self.persist_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)

    def load(self):
        if os.path.exists(self.persist_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.persist_path)
            with open(self.meta_path, "rb") as f:
                self.meta = pickle.load(f)
