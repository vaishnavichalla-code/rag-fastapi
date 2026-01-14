import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add_texts(self, texts, embeddings):
        self.texts.extend(texts)
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, top_k=2):
        _, indices = self.index.search(np.array(query_embedding), top_k)
        return [self.texts[i] for i in indices[0]]
