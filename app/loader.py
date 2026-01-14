import os
from pdfminer.high_level import extract_text
import json

class DataLoader:
    @staticmethod
    def chunk_text(text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def load_pdfs(pdf_folder="data/pdfs"):
        docs = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                path = os.path.join(pdf_folder, filename)
                text = extract_text(path)
                for i, chunk in enumerate(DataLoader.chunk_text(text)):
                    docs.append({
                        "text": chunk,
                        "source": filename,
                        "page": i+1
                    })
        return docs

    @staticmethod
    def load_kb_articles(kb_folder="data/kb_articles"):
        docs = []
        for filename in os.listdir(kb_folder):
            if filename.endswith(".json"):
                path = os.path.join(kb_folder, filename)
                with open(path, "r") as f:
                    data = json.load(f)
                    for article in data:
                        content = article.get("content", "")
                        title = article.get("title", "")
                        for i, chunk in enumerate(DataLoader.chunk_text(content)):
                            docs.append({
                                "text": chunk,
                                "title": title,
                                "source": filename,
                                "chunk": i+1
                            })
        return docs
