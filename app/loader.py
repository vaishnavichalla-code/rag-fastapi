import os
from pdfminer.high_level import extract_text
import json

class DataLoader:
    @staticmethod
    def load_pdfs(pdf_folder="data/pdfs"):
        if not os.path.exists(pdf_folder):
            print(f"PDF folder not found: {pdf_folder}")
            return []

        docs = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                path = os.path.join(pdf_folder, filename)
                try:
                    text = extract_text(path)
                    if text.strip():  # only add non-empty text
                        docs.append(text)
                    else:
                        print(f"PDF empty: {filename}")
                except Exception as e:
                    print(f"Failed to read {filename}: {e}")
        return docs

    @staticmethod
    def load_kb_articles(kb_folder="data/kb_articles"):
        if not os.path.exists(kb_folder):
            print(f"KB folder not found: {kb_folder}")
            return []

        docs = []
        for filename in os.listdir(kb_folder):
            if filename.endswith(".json"):
                path = os.path.join(kb_folder, filename)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        for article in data:
                            content = article.get("content", "").strip()
                            if content:
                                docs.append(content)
                            else:
                                print(f"Empty content in {filename}")
                except Exception as e:
                    print(f"Failed to read {filename}: {e}")
        return docs
