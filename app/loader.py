import os
import json
import base64
import requests
from pdfminer.high_level import extract_text
from urllib.parse import urlparse, parse_qs, unquote


class DataLoader:

    # ---------------- TEXT CHUNKING ----------------
    @staticmethod
    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ---------------- PDF LOADER ----------------
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
                        "page": i + 1
                    })
        return docs

    # ---------------- LOCAL KB JSON ----------------
    @staticmethod
    def load_local_kb(kb_folder="data/kb_articles"):
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
                                "chunk": i + 1
                            })
        return docs

    # ---------------- SERVICENOW QUERY EXTRACTOR ----------------
    @staticmethod
    def extract_sysparm_query(input_str: str) -> str:
        """
        Accepts:
        - Raw query
        - Encoded query
        - Full ServiceNow UI URL
        Returns clean sysparm_query
        """
        # If full URL pasted
        if "sysparm_query" in input_str:
            decoded = unquote(input_str)
            parsed = urlparse(decoded)
            params = parse_qs(parsed.query)
            query = params.get("sysparm_query", [""])[0]
        else:
            query = input_str

        # Decode repeatedly until stable
        while "%3D" in query or "%253D" in query:
            query = unquote(query)

        return query

    # ---------------- SERVICENOW KB LOADER ----------------
    @staticmethod
    def load_servicenow_kb(
        instance_url: str,
        user: str,
        password: str,
        query: str = "workflow_state=published",
        limit: int = 1000
    ):
        url = f"{instance_url.rstrip('/')}/api/now/table/kb_knowledge"

        # 🔥 FIX: extract + decode query safely
        query = DataLoader.extract_sysparm_query(query)

        auth_str = f"{user}:{password}"
        encoded_auth = base64.b64encode(auth_str.encode()).decode()

        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {encoded_auth}"
        }

        params = {
            "sysparm_query": query,
            "sysparm_fields": "sys_id,short_description,text",
            "sysparm_limit": limit
        }

        response = requests.get(url, headers=headers, params=params, timeout=30)

        print("Final sysparm_query:", query)
        print("ServiceNow status:", response.status_code)

        if response.status_code != 200:
            raise Exception(f"ServiceNow API error {response.status_code}: {response.text}")

        result = response.json().get("result", [])

        docs = []
        for r in result:
            text = f"{r.get('short_description','')}\n{r.get('text','')}".strip()
            if not text:
                continue

            docs.append({
                "text": text,
                "source": "ServiceNow",
                "sys_id": r["sys_id"],
                "title": r.get("short_description")
            })

        print(f"Loaded {len(docs)} ServiceNow KB articles")
        return docs
