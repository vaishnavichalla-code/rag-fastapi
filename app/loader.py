import os
import requests
from pdfminer.high_level import extract_text
import json
import base64

class DataLoader:
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

    # ---------------- PDF Loading ----------------
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

    # ---------------- Local KB JSON (Optional Backup) ----------------
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
                                "chunk": i+1
                            })
        return docs

    # ---------------- ServiceNow KB API ----------------
    @staticmethod
   

    def load_servicenow_kb(instance_url, user, password):
        url = f"{instance_url}/api/now/table/kb_knowledge"
        auth_str = f"{user}:{password}"
        encoded_auth = base64.b64encode(auth_str.encode()).decode()
        headers = {
        "Accept": "application/json",
        "Authorization": f"Basic {encoded_auth}"
            }
        print("headers:", headers)
        params = {
            "sysparm_query": "workflow_state=published",
            "sysparm_fields": "sys_id,short_description,text",
            "sysparm_limit": 1000
        }

        response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
        )

        print("ServiceNow Status:", response.status_code)

        if response.status_code != 200:
            print("ServiceNow Raw Response:", response.text)
            raise Exception(f"ServiceNow API error: {response.status_code}")

        result = response.json()["result"]

        docs = []
        for r in result:
            docs.append({
                "text": f"{r.get('short_description','')}\n{r.get('text','')}",
                "source": "ServiceNow",
                "sys_id": r["sys_id"],
                "title": r.get("short_description")
            })

        print(f"Loaded {len(docs)} ServiceNow KB articles")
        return docs

