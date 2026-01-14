class Retrieval:
    @staticmethod
    def build_prompt(docs, question):
        context = "\n".join(docs)
        return f"""
Answer the question using ONLY the context below.
If the answer is not found in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
