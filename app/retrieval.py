class Retrieval:
    @staticmethod
    def build_prompt(docs, question):
        # context = "\n".join(docs)
        context = "\n".join([doc["text"] if isinstance(doc, dict) else doc for doc in docs])

        prompt = f"""
        Answer the question using ONLY the context below.
        If not found, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
        return prompt
