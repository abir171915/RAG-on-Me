from pathlib import Path
from langchain_groq import ChatGroq
from search import Retriever
from query_rewriter import QueryRewriter


Prompt = """You answer question about Md Abir Hossain's professional background.
Use ONLY the context bellow which is enclosed by triple backticks based the question asked
which is also enclosed by triple backticks. If the answer is not in the context, say you 
don't know. If you manage to find the answer, please mention the source(s) you used.

Context: ```{context}```

Question: ```{question}```


Answer:"""

class RagGenerator:
    """Retrieve relevant chunks and ask the LLM to answer from them."""

    def __init__(self, retriever: Retriever, rewriter: QueryRewriter, model_name: str = "openai/gpt-oss-20b"):
        self.retriever = retriever
        self.rewriter = rewriter
        self.llm_model = ChatGroq(model = model_name, temperature = 0)

    def answer(self, question: str, top_k: int = 5) -> str:
        queries = self.rewriter.rewrite(question) + [question]
        hits = self.retriever.search_many(queries, top_k= top_k)
        context = "\n\n".join(
            f"[Source: {h['metadata'].get('repo') or Path(h['metadata'].get('source', '')).name}]\n{h['text']}"
            for h in hits
        )
        prompt = Prompt.format(context = context, question = question)
        return self.llm_model.invoke(prompt).content