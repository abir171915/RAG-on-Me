from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import SecretStr
from langchain_groq import ChatGroq
from rag_on_me.search import Retriever
from rag_on_me.query_rewriter import QueryRewriter


Prompt = """You answer question about Md Abir Hossain's professional background.
Use ONLY the context bellow which is enclosed by triple backticks based the question asked
which is also enclosed by triple backticks. If the answer is not in the context, say you 
don't know. If you manage to find the answer, please mention the source(s) you used.

Context: ```{context}```

Question: ```{question}```


Answer:"""

class RagGenerator:
    """Retrieve relevant chunks and ask the LLM to answer from them."""

    def __init__(self, retriever: Retriever, rewriter: QueryRewriter, model_name: str = "openai/gpt-oss-20b",
                 api_key: Optional[SecretStr] = None):
        self.retriever = retriever
        self.rewriter = rewriter
        self.llm_model = ChatGroq(model = model_name, temperature = 0, api_key = api_key)

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        queries = self.rewriter.rewrite(question)
        hits = self.retriever.search_many(queries, top_k= top_k)
        sources = [h['metadata'].get('source_name', 'unknown') for h in hits]
        context = "\n\n".join(
            f"[Source: {source}]\n{h['text']}"
            for source, h in zip(sources, hits)
        )
        prompt = Prompt.format(context = context, question = question)
        return {
            "answer": self.llm_model.invoke(prompt).content,
            "sources": list(dict.fromkeys(sources)),
        }