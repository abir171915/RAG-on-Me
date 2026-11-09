from typing import List
from langchain_groq import ChatGroq


REWRITE_PROMPT = """You reformulate a user's question to improve retrieval from a document search system.
The documents are one person's CV, research papers, and GitHub project READMEs.

Rewrite the question to be more specific and detailed, so it is more likely to match relevant passages:
- Keep the key terms from the original question.
- Add closely related technical terms and synonyms.
- Do NOT invent names, dates, titles, or placeholders like [University].
- Output ONLY the rewritten query, on one line.

Original question: {question}

Rewritten query:"""


class QueryRewriter:
    """Rewrite a user question into a more specific search query."""

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.llm = ChatGroq(model=model_name, temperature=0, reasoning_effort="low")

    def rewrite(self, question: str) -> List[str]:
        output = self.llm.invoke(REWRITE_PROMPT.format(question=question)).content
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            print("[WARN] Rewriter returned no query, falling back to the original question")
            return [question]
        return [lines[0]]
