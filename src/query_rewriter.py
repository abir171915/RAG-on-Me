from typing import List
from langchain_groq import ChatGroq


REWRITE_PROMPT = """You rewrite a user's question into search queries for a document search system.
The documents are one person's CV, research papers, and GitHub project READMEs.

These technical terms appear in the documents. Prefer them when they are relevant:
PyTorch, CNN, ResNet, transfer learning, image classification, medical image segmentation,
LSTM, time series forecasting, neural network, XGBoost, scikit-learn, random forest,
churn prediction, delivery time prediction, LangChain, LangGraph, EEG, parking prediction

Rules:
- Write exactly 3 short search queries (3-8 words each).
- Each query should target a different angle of the question.
- Use terms from the list above where they fit. Only use other terms if none fit.
- Do NOT include the person's name. Every document is about them, so it does not help the search.
- Output ONLY the 3 queries, one per line. No numbering, no explanations.


Example:
Question: Has this person built any neural network models?
PyTorch CNN image classification
medical image segmentation neural network
LSTM time series forecasting

Example:
Question: What machine learning projects has he done?
XGBoost scikit-learn classification model
regression prediction feature engineering
churn prediction random forest

Question: {question}""" 


class QueryRewriter:
    """Turn a user question into several focused search queries."""

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.llm = ChatGroq(model=model_name, temperature=0, reasoning_effort = "low")

    def rewrite(self, question: str) -> List[str]:
        output = self.llm.invoke(REWRITE_PROMPT.format(question=question)).content
        queries = [line.strip() for line in output.splitlines() if line.strip()]
        if not queries:
            print("[WARN] Rewriter returned no queries, falling back to the original question")
            return [question]
        return queries[:3] or [question]
