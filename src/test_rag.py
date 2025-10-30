from rag_pipeline import rag_query
from sentence_transformers import SentenceTransformer

# ----------------- Settings -----------------
domain = "law"
query = "How to get insurance claim for car accident?"

# ----------------- Run RAG Query -----------------
answer = rag_query(user_query=query, domain=domain, top_k=5)

# ----------------- Print Output -----------------
print("\nGenerated Answer:\n")
print(answer)
