import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from config import GEMINI_API_KEY, PROVIDER

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ----------------- Embedding Model -----------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- Gemini Generation -----------------
def generate_with_gemini(prompt, api_key):
    try:
        model = genai.GenerativeModel("gemini-pro-latest")
        response = model.generate_content(prompt)
        return response.text or "[No response from Gemini]"
    except Exception as e:
        return f"[Gemini API error]: {str(e)}"


# ----------------- Semantic Search -----------------
def load_index(domain: str):
    return faiss.read_index(f"vector_db/{domain}_index/index.faiss")

def load_embeddings(domain: str):
    with open(f"embeddings/{domain}_embeddings.pkl", "rb") as f:
        return pickle.load(f)

def semantic_search(query: str, domain: str, top_k: int = 5):
    index = load_index(domain)
    data = load_embeddings(domain)
    query_emb = EMBED_MODEL.encode([query])
    query_emb = np.array(query_emb).astype("float32")
    D, I = index.search(query_emb, top_k)
    results = [data['texts'][i] for i in I[0]]
    return results

# ----------------- RAG Pipeline -----------------
def rag_query(user_query: str, domain: str, top_k: int = 5) -> str:
    # Step 1: Retrieve relevant context
    context_chunks = semantic_search(user_query, domain, top_k)
    context_text = "\n".join(context_chunks)

    # Step 2: Build prompt
    prompt = f"Use the following context to answer the question:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"

    # Step 3: Generate answer using Gemini
    if PROVIDER.lower() == "gemini":
        return generate_with_gemini(prompt, GEMINI_API_KEY)
    else:
        raise ValueError("Currently only Gemini provider is supported")
    # Future: Add OpenAI or other providers here