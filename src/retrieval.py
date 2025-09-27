import faiss
import pickle
import numpy as np

def load_index(domain: str):
    return faiss.read_index(f"vector_db/{domain}_index/index.faiss")

def load_embeddings(domain: str):
    with open(f"embeddings/{domain}_embeddings.pkl", "rb") as f:
        return pickle.load(f)

def semantic_search(query: str, domain: str, top_k: int = 5, embed_model=None):
    """
    Return top-k most similar chunks for a query
    """
    # Load embeddings and FAISS
    index = load_index(domain)
    data = load_embeddings(domain)
    
    # Generate embedding for query
    query_emb = embed_model.encode([query])
    query_emb = np.array(query_emb).astype("float32")
    
    # Search
    D, I = index.search(query_emb, top_k)
    
    results = [data['texts'][i] for i in I[0]]
    return results
