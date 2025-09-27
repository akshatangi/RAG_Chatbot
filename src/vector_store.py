# src/vector_store.py

import os
import pickle
import faiss
import numpy as np

# ------------------------------
# Config
# ------------------------------
DOMAINS = ["law", "health", "finance"]  # Add more domains if needed
VECTOR_DB_FOLDER = "vector_db"
DIM = 384  # Embedding dimension for all-MiniLM-L6-v2

# ------------------------------
# Ensure vector_db folder exists
# ------------------------------
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# ------------------------------
# Function to build FAISS index
# ------------------------------
def build_faiss_index(domain: str, dim: int = DIM):
    emb_file = f"embeddings/{domain}_embeddings.pkl"
    
    if not os.path.exists(emb_file):
        print(f"[!] Embeddings file not found for domain '{domain}': {emb_file}")
        return None

    # Load embeddings
    with open(emb_file, "rb") as f:
        data = pickle.load(f)

    embeddings = np.array(data["embeddings"]).astype("float32")
    if embeddings.shape[0] == 0:
        print(f"[!] No embeddings found for domain '{domain}'")
        return None

    # Create FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    domain_index_folder = os.path.join(VECTOR_DB_FOLDER, f"{domain}_index")
    os.makedirs(domain_index_folder, exist_ok=True)
    faiss.write_index(index, os.path.join(domain_index_folder, "index.faiss"))

    print(f"[+] FAISS index built and saved for domain '{domain}' at {domain_index_folder}")
    return index

# ------------------------------
# Build indices for all domains
# ------------------------------
if __name__ == "__main__":
    for domain in DOMAINS:
        build_faiss_index(domain)
