# src/embeddings.py

import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # Hugging Face model
DOMAINS = ["law"]  # Add more domains if needed
# DOMAINS = ["law", "health", "finance"]  
CHUNK_COLUMN = "text"  # Name of the column in your CSV containing text

# ------------------------------
# Prepare model
# ------------------------------
print(f"[+] Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
EMBED_MODEL = model  # Export for use in other modules

# ------------------------------
# Ensure embeddings folder exists
# ------------------------------
os.makedirs("embeddings", exist_ok=True)

# ------------------------------
# Process each domain
# ------------------------------
for domain in DOMAINS:
    csv_path = f"preprocessed/{domain}_chunks.csv"
    emb_path = f"embeddings/{domain}_embeddings.pkl"

    if not os.path.exists(csv_path):
        print(f"[!] CSV file not found for domain '{domain}': {csv_path}")
        continue

    # Load CSV
    df = pd.read_csv(csv_path)

    if CHUNK_COLUMN not in df.columns:
        print(f"[!] Column '{CHUNK_COLUMN}' not found in {csv_path}")
        print(f"    Available columns: {df.columns.tolist()}")
        continue

    if df.empty:
        print(f"[!] No chunks found in {csv_path}")
        continue

    # Get text chunks
    texts = df[CHUNK_COLUMN].tolist()
    print(f"[+] Loaded {len(texts)} chunks for domain '{domain}'")

    # Generate embeddings
    print(f"[+] Generating embeddings for {domain}...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save embeddings
    with open(emb_path, "wb") as f:
        pickle.dump({"texts": texts, "embeddings": embeddings}, f)

    print(f"[+] Embeddings saved for domain '{domain}' at: {emb_path}\n")
