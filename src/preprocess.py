import os
import pandas as pd
from utils import read_txt, read_pdf, read_docx, clean_text, chunk_text

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Domains to process
DOMAINS = ["law", "health", "finance"]

for domain in DOMAINS:
    domain_path = os.path.join(DATA_DIR, domain)
    all_chunks = []

    for filename in os.listdir(domain_path):
        file_path = os.path.join(domain_path, filename)
        ext = filename.split(".")[-1].lower()

        # Read file based on extension
        if ext == "pdf":
            text = read_pdf(file_path)
        elif ext == "docx":
            text = read_docx(file_path)
        elif ext == "txt":
            text = read_txt(file_path)
        else:
            print(f"[!] Unsupported file type: {filename}")
            continue

        text = clean_text(text)
        chunks = chunk_text(text, max_words=200)

        # Store each chunk with metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "domain": domain,
                "doc_name": filename,
                "chunk_id": i,
                "text": chunk
            })

    # Save chunks as CSV
    df = pd.DataFrame(all_chunks)
    df.to_csv(os.path.join(OUTPUT_DIR, f"{domain}_chunks.csv"), index=False)
    print(f"[+] {domain} chunks saved: {len(all_chunks)} chunks")
