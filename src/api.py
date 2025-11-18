# src/api.py
import os
import time
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# import your existing functions/modules
from retrieval import semantic_search, load_embeddings, load_index
from rag_pipeline import rag_query
from embeddings import EMBED_MODEL  # if your file exports model variable
from config import PROVIDER

app = FastAPI(title="RAG Chatbot API", version="1.0")

logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO)

# --- Simple API key auth dependency (replace with real auth in prod) ---
API_KEYS = {os.getenv("API_KEY", "devkey123")}

def require_api_key(x_api_key: str = Depends(lambda: os.getenv("API_KEY", "devkey123"))):
    # a simple placeholder — in FastAPI you'd implement header reading and compare
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# --------------------
# Request / Response models
# --------------------
class SearchRequest(BaseModel):
    query: str
    domain: str = "law"
    top_k: int = 5

class RagRequest(BaseModel):
    query: str
    domain: str = "law"
    top_k: int = 5

class ReindexRequest(BaseModel):
    domain: str

# --------------------
# Endpoints
# --------------------
@app.post("/search")
async def search(req: SearchRequest, authorized: bool = Depends(require_api_key)):
    """
    Returns top-k chunks (raw) from FAISS for debugging or UI.
    """
    try:
        texts = semantic_search(query=req.query, domain=req.domain, top_k=req.top_k, embed_model=EMBED_MODEL)
        return {"query": req.query, "domain": req.domain, "top_k": req.top_k, "results": texts}
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
async def rag(req: RagRequest, authorized: bool = Depends(require_api_key)):
    """
    Full RAG: retrieve + generate (blocking).
    For large-scale: convert to background + streaming.
    """
    try:
        answer = rag_query(user_query=req.query, domain=req.domain, top_k=req.top_k)
        return {"query": req.query, "domain": req.domain, "answer": answer}
    except Exception as e:
        logger.exception("RAG failed")
        raise HTTPException(status_code=500, detail=str(e))

# Upload endpoint (simple synchronous ingestion + background reindex)
@app.post("/upload")
async def upload(domain: str, file: UploadFile = File(...), background_tasks: BackgroundTasks = None, authorized: bool = Depends(require_api_key)):
    """
    Upload a document (pdf/docx/txt). Saves to data/<domain>/ and schedules background job to preprocess + reindex.
    """
    try:
        target_dir = os.path.join("data", domain)
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        # schedule background reindex
        if background_tasks:
            background_tasks.add_task(reindex_domain, domain)
        return {"status": "uploaded", "path": file_path}
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# Reindex implementation: wrapper that calls your preprocess -> embeddings -> faiss pipeline
@app.post("/reindex")
async def reindex(req: ReindexRequest, authorized: bool = Depends(require_api_key)):
    try:
        # blocking reindex for simplicity — you can run as background task/job
        reindex_domain(req.domain)
        return {"status": "reindex_started", "domain": req.domain}
    except Exception as e:
        logger.exception("Reindex failed")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------
# Helper functions (wrap your phase2 scripts)
# --------------------
def reindex_domain(domain: str):
    """
    1) run your preprocess for domain
    2) generate embeddings for domain
    3) build faiss index for domain
    Replace with imports to your exact functions.
    """
    # Example calls (replace with your actual function names)
    from preprocess import process_domain  # if you have such a function
    from embeddings import embed_domain
    from vector_store import build_faiss_index

    try:
        # 1) preprocess (if you have)
        if hasattr(process_domain, "__call__"):
            process_domain(domain)
        # 2) run embeddings per domain
        if hasattr(embed_domain, "__call__"):
            embed_domain(domain)
        # 3) build faiss
        build_faiss_index(domain)
        logger.info(f"Reindex complete for {domain}")
    except Exception:
        logger.exception("Reindex domain failed")
        raise

# --------------------
# Root health
# --------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}

# To run using `python -m uvicorn src.api:app --reload` or uvicorn CLI
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)