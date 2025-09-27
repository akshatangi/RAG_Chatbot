source conversational_env/bin/activate 

rag-chatbot/
│
├── data/                       # Raw domain documents
│   ├── law/
│   │   ├── case1.pdf
│   │   ├── contract_law.docx
│   │   └── statutes.txt
│   ├── health/
│   │   ├── diet_guidelines.pdf
│   │   ├── diabetes_research.docx
│   │   └── medical_notes.txt
│   └── finance/
│       ├── financial_report.pdf
│       ├── regulations.docx
│       └── tutorials.txt
│
├── preprocessed/               # Chunked & cleaned text per domain
│   ├── law_chunks.csv
│   ├── health_chunks.csv
│   └── finance_chunks.csv
│
├── embeddings/                 # Embeddings for each domain
│   ├── law_embeddings.pkl
│   ├── health_embeddings.pkl
│   └── finance_embeddings.pkl
│
├── vector_db/                  # FAISS / Chroma indices per domain
│   ├── law_index/
│   ├── health_index/
│   └── finance_index/
│
├── src/                        # Source code
│   ├── preprocess.py           # Phase 1: document parsing + chunking
│   ├── embeddings.py           # Phase 2: embedding generation
│   ├── vector_store.py         # Phase 2: FAISS / Chroma setup
│   ├── retrieval.py            # Retrieval logic
│   ├── rag_pipeline.py         # Full RAG pipeline (retrieval + GPT)
│   ├── api.py                  # FastAPI backend endpoints
│   └── utils.py                # Helper functions (file read, chunking, metadata)
│
├── tests/                      # Test scripts
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   └── test_api.py
│
├── frontend/                   # Optional Streamlit / React frontend
│   ├── app.py
│   └── components/
│
├── requirements.txt
└── README.md
