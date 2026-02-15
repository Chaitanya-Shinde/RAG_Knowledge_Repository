# Retrieval-Augmented Knowledge Repository (RAKR) — Starter Codebase

This is a starter project for a Retrieval-Augmented Knowledge Repository supporting multi-format data (PDF, DOCX, TXT, CSV, images) with semantic search and an LLM-based RAG pipeline.

## What is included
- FastAPI backend (`app/main.py`) with endpoints:
  - `POST /upload` — upload files (pdf, docx, txt, csv, json, images)
  - `POST /ingest` — process already uploaded files into chunks, embeddings, and store in ChromaDB
  - `POST /query` — query the RAG pipeline and get an answer
  - `GET /health` — health check

- Ingestion parsers (`app/ingestion/parser.py`)
- Chunking (`app/chunking.py`)
- Embeddings wrapper (`app/embeddings.py`)
- ChromaDB client (`app/db_client.py`)
- RAG pipeline orchestration (`app/rag.py`)
- Simple static frontend (`web/index.html`, `web/app.js`)
- Example evaluation script (`evaluation/evaluate.py`)
- `.env.example` for configuration

## Quickstart (local, minimal)
1. Create a Python 3.10+ virtualenv and activate it.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure environment in `.env` (copy `.env.example`).
   - If you have OpenAI set up and want to use it, set `OPENAI_API_KEY`.
   - Or set `LLAMA_CPP_MODEL_PATH` to a local GGUF model if using `llama-cpp-python`.
4. Run the app:
   ```
   uvicorn app.main:app --reload --port 8000
   ```
5. Open `web/index.html` in a browser (or serve it) and interact with the API.

## Notes
- This starter contains best-effort, production-minded code but **you must** adapt LLM integration to your available model or API.
- ChromaDB will store embeddings locally by default; change client settings as needed.
- The frontend is intentionally minimal — you can replace it with React or any UI framework.

Good luck — if you want, I can now:
- populate sample documents and run an end-to-end demo (if you allow),
- or expand the frontend to React with authentication.
