# app/main.py  (replace your existing file with this)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, shutil, logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles


load_dotenv()
LOG = logging.getLogger("uvicorn.error")

# Keep imports for classes but avoid heavy initialization here
from .ingestion.parser import parse_file
from .chunking import chunk_texts
from .embeddings import EmbeddingModel
from .db_client import ChromaClient
from .rag import RAGSystem

UPLOAD_DIR = Path("./uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAKR - Retrieval-Augmented Knowledge Repo")


# app/main.py (snippet to add near top, after `app = FastAPI(...)`)
from fastapi.middleware.cors import CORSMiddleware

# Development CORS settings (allow everything). For production, restrict origins.
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["*"],            # change to your frontend origin in prod
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# place holders to be filled at startup
embed_model = None
db = None
rag = None

@app.on_event("startup")
def startup_event():
    global embed_model, db, rag
    LOG.info("Startup event: initializing resources...")

    # Initialize embedding model lazily
    try:
        LOG.info("Loading embedding model: %s", os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"))
        embed_model = EmbeddingModel(model_name=os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"))
        LOG.info("Embedding model loaded.")
    except Exception as e:
        LOG.exception("Failed to load embedding model: %s", e)
        embed_model = None

    # Initialize Chroma client (persistent if configured)
    try:
        persist_dir = os.path.abspath(os.getenv("CHROMA_DB_PARENT_DIR", "./chroma_db"))
        LOG.info("Creating Chroma client (persist_dir=%s)", persist_dir)
        db = ChromaClient(client_settings={"persist_directory": persist_dir} if persist_dir else None)
        LOG.info("Chroma client ready.")
    except Exception as e:
        LOG.exception("Failed to initialize Chroma client: %s", e)
        db = None

    # RAG orchestration (works even if embed_model/db are None; errors handled later)
    try:
        rag = RAGSystem(embed_model=embed_model, db_client=db)
        LOG.info("RAG system initialized.")
    except Exception as e:
        LOG.exception("Failed to create RAGSystem: %s", e)
        rag = None

    LOG.info("Startup complete.")

@app.get("/health")
def health():
    """
    Lightweight health endpoint that returns quickly.
    Uses simple checks rather than waiting on heavy objects.
    """
    status = {"status": "ok", "embed_model_loaded": bool(embed_model), "db_ready": bool(db), "rag_ready": bool(rag)}
    return status

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status":"uploaded", "filename": str(dest)}

@app.post('/ingest')
def ingest():
    if rag is None:
        return JSONResponse({"error": "RAG system not initialized"}, status_code=500)

    files = list(UPLOAD_DIR.iterdir())
    added = 0
    for f in files:
        parsed = parse_file(str(f))
        chunks = chunk_texts(parsed["content"], max_tokens=int(os.getenv("MAX_CHUNK_TOKENS",500)))
        docs = []
        for i,c in enumerate(chunks):
            docs.append({
                "id": f"{f.name}::chunk_{i}",
                "text": c,
                "metadata": {"source": str(f.name)}
            })
        rag.index_documents(docs)
        added += len(docs)
    return {"status":"ingested", "documents_added": added}

@app.post('/query')
async def query(prompt: str = Form(...), k: int = Form(5)):
    if rag is None:
        return JSONResponse({"error": "RAG system not initialized"}, status_code=500)
    result = rag.answer(prompt, top_k=k)
    return JSONResponse(result)


# Serve frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")
