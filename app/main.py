# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import os, logging
from pathlib import Path
from datetime import datetime, UTC
import uuid
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.drive.drive_service import get_drive_service, upload_file_to_drive, stream_file_from_drive
from app.db.mongo import users_collection, chats_collection, messages_collection
from app.auth.dependencies import get_current_user

load_dotenv()
LOG = logging.getLogger("uvicorn.error")

from .ingestion.parser import parse_file
from .chunking import chunk_texts
from .embeddings import EmbeddingModel
from .db_client import ChromaClient
from .rag import RAGSystem

UPLOAD_DIR = Path("./uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAKR - Retrieval-Augmented Knowledge Repo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embed_model = None
db = None
rag = None

from app.auth.google_auth import router as google_auth_router
from app.chats import router as chats_router

app.include_router(google_auth_router, prefix="/auth")
app.include_router(chats_router)   # prefix="/chats" declared inside chats.py


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    global embed_model, db, rag
    LOG.info("Startup: initializing resources…")

    try:
        embed_model = EmbeddingModel(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        LOG.info("Embedding model loaded.")
    except Exception as e:
        LOG.exception("Failed to load embedding model: %s", e)
        embed_model = None

    try:
        persist_dir = os.path.abspath(os.getenv("CHROMA_DB_PARENT_DIR", "./chroma_db"))
        db = ChromaClient(client_settings={"persist_directory": persist_dir})
        LOG.info("Chroma client ready.")
    except Exception as e:
        LOG.exception("Failed to initialize Chroma client: %s", e)
        db = None

    try:
        rag = RAGSystem(embed_model=embed_model, db_client=db)
        LOG.info("RAG system initialized.")
    except Exception as e:
        LOG.exception("Failed to create RAGSystem: %s", e)
        rag = None

    LOG.info("Startup complete.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_model_loaded": bool(embed_model),
        "db_ready": bool(db),
        "rag_ready": bool(rag),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Upload — immediately ingests after uploading to Drive
# ─────────────────────────────────────────────────────────────────────────────

def _ingest_file_stream(file_stream, filename: str, google_id: str):
    """Parse → chunk → embed → store. Returns chunk count."""
    collection = db.get_user_collection(google_id)

    # Skip if already ingested
    try:
        existing = collection.get(where={"source": filename}, limit=1)
        if len(existing.get("ids", [])) > 0:
            LOG.info("Already ingested: %s — skipping.", filename)
            return {"skipped": True, "chunks": 0}
    except Exception:
        pass

    text = parse_file(file_stream, filename)
    chunks = chunk_texts(text)
    if not chunks:
        return {"skipped": False, "chunks": 0}

    embeddings = embed_model.embed_documents(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{filename}_{i}" for i in range(len(chunks))]
    )
    
    embeddings = embed_model.embed_documents(chunks)

    BATCH_SIZE = 100

    metadatas = [
        {"source": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    for i in range(0, len(chunks), BATCH_SIZE):

        batch_docs = chunks[i:i+BATCH_SIZE]
        batch_embeds = embeddings[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_meta,
            ids=batch_ids
        )

        LOG.info(
            "Inserted batch %d–%d",
            i,
            min(i + BATCH_SIZE, len(chunks))
        )
    LOG.info("Ingested '%s': %d chunks.", filename, len(chunks))
    return {"skipped": False, "chunks": len(chunks)}


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """
    Upload a file to Google Drive and immediately ingest it into Chroma.
    No separate /ingest step needed.
    """
    try:
        import io
        raw_bytes = await file.read()

        if len(raw_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

        # Upload to Drive
        drive_service = get_drive_service(user["refresh_token"])
        # Reset stream so Drive can read it
        file.file.seek(0)

        drive_file_id = upload_file_to_drive(
            drive_service,
            file,
            file.filename,
            user["drive_folder_id"]
        )

        # Ingest immediately from the same bytes
        ingest_result = _ingest_file_stream(
            io.BytesIO(raw_bytes), file.filename, user["google_id"]
        )

        return {
            "status": "uploaded_and_ingested",
            "filename": file.filename,
            "drive_file_id": drive_file_id,
            "chunks": ingest_result["chunks"],
            "skipped": ingest_result["skipped"],
        }

    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("Upload/ingest error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Query — loads history from MongoDB, passes it to RAG, persists messages
# ─────────────────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4())

def _now() -> str:
    return datetime.now(UTC).isoformat()


@app.post("/query")
async def query(
    prompt: str = Form(...),
    model: str = Form("gemini"),
    k: int = Form(10),
    chat_id: str = Form(None),
    user=Depends(get_current_user)
):
    if embed_model is None:
        return JSONResponse({"error": "Embedding model not initialized"}, status_code=500)

    # 1. Load this chat's full conversation history from MongoDB
    history: list[dict] = []
    chat = None
    if chat_id:
        chat = chats_collection.find_one({"id": chat_id, "google_id": user["google_id"]})
        if chat:
            raw_msgs = list(
                messages_collection.find(
                    {"chat_id": chat_id},
                    sort=[("created_at", 1)]
                )
            )
            history = [{"role": m["role"], "text": m["text"]} for m in raw_msgs]

    # 2. Run RAG with history
    result = rag.answer(prompt, user["google_id"], k, model, history=history)

    # 3. Persist user + assistant messages
    if chat_id and chat:
        user_msg = {
            "_id": _new_id(), "chat_id": chat_id,
            "role": "user", "text": prompt,
            "sources": [], "eval": {},
            "created_at": _now(),
        }
        user_msg["id"] = user_msg["_id"]
        messages_collection.insert_one(user_msg)

        asst_msg = {
            "_id": _new_id(), "chat_id": chat_id,
            "role": "assistant", "text": result["answer"],
            "sources": result.get("sources", []),
            "eval": result.get("eval", {}),
            "created_at": _now(),
        }
        asst_msg["id"] = asst_msg["_id"]
        messages_collection.insert_one(asst_msg)

        update_fields: dict = {"updated_at": _now()}
        if chat.get("title") == "New chat":
            update_fields["title"] = prompt[:60] + ("…" if len(prompt) > 60 else "")
        chats_collection.update_one({"id": chat_id}, {"$set": update_fields})

    return JSONResponse({
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "eval": result.get("eval", {}),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/documents")
async def list_documents(user=Depends(get_current_user)):
    drive_service = get_drive_service(user["refresh_token"])
    results = drive_service.files().list(
        q=f"'{user['drive_folder_id']}' in parents and trashed=false",
        fields="files(id, name, size)"
    ).execute()
    files = results.get("files", [])
    collection = db.get_user_collection(user["google_id"])
    documents = []
    for f in files:
        filename = f["name"]
        try:
            res = collection.get(where={"source": filename})
            chunk_count = len(res.get("ids", []))
        except Exception:
            chunk_count = 0
        documents.append({"filename": filename, "chunks": chunk_count})
    return {"documents": documents}


@app.delete("/documents/{filename}")
async def delete_document(filename: str, user=Depends(get_current_user)):
    drive_service = get_drive_service(user["refresh_token"])
    results = drive_service.files().list(
        q=f"name='{filename}' and '{user['drive_folder_id']}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()
    files = results.get("files", [])
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    drive_service.files().delete(fileId=files[0]["id"]).execute()
    collection = db.get_user_collection(user["google_id"])
    collection.delete(where={"source": filename})
    return {"status": "deleted", "filename": filename}


@app.get("/me")
def me(user=Depends(get_current_user)):
    return {"email": user["email"], "name": user["name"]}


# Serve frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")