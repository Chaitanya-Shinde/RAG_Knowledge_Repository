# app/main.py  (replace your existing file with this)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import os, shutil, logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from app.drive.drive_service import get_drive_service, upload_file_to_drive, stream_file_from_drive
from app.db.mongo import users_collection
from app.auth.jwt_handler import verify_token
from app.auth.dependencies import get_current_user




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
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# place holders to be filled at startup
embed_model = None
db = None
rag = None

from app.auth.google_auth import router as google_auth_router

app.include_router(google_auth_router, prefix="/auth")

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


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    try:

        drive_service = get_drive_service(user["refresh_token"])

        file_id = upload_file_to_drive(
            drive_service,
            file,
            file.filename, 
            user["drive_folder_id"],
        )

        return {
            "status": "uploaded_to_drive",
            "filename": file.filename,
            "drive_file_id": file_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def file_already_ingested(collection, filename: str) -> bool:
    try:
        # Try fetching one chunk from this file
        res = collection.get(where={"source": filename}, limit=1)
        return len(res.get("ids", [])) > 0
    except Exception:
        return False



MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
@app.post("/ingest")
async def ingest(user=Depends(get_current_user)):
    try:
        drive_service = get_drive_service(user["refresh_token"])

        results = drive_service.files().list(
            q=f"'{user['drive_folder_id']}' in parents and trashed=false",
            fields="files(id, name, size)"
        ).execute()

        files = results.get("files", [])

        if not files:
            return {"message": "No files found in Drive folder."}

        collection = db.get_user_collection(user["google_id"])

        for file in files:
            file_id = file["id"]
            filename = file["name"]
            size = int(file.get("size", 0))

            # 🔒 1️⃣ File size validation
            if size > MAX_FILE_SIZE:
                print(f"Skipping {filename} — too large.")
                continue

            # 🔁 Skip already ingested files
            if file_already_ingested(collection, filename):
                continue

            print(f"Ingesting {filename}...")

            # 📥 2️⃣ Stream file from Drive (no disk)
            file_stream = stream_file_from_drive(drive_service, file_id)

            # 📝 3️⃣ Parse in memory
            text = parse_file(file_stream, filename)

            # ✂ 4️⃣ Chunk immediately
            chunks = chunk_texts(text)

            # 🧠 5️⃣ Embed immediately
            embeddings = embed_model.embed_documents(chunks)

            # 💾 6️⃣ Store in per-user collection
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=[
                    {"source": filename, "chunk_index": i}
                    for i in range(len(chunks))
                ],
                ids=[f"{filename}_{i}" for i in range(len(chunks))]
            )

        return {"message": "Ingestion complete."}

    except Exception as e:
        print("INGEST ERROR:", e)
        raise


@app.post('/query')
async def query(
    prompt: str = Form(...),
    model: str = Form("gemini"),
    k: int = Form(10),
    user=Depends(get_current_user)
):
    if embed_model is None:
        return JSONResponse({"error": "Embedding model not initialized"}, status_code=500)

    result = rag.answer(prompt, user["google_id"], k, model)

    # Return answer + sources + eval metadata for benchmarking
    return JSONResponse({
        "answer": result["answer"],
        "sources": result["sources"],
        "eval": result.get("eval", {})   # <-- add this line
    })

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
        except:
            chunk_count = 0

        documents.append({
            "filename": filename,
            "chunks": chunk_count
        })

    return {"documents": documents}

@app.delete("/documents/{filename}")
async def delete_document(filename: str, user=Depends(get_current_user)):

    drive_service = get_drive_service(user["refresh_token"])

    # Find file in Drive
    results = drive_service.files().list(
        q=f"name='{filename}' and '{user['drive_folder_id']}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])

    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_id = files[0]["id"]

    # Delete from Drive
    drive_service.files().delete(fileId=file_id).execute()

    # Delete vectors
    collection = db.get_user_collection(user["google_id"])

    collection.delete(
        where={"source": filename}
    )

    return {
        "status": "deleted",
        "filename": filename
    }

@app.get("/me")
def me(user=Depends(get_current_user)):
    return {
        "email": user["email"],
        "name": user["name"]
    }

# Serve frontend
app.mount("/", StaticFiles(directory="web", html=True), name="web")
