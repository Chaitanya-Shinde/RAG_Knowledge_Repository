# app/chats.py
"""
Chat & Message persistence layer.

Collections
-----------
chats    : { id, google_id, title, created_at, updated_at }
messages : { id, chat_id, role, text, sources, eval, created_at }
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, UTC
import uuid

from app.auth.dependencies import get_current_user
from app.db.mongo import chats_collection, messages_collection

router = APIRouter(prefix="/chats", tags=["chats"])


def _new_id() -> str:
    return str(uuid.uuid4())

def _now() -> str:
    return datetime.now(UTC).isoformat()

def _clean(doc: dict) -> dict:
    doc.pop("_id", None)
    return doc


# ── Schemas ──────────────────────────────────────────────────────────────────

class CreateChatBody(BaseModel):
    title: str = "New chat"

class AddMessageBody(BaseModel):
    role: str
    text: str
    sources: Optional[list] = []
    eval: Optional[dict] = {}


# ── Chat endpoints ────────────────────────────────────────────────────────────

@router.get("")
def list_chats(user=Depends(get_current_user)):
    docs = list(chats_collection.find(
        {"google_id": user["google_id"]},
        sort=[("updated_at", -1)]
    ))
    return [_clean(d) for d in docs]


@router.post("")
def create_chat(body: CreateChatBody, user=Depends(get_current_user)):
    now = _now()
    chat_id = _new_id()
    chat = {
        "_id": chat_id,
        "id": chat_id,
        "google_id": user["google_id"],
        "title": body.title,
        "created_at": now,
        "updated_at": now,
    }
    chats_collection.insert_one(chat)
    return _clean(chat)


@router.delete("/{chat_id}")
def delete_chat(chat_id: str, user=Depends(get_current_user)):
    res = chats_collection.delete_one({"id": chat_id, "google_id": user["google_id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages_collection.delete_many({"chat_id": chat_id})
    return {"status": "deleted"}


# ── Message endpoints ─────────────────────────────────────────────────────────

@router.get("/{chat_id}/messages")
def get_messages(chat_id: str, user=Depends(get_current_user)):
    chat = chats_collection.find_one({"id": chat_id, "google_id": user["google_id"]})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    docs = list(messages_collection.find({"chat_id": chat_id}, sort=[("created_at", 1)]))
    return [_clean(d) for d in docs]