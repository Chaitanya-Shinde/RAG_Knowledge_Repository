const API = "http://localhost:8000";

// ─── LLM Query ───────────────────────────────────────────────────────────────

export async function queryLLM(prompt, model, chatId = null) {
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("model", model);
  form.append("k", 10);
  if (chatId) form.append("chat_id", chatId);

  const res = await fetch(`${API}/query`, {
    method: "POST",
    body: form,
    credentials: "include",
  });
  return res.json();
}

// ─── File management ─────────────────────────────────────────────────────────

/**
 * Upload a file. The backend now immediately ingests it — no separate
 * ingest step needed. Returns { filename, chunks, skipped, drive_file_id }.
 */
export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/upload`, {
    method: "POST",
    body: form,
    credentials: "include",
  });
  return res.json();
}

export async function getDocuments() {
  const res = await fetch(`${API}/documents`, { credentials: "include" });
  return res.json();
}

export async function deleteDocument(filename) {
  const res = await fetch(`${API}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    credentials: "include",
  });
  return res.json();
}

// ─── Auth ─────────────────────────────────────────────────────────────────────

export async function getCurrentUser() {
  const res = await fetch(`${API}/me`, { credentials: "include" });
  if (!res.ok) return null;
  return res.json();
}

// ─── Chats ───────────────────────────────────────────────────────────────────

export async function getChats() {
  const res = await fetch(`${API}/chats`, { credentials: "include" });
  if (!res.ok) return [];
  return res.json();
}

export async function createChat(title = "New chat") {
  const res = await fetch(`${API}/chats`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ title }),
  });
  return res.json();
}

export async function deleteChat(chatId) {
  const res = await fetch(`${API}/chats/${chatId}`, {
    method: "DELETE",
    credentials: "include",
  });
  return res.json();
}

// ─── Messages ────────────────────────────────────────────────────────────────

export async function getMessages(chatId) {
  const res = await fetch(`${API}/chats/${chatId}/messages`, {
    credentials: "include",
  });
  if (!res.ok) return [];
  return res.json();
}