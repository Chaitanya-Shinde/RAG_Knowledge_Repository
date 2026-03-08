const API = "http://localhost:8000";

export async function queryLLM(prompt, model) {

  const form = new FormData();
  form.append("prompt", prompt);
  form.append("model", model);
  form.append("k", 10);

  const res = await fetch(`${API}/query`, {
    method: "POST",
    body: form,
    credentials: "include"
  });

  return res.json();
}

export async function uploadFile(file) {

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API}/upload`, {
    method: "POST",
    body: form,
    credentials: "include"
  });

  return res.json();
}

export async function ingestDocs() {

  const res = await fetch(`${API}/ingest`, {
    method: "POST",
    credentials: "include"
  });

  return res.json();
}

export async function getDocuments() {

  const res = await fetch(`${API}/documents`, {
    credentials: "include"
  });

  return res.json();
}

export async function getCurrentUser() {

  const res = await fetch("http://localhost:8000/me", {
    credentials: "include"
  });

  if (!res.ok) return null;

  return res.json();
}