const API_BASE = "http://127.0.0.1:8000";

/* ---------------- CHAT ---------------- */

async function send() {
  const p = document.getElementById('prompt').value;
  if (!p) return;

  append('user', p);

  const form = new FormData();
  form.append('prompt', p);
  form.append('k', 5);

  try {
    const res = await fetch(API_BASE + '/query', {
      method: 'POST',
      body: form
    });

    if (!res.ok) {
      append('bot', 'Error: ' + res.statusText);
      return;
    }

    const data = await res.json();

    append(
      'bot',
      data.answer + '\n\nSources: ' + (data.sources || []).join(', ')
    );

  } catch (err) {
    append('bot', 'Server error.');
  }
}

function append(cls, text) {
  const div = document.getElementById('chat');
  const el = document.createElement('div');
  el.className = cls === 'user' ? 'user' : 'bot';
  el.textContent = text;
  div.appendChild(el);
  div.scrollTop = div.scrollHeight;
}


/* ---------------- UPLOAD + AUTO INGEST ---------------- */

async function uploadAndIngest() {
  const fileInput = document.getElementById('fileInput');
  const status = document.getElementById('uploadStatus');

  if (!fileInput.files.length) {
    status.innerText = "Please select a file.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    status.innerText = "Uploading file...";

    // 1️⃣ Upload file
    const uploadRes = await fetch(API_BASE + "/upload", {
      method: "POST",
      body: formData
    });

    if (!uploadRes.ok) {
      status.innerText = "❌ Upload failed.";
      return;
    }

    const uploadData = await uploadRes.json();

    status.innerText = "File uploaded. Ingesting...";

    // 2️⃣ Trigger ingestion
    const ingestRes = await fetch(API_BASE + "/ingest", {
      method: "POST"
    });

    if (!ingestRes.ok) {
      status.innerText = "❌ Ingestion failed.";
      return;
    }

    const ingestData = await ingestRes.json();

    status.innerText =
      `✅ ${uploadData.filename} uploaded and ingested successfully!`;

    // Optional: clear file input
    fileInput.value = "";

  } catch (err) {
    status.innerText = "❌ Operation failed.";
  }
}