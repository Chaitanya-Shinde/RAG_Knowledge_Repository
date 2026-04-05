/**
 * Sidebar.jsx
 * - Single "Upload" button (backend auto-ingests on upload)
 * - Chat list with delete (two-click confirm)
 * - Document list with chunk count
 * - User profile / logout
 */

import { useState, useRef } from "react";
import { uploadFile, getDocuments } from "../api";

function SidebarSection({ label, children }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: "0.1em",
        color: "rgba(231,229,228,0.3)", textTransform: "uppercase",
        padding: "0 12px", marginBottom: 6,
      }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function ChatRow({ chat, active, onSelect, onDelete }) {
  const [hover, setHover] = useState(false);
  const [armed, setArmed] = useState(false);

  function handleDelete(e) {
    e.stopPropagation();
    if (armed) {
      onDelete(chat.id);
    } else {
      setArmed(true);
      setTimeout(() => setArmed(false), 2500);
    }
  }

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => { setHover(false); setArmed(false); }}
      onClick={() => onSelect(chat.id)}
      style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "7px 12px", borderRadius: 8, cursor: "pointer", marginBottom: 2,
        background: active ? "rgba(217,119,6,0.12)" : hover ? "rgba(255,255,255,0.04)" : "transparent",
        borderLeft: active ? "2px solid #D97706" : "2px solid transparent",
        transition: "background 0.15s",
      }}
    >
      <span style={{
        fontSize: 13,
        color: active ? "#FCD34D" : "rgba(231,229,228,0.75)",
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1,
      }}>
        {chat.title || "New chat"}
      </span>

      {hover && (
        <button
          onClick={handleDelete}
          title={armed ? "Click again to confirm delete" : "Delete chat"}
          style={{
            background: armed ? "rgba(220,38,38,0.2)" : "none",
            border: "none", cursor: "pointer",
            color: armed ? "#F87171" : "rgba(231,229,228,0.35)",
            fontSize: 13, padding: "2px 4px", borderRadius: 4,
            flexShrink: 0, marginLeft: 6,
            transition: "color 0.15s, background 0.15s",
          }}
        >
          {armed ? "✓" : "×"}
        </button>
      )}
    </div>
  );
}

export default function Sidebar({
  chats, activeChatId, onNewChat, onSelectChat, onDeleteChat, user, onLogout,
}) {
  const [documents, setDocuments] = useState([]);
  const [docsOpen, setDocsOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // { filename, chunks } | null
  const [profileOpen, setProfileOpen] = useState(false);
  const fileInputRef = useRef(null);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    setUploading(true);
    setUploadStatus(null);
    try {
      const result = await uploadFile(file);
      setUploadStatus({
        filename: result.filename,
        chunks: result.chunks,
        skipped: result.skipped,
      });
      // Refresh doc list
      const data = await getDocuments();
      setDocuments(data.documents ?? []);
      setDocsOpen(true);
    } catch (err) {
      setUploadStatus({ error: true });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(null), 4000);
    }
  }

  async function toggleDocs() {
    if (!docsOpen && documents.length === 0) {
      const data = await getDocuments();
      setDocuments(data.documents ?? []);
    }
    setDocsOpen((v) => !v);
  }

  return (
    <div style={{
      width: 240, minWidth: 200, background: "#111110",
      borderRight: "1px solid rgba(255,255,255,0.06)",
      display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden",
    }}>

      {/* Header */}
      <div style={{ padding: "18px 14px 10px" }}>
        <div style={{ fontSize: 16, fontWeight: 800, color: "#FCD34D", letterSpacing: "-0.02em" }}>
          Archivum
        </div>
        <div style={{ fontSize: 10, color: "rgba(231,229,228,0.3)", marginTop: 1 }}>
          RAG Knowledge Repository
        </div>
      </div>

      {/* New chat */}
      <div style={{ padding: "6px 10px 10px" }}>
        <button
          onClick={onNewChat}
          style={{
            width: "100%", background: "rgba(217,119,6,0.1)",
            border: "1px solid rgba(217,119,6,0.25)", borderRadius: 8,
            color: "#D97706", fontSize: 13, fontWeight: 600,
            padding: "7px 0", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
          }}
        >
          + New chat
        </button>
      </div>

      {/* Chat list */}
      <div style={{ flex: 1, overflowY: "auto", padding: "0 4px" }}>
        {chats.length > 0 && (
          <SidebarSection label="Chats">
            {chats.map((chat) => (
              <ChatRow
                key={chat.id}
                chat={chat}
                active={chat.id === activeChatId}
                onSelect={onSelectChat}
                onDelete={onDeleteChat}
              />
            ))}
          </SidebarSection>
        )}
      </div>

      {/* Documents */}
      <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "10px 10px 4px" }}>
        <SidebarSection label="Documents">

          {/* Single upload button */}
          <div style={{ padding: "0 2px", marginBottom: 6 }}>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              style={{
                width: "100%", background: "#1C1917",
                border: "1px solid rgba(255,255,255,0.08)", borderRadius: 7,
                color: uploading ? "rgba(231,229,228,0.35)" : "#E7E5E4",
                fontSize: 12, padding: "6px 0",
                cursor: uploading ? "default" : "pointer",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 5,
              }}
            >
              {uploading ? (
                <>
                  <span style={{
                    width: 10, height: 10, border: "1.5px solid rgba(217,119,6,0.3)",
                    borderTopColor: "#D97706", borderRadius: "50%",
                    animation: "spin 0.7s linear infinite", display: "inline-block",
                  }} />
                  Uploading & ingesting…
                </>
              ) : "↑ Upload document"}
            </button>
            <input ref={fileInputRef} type="file" style={{ display: "none" }} onChange={handleUpload} />
          </div>

          {/* Upload feedback */}
          {uploadStatus && (
            <div style={{
              fontSize: 11, padding: "4px 6px", borderRadius: 6, marginBottom: 6,
              background: uploadStatus.error ? "rgba(220,38,38,0.1)" : "rgba(34,197,94,0.1)",
              color: uploadStatus.error ? "#F87171" : "#86EFAC",
              border: `1px solid ${uploadStatus.error ? "rgba(220,38,38,0.2)" : "rgba(34,197,94,0.2)"}`,
            }}>
              {uploadStatus.error
                ? "Upload failed — please try again."
                : uploadStatus.skipped
                  ? `"${uploadStatus.filename}" already ingested.`
                  : `✓ "${uploadStatus.filename}" — ${uploadStatus.chunks} chunks ingested`}
            </div>
          )}

          {/* Toggle doc list */}
          <button
            onClick={toggleDocs}
            style={{
              width: "100%", background: "none", border: "none",
              color: "rgba(231,229,228,0.45)", fontSize: 11, textAlign: "left",
              padding: "2px 2px 4px", cursor: "pointer",
              display: "flex", alignItems: "center", gap: 4,
            }}
          >
            <span style={{ fontSize: 9 }}>{docsOpen ? "▼" : "▶"}</span>
            {documents.length > 0 ? `${documents.length} file${documents.length !== 1 ? "s" : ""}` : "View files"}
          </button>

          {docsOpen && (
            <div style={{ paddingLeft: 4, marginBottom: 4 }}>
              {documents.length === 0 ? (
                <div style={{ fontSize: 11, color: "rgba(231,229,228,0.3)", padding: "4px 0" }}>
                  No files yet.
                </div>
              ) : (
                documents.map((doc) => (
                  <div key={doc.filename} style={{
                    fontSize: 11, color: "rgba(231,229,228,0.6)", padding: "3px 0",
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                  }}>
                    <span style={{
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                      flex: 1,
                    }} title={doc.filename}>
                      📄 {doc.filename}
                    </span>
                    <span style={{ color: "rgba(231,229,228,0.3)", flexShrink: 0, marginLeft: 4 }}>
                      {doc.chunks}c
                    </span>
                  </div>
                ))
              )}
            </div>
          )}
        </SidebarSection>
      </div>

      {/* Profile */}
      <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "10px", position: "relative" }}>
        <button
          onClick={() => setProfileOpen((v) => !v)}
          style={{
            width: "100%", background: "none", border: "none",
            display: "flex", alignItems: "center", gap: 8,
            cursor: "pointer", padding: "4px 2px",
          }}
        >
          <div style={{
            width: 28, height: 28, borderRadius: "50%",
            background: "linear-gradient(135deg,#D97706,#92400E)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 13, color: "#fff", flexShrink: 0,
          }}>
            {user?.name?.[0]?.toUpperCase() ?? "U"}
          </div>
          <div style={{ textAlign: "left", overflow: "hidden" }}>
            <div style={{
              fontSize: 12, fontWeight: 600, color: "#E7E5E4",
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>
              {user?.name ?? "User"}
            </div>
            <div style={{
              fontSize: 10, color: "rgba(231,229,228,0.4)",
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>
              {user?.email ?? ""}
            </div>
          </div>
        </button>

        {profileOpen && (
          <div style={{
            position: "absolute", bottom: "100%", left: 10, right: 10,
            background: "#1C1917", border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 10, padding: 6, marginBottom: 4,
          }}>
            <button
              onClick={onLogout}
              style={{
                width: "100%", background: "none", border: "none",
                color: "#F87171", fontSize: 13, cursor: "pointer",
                padding: "7px 10px", textAlign: "left", borderRadius: 6,
              }}
            >
              Sign out
            </button>
          </div>
        )}
      </div>
    </div>
  );
}