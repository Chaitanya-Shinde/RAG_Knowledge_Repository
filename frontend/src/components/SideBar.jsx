import { useEffect, useRef, useState } from "react";
import { getDocuments, uploadFile, ingestDocs } from "../api";

export default function Sidebar({ chats, activeChatId, onNewChat, onSelectChat, user, onLogout }) {
  const [docs, setDocs] = useState([]);
  const [tab, setTab] = useState("chats");
  const [uploading, setUploading] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef(null);

  async function loadDocs() {
    try {
      const data = await getDocuments();
      setDocs(data.documents || []);
    } catch (err) {
      console.error("Failed to load docs", err);
    }
  }

  useEffect(() => {
    loadDocs();
  }, []);

  useEffect(() => {
    function handleClick(e) {
      if (profileRef.current && !profileRef.current.contains(e.target)) {
        setProfileOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    try {
      await uploadFile(file);
      await ingestDocs();
      await loadDocs();
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  const initials = user?.name
    ? user.name.split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()
    : "U";

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <svg width="26" height="26" viewBox="0 0 36 36" fill="none">
          <rect width="36" height="36" rx="10" fill="#D97706" />
          <path d="M10 26L18 10L26 26" stroke="#1C1917" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M13 21H23" stroke="#1C1917" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
        <span className="sidebar-brand-name">Archivum</span>
      </div>

      {/* New Chat */}
      <button className="new-chat-btn" onClick={onNewChat}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <line x1="7" y1="1" x2="7" y2="13" /><line x1="1" y1="7" x2="13" y2="7" />
        </svg>
        New chat
      </button>

      {/* Tabs */}
      <div className="sidebar-tabs">
        <button className={`sidebar-tab ${tab === "chats" ? "active" : ""}`} onClick={() => setTab("chats")}>Chats</button>
        <button className={`sidebar-tab ${tab === "docs" ? "active" : ""}`} onClick={() => setTab("docs")}>Documents</button>
      </div>

      {/* Tab Content */}
      <div className="sidebar-scroll">
        {tab === "chats" && (
          <div className="chat-list">
            {chats.length === 0 && (
              <p className="sidebar-empty">No chats yet. Start a new one above.</p>
            )}
            {chats.map(chat => (
              <button
                key={chat.id}
                className={`chat-item ${chat.id === activeChatId ? "active" : ""}`}
                onClick={() => onSelectChat(chat.id)}
              >
                <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{opacity: 0.5, flexShrink:0}}>
                  <path d="M11 1H2a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h1.5L6 12l2.5-3H11a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1z" />
                </svg>
                <span className="chat-item-title">{chat.title}</span>
              </button>
            ))}
          </div>
        )}

        {tab === "docs" && (
          <div className="doc-list">
            {docs.length === 0 && (
              <p className="sidebar-empty">No documents ingested yet.</p>
            )}
            {docs.map((d, i) => (
              <div key={i} className="doc-item">
                <svg width="13" height="14" viewBox="0 0 13 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{opacity:0.5, flexShrink:0}}>
                  <path d="M7.5 1H2.5A1.5 1.5 0 0 0 1 2.5v9A1.5 1.5 0 0 0 2.5 13h8A1.5 1.5 0 0 0 12 11.5V5.5L7.5 1z" />
                  <polyline points="7.5 1 7.5 5.5 12 5.5" />
                </svg>
                <div className="doc-item-info">
                  <span className="doc-item-name">{d.filename}</span>
                  <span className="doc-item-meta">{d.chunks} chunks</span>
                </div>
              </div>
            ))}

            <label className={`upload-btn ${uploading ? "uploading" : ""}`}>
              {uploading ? (
                <>
                  <span className="spinner" />
                  Ingesting…
                </>
              ) : (
                <>
                  <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <line x1="6.5" y1="1" x2="6.5" y2="9" />
                    <polyline points="3 5.5 6.5 1 10 5.5" />
                    <path d="M1 11h11" />
                  </svg>
                  Upload document
                </>
              )}
              <input type="file" className="hidden" onChange={handleUpload} disabled={uploading} />
            </label>
          </div>
        )}
      </div>

      {/* User Profile */}
      <div className="sidebar-footer" ref={profileRef}>
        <button className="profile-btn" onClick={() => setProfileOpen(o => !o)}>
          <div className="avatar">{initials}</div>
          <div className="profile-info">
            <span className="profile-name">{user?.name || "User"}</span>
            <span className="profile-email">{user?.email || ""}</span>
          </div>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" style={{opacity:0.4}}>
            <polyline points="3 5 7 9 11 5" />
          </svg>
        </button>

        {profileOpen && (
          <div className="profile-menu">
            <div className="profile-menu-header">
              <div className="avatar avatar-lg">{initials}</div>
              <div>
                <div className="profile-name">{user?.name || "User"}</div>
                <div className="profile-email">{user?.email || ""}</div>
              </div>
            </div>
            <div className="profile-menu-divider" />
            <button className="profile-menu-item logout" onClick={onLogout}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12H2.5A1.5 1.5 0 0 1 1 10.5v-7A1.5 1.5 0 0 1 2.5 2H5" />
                <polyline points="9 10 13 7 9 4" /><line x1="13" y1="7" x2="5" y2="7" />
              </svg>
              Sign out
            </button>
          </div>
        )}
      </div>

      <style>{`
        .sidebar {
          width: 240px;
          min-width: 240px;
          background: #161412;
          border-right: 1px solid rgba(255,255,255,0.06);
          display: flex;
          flex-direction: column;
          height: 100vh;
          position: relative;
          font-family: 'DM Sans', 'Segoe UI', sans-serif;
        }
        .sidebar-brand {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 20px 16px 16px;
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .sidebar-brand-name {
          font-size: 16px;
          font-weight: 700;
          color: #F5F5F4;
          letter-spacing: -0.02em;
        }
        .new-chat-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 12px 12px 8px;
          background: rgba(217,119,6,0.12);
          color: #D97706;
          border: 1px solid rgba(217,119,6,0.25);
          border-radius: 8px;
          padding: 9px 14px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s;
          width: calc(100% - 24px);
        }
        .new-chat-btn:hover { background: rgba(217,119,6,0.2); }
        .sidebar-tabs {
          display: flex;
          padding: 4px 12px 8px;
          gap: 4px;
        }
        .sidebar-tab {
          flex: 1;
          background: none;
          border: none;
          border-radius: 6px;
          padding: 6px;
          font-size: 12px;
          font-weight: 500;
          color: #78716C;
          cursor: pointer;
          transition: background 0.15s, color 0.15s;
        }
        .sidebar-tab.active {
          background: rgba(255,255,255,0.06);
          color: #E7E5E4;
        }
        .sidebar-tab:hover:not(.active) { color: #A8A29E; }
        .sidebar-scroll {
          flex: 1;
          overflow-y: auto;
          padding: 4px 8px;
        }
        .sidebar-scroll::-webkit-scrollbar { width: 4px; }
        .sidebar-scroll::-webkit-scrollbar-track { background: transparent; }
        .sidebar-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
        .sidebar-empty {
          font-size: 12px;
          color: #57534E;
          text-align: center;
          padding: 24px 12px;
          line-height: 1.6;
          margin: 0;
        }
        .chat-list, .doc-list { display: flex; flex-direction: column; gap: 2px; }
        .chat-item {
          display: flex;
          align-items: center;
          gap: 8px;
          background: none;
          border: none;
          border-radius: 7px;
          padding: 8px 10px;
          text-align: left;
          cursor: pointer;
          color: #A8A29E;
          font-size: 13px;
          transition: background 0.12s, color 0.12s;
          width: 100%;
        }
        .chat-item:hover { background: rgba(255,255,255,0.05); color: #E7E5E4; }
        .chat-item.active { background: rgba(217,119,6,0.1); color: #E7E5E4; }
        .chat-item-title {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          flex: 1;
        }
        .doc-item {
          display: flex;
          align-items: flex-start;
          gap: 8px;
          padding: 8px 10px;
          border-radius: 7px;
          color: #A8A29E;
        }
        .doc-item-info { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
        .doc-item-name {
          font-size: 12px;
          color: #D6D3D1;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .doc-item-meta { font-size: 11px; color: #57534E; }
        .upload-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 7px;
          margin-top: 12px;
          padding: 9px;
          border-radius: 8px;
          border: 1px dashed rgba(255,255,255,0.1);
          color: #78716C;
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          transition: border-color 0.15s, color 0.15s;
          background: none;
          width: 100%;
        }
        .upload-btn:hover { border-color: rgba(217,119,6,0.4); color: #D97706; }
        .upload-btn.uploading { opacity: 0.6; cursor: default; }
        .spinner {
          width: 12px; height: 12px;
          border: 1.5px solid rgba(217,119,6,0.3);
          border-top-color: #D97706;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
          flex-shrink: 0;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .sidebar-footer {
          padding: 12px;
          border-top: 1px solid rgba(255,255,255,0.05);
          position: relative;
        }
        .profile-btn {
          display: flex;
          align-items: center;
          gap: 10px;
          width: 100%;
          background: none;
          border: none;
          border-radius: 8px;
          padding: 8px;
          cursor: pointer;
          transition: background 0.12s;
        }
        .profile-btn:hover { background: rgba(255,255,255,0.05); }
        .avatar {
          width: 30px; height: 30px;
          border-radius: 50%;
          background: rgba(217,119,6,0.2);
          color: #D97706;
          font-size: 11px;
          font-weight: 700;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          letter-spacing: 0.02em;
        }
        .avatar-lg { width: 38px; height: 38px; font-size: 13px; }
        .profile-info { flex: 1; min-width: 0; text-align: left; }
        .profile-name {
          font-size: 12px;
          font-weight: 600;
          color: #E7E5E4;
          display: block;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .profile-email {
          font-size: 11px;
          color: #57534E;
          display: block;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .profile-menu {
          position: absolute;
          bottom: calc(100% + 6px);
          left: 12px;
          right: 12px;
          background: #1C1917;
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 10px;
          padding: 12px;
          z-index: 100;
          box-shadow: 0 -8px 32px rgba(0,0,0,0.4);
        }
        .profile-menu-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
        }
        .profile-menu-divider {
          height: 1px;
          background: rgba(255,255,255,0.06);
          margin: 8px 0;
        }
        .profile-menu-item {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          background: none;
          border: none;
          border-radius: 6px;
          padding: 8px;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: background 0.12s;
        }
        .profile-menu-item.logout { color: #EF4444; }
        .profile-menu-item.logout:hover { background: rgba(239,68,68,0.1); }
      `}</style>
    </aside>
  );
}
