import { useState, useEffect, useCallback } from "react";
import { getCurrentUser, getChats, createChat, deleteChat } from "./api";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import Login from "./pages/Login";

export default function App() {
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);

  // chats: [{ id, title, created_at, updated_at }]
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);

  // messages are owned by <Chat> and loaded fresh on each chat switch
  // we pass a "key" to force remount when switching chats

  // ── auth check ────────────────────────────────────────────────────────
  useEffect(() => {
    async function checkAuth() {
      const u = await getCurrentUser();
      setUser(u);
      setAuthChecked(true);
    }
    checkAuth();
  }, []);

  // ── load chats from backend once authenticated ─────────────────────────
  useEffect(() => {
    if (!user) return;
    async function loadChats() {
      const data = await getChats();
      if (data.length > 0) {
        setChats(data);
        setActiveChatId(data[0].id);
      } else {
        // Create a first chat automatically
        const chat = await createChat("New chat");
        setChats([chat]);
        setActiveChatId(chat.id);
      }
    }
    loadChats();
  }, [user]);

  // ── handlers ──────────────────────────────────────────────────────────
  async function handleNewChat() {
    const chat = await createChat("New chat");
    setChats((prev) => [chat, ...prev]);
    setActiveChatId(chat.id);
  }

  function handleSelectChat(id) {
    setActiveChatId(id);
  }

  async function handleDeleteChat(id) {
    await deleteChat(id);
    setChats((prev) => {
      const remaining = prev.filter((c) => c.id !== id);
      if (activeChatId === id) {
        setActiveChatId(remaining[0]?.id ?? null);
      }
      return remaining;
    });
  }

  /** Called by Chat when the backend auto-renames a chat after first message. */
  function handleChatTitleUpdate(chatId, newTitle) {
    setChats((prev) =>
      prev.map((c) => (c.id === chatId ? { ...c, title: newTitle } : c))
    );
  }

  function handleLogout() {
    window.location.href = "http://localhost:8000/auth/logout";
  }

  // ── render ────────────────────────────────────────────────────────────
  if (!authChecked) {
    return (
      <div
        style={{
          height: "100vh",
          background: "#0C0A09",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: 28,
            height: 28,
            border: "2px solid rgba(217,119,6,0.2)",
            borderTopColor: "#D97706",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }}
        />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!user) return <Login />;

  const activeChat = chats.find((c) => c.id === activeChatId);

  return (
    <div style={{ height: "100vh", display: "flex", background: "#0C0A09" }}>
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        user={user}
        onLogout={handleLogout}
      />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {activeChat ? (
          <Chat
            key={activeChat.id}          // remounts on chat switch → fresh message load
            chatId={activeChat.id}
            onTitleUpdate={handleChatTitleUpdate}
          />
        ) : (
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "rgba(231,229,228,0.3)",
              fontSize: 15,
            }}
          >
            Select or create a chat to begin.
          </div>
        )}
      </div>
    </div>
  );
}
