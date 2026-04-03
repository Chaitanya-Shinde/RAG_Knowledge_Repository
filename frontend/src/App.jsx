import { useState, useEffect } from "react";
import { getCurrentUser } from "./api";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import Login from "./pages/Login";

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

function createChat() {
  return { id: generateId(), title: "New chat", messages: [] };
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [chats, setChats] = useState([createChat()]);
  const [activeChatId, setActiveChatId] = useState(null);

  useEffect(() => {
    async function checkAuth() {
      const u = await getCurrentUser();
      setUser(u);
      setAuthChecked(true);
    }
    checkAuth();
  }, []);

  useEffect(() => {
    if (chats.length > 0 && !activeChatId) {
      setActiveChatId(chats[0].id);
    }
  }, [chats, activeChatId]);

  if (!authChecked) {
    return (
      <div style={{ height: "100vh", background: "#0C0A09", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ width: 28, height: 28, border: "2px solid rgba(217,119,6,0.2)", borderTopColor: "#D97706", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (!user) {
    return <Login />;
  }

  const activeChat = chats.find(c => c.id === activeChatId) || chats[0];

  function handleNewChat() {
    const chat = createChat();
    setChats(prev => [chat, ...prev]);
    setActiveChatId(chat.id);
  }

  function handleSelectChat(id) {
    setActiveChatId(id);
  }

  function setMessages(updater) {
    setChats(prev => prev.map(c => {
      if (c.id !== activeChatId) return c;
      const newMessages = typeof updater === "function" ? updater(c.messages) : updater;
      // Auto-title from first user message
      const title = c.title === "New chat" && newMessages.find(m => m.role === "user")
        ? newMessages.find(m => m.role === "user").text.slice(0, 36) + (newMessages.find(m => m.role === "user").text.length > 36 ? "…" : "")
        : c.title;
      return { ...c, messages: newMessages, title };
    }));
  }

  function handleLogout() {
    window.location.href = "http://localhost:8000/auth/logout";
  }

  return (
    <div style={{ height: "100vh", display: "flex", background: "#0C0A09" }}>
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        user={user}
        onLogout={handleLogout}
      />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {activeChat && (
          <Chat
            key={activeChat.id}
            messages={activeChat.messages}
            setMessages={setMessages}
          />
        )}
      </div>
    </div>
  );
}
