import { useEffect, useRef } from "react";
import Message from "./Message";
import Input from "./Input";

export default function Chat({ messages, setMessages }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-root">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <svg width="32" height="32" viewBox="0 0 36 36" fill="none">
                <rect width="36" height="36" rx="10" fill="rgba(217,119,6,0.15)" />
                <path d="M10 26L18 10L26 26" stroke="#D97706" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M13 21H23" stroke="#D97706" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
            <h2 className="chat-empty-title">Ask your repository</h2>
            <p className="chat-empty-sub">Ask any question and get answers grounded in your uploaded documents.</p>
            <div className="chat-suggestions">
              {["Summarize the key points", "What are the main topics?", "Explain the methodology"].map(s => (
                <button key={s} className="suggestion-chip" onClick={() => setMessages(m => [...m, { role: "user", text: s }])}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => <Message key={i} message={m} />)
        )}
        <div ref={bottomRef} />
      </div>

      <Input messages={messages} setMessages={setMessages} />

      <style>{`
        .chat-root {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: #0C0A09;
          font-family: 'DM Sans', 'Segoe UI', sans-serif;
        }
        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 32px 24px 16px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .chat-messages::-webkit-scrollbar { width: 4px; }
        .chat-messages::-webkit-scrollbar-track { background: transparent; }
        .chat-messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
        .chat-empty {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 60px 24px;
          gap: 12px;
        }
        .chat-empty-icon { margin-bottom: 4px; }
        .chat-empty-title {
          font-size: 22px;
          font-weight: 700;
          color: #F5F5F4;
          letter-spacing: -0.03em;
          margin: 0;
        }
        .chat-empty-sub {
          font-size: 14px;
          color: #57534E;
          line-height: 1.6;
          max-width: 340px;
          margin: 0;
        }
        .chat-suggestions {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          justify-content: center;
          margin-top: 16px;
        }
        .suggestion-chip {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 20px;
          color: #A8A29E;
          font-size: 13px;
          padding: 7px 14px;
          cursor: pointer;
          transition: background 0.15s, color 0.15s, border-color 0.15s;
          font-family: inherit;
        }
        .suggestion-chip:hover {
          background: rgba(217,119,6,0.1);
          border-color: rgba(217,119,6,0.25);
          color: #D97706;
        }
      `}</style>
    </div>
  );
}
