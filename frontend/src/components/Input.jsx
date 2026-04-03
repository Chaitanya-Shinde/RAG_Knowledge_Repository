import { useRef, useState } from "react";
import { queryLLM } from "../api";

export default function Input({ messages, setMessages }) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("gemini");
  const textareaRef = useRef(null);

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  async function send() {
    if (!text.trim() || loading) return;
    const userMsg = { role: "user", text: text.trim() };
    setMessages(m => [...m, userMsg]);
    setLoading(true);
    setText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    try {
      const res = await queryLLM(text.trim(), model);
      setMessages(m => [...m, { role: "bot", text: res.answer, sources: res.sources }]);
    } catch {
      setMessages(m => [...m, { role: "bot", text: "Something went wrong. Please try again.", sources: [] }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="input-shell">
      <div className="input-box">
        <textarea
          ref={textareaRef}
          className="input-textarea"
          value={text}
          onChange={e => { setText(e.target.value); autoResize(); }}
          onKeyDown={handleKeyDown}
          placeholder="Ask something about your documents…"
          rows={1}
          disabled={loading}
        />
        <div className="input-actions">
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="model-select"
          >
            <option value="gemini">Gemini</option>
            <option value="ollama">Llama 3.2</option>
            <option value="deepseek">DeepSeek R1</option>
          </select>

          <button className={`send-btn ${loading ? "loading" : ""}`} onClick={send} disabled={loading || !text.trim()}>
            {loading ? (
              <span className="send-spinner" />
            ) : (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="7" y1="13" x2="7" y2="1" />
                <polyline points="2 6 7 1 12 6" />
              </svg>
            )}
          </button>
        </div>
      </div>
      <p className="input-hint">Enter to send · Shift+Enter for new line</p>

      <style>{`
        .input-shell {
          padding: 12px 20px 16px;
          border-top: 1px solid rgba(255,255,255,0.05);
          background: #0C0A09;
          font-family: 'DM Sans', 'Segoe UI', sans-serif;
        }
        .input-box {
          background: #1C1917;
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 12px;
          display: flex;
          flex-direction: column;
          transition: border-color 0.15s;
        }
        .input-box:focus-within { border-color: rgba(217,119,6,0.4); }
        .input-textarea {
          background: none;
          border: none;
          outline: none;
          resize: none;
          color: #E7E5E4;
          font-size: 14px;
          line-height: 1.6;
          padding: 12px 14px 8px;
          font-family: inherit;
          min-height: 44px;
          max-height: 160px;
          overflow-y: auto;
        }
        .input-textarea::placeholder { color: #44403C; }
        .input-textarea::-webkit-scrollbar { width: 4px; }
        .input-textarea::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
        .input-actions {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 10px 10px;
        }
        .model-select {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 6px;
          color: #78716C;
          font-size: 12px;
          font-weight: 500;
          padding: 4px 8px;
          cursor: pointer;
          outline: none;
          font-family: inherit;
        }
        .model-select:focus { border-color: rgba(217,119,6,0.3); }
        .send-btn {
          width: 32px; height: 32px;
          border-radius: 8px;
          border: none;
          background: #D97706;
          color: #1C1917;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: background 0.15s, transform 0.1s, opacity 0.15s;
          flex-shrink: 0;
        }
        .send-btn:hover:not(:disabled) { background: #F59E0B; }
        .send-btn:active:not(:disabled) { transform: scale(0.94); }
        .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .send-spinner {
          width: 12px; height: 12px;
          border: 1.5px solid rgba(28,25,23,0.3);
          border-top-color: #1C1917;
          border-radius: 50%;
          animation: ispin 0.7s linear infinite;
        }
        @keyframes ispin { to { transform: rotate(360deg); } }
        .input-hint {
          font-size: 11px;
          color: #44403C;
          text-align: center;
          margin: 8px 0 0;
        }
      `}</style>
    </div>
  );
}
