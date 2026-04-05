/**
 * Chat.jsx
 *
 * - Loads persisted message history on mount.
 * - Sends chat_id with every query so the backend persists messages AND
 *   passes the full conversation history to the LLM.
 * - Displays Sources (potentially multiple files) and Performance metrics
 *   as collapsible panels under each assistant message.
 */

import { useState, useEffect, useRef } from "react";
import { getMessages, queryLLM } from "../api";

// ─── Shared primitives ────────────────────────────────────────────────────────

function Spinner({ size = 14 }) {
  return (
    <span style={{
      display: "inline-block", width: size, height: size,
      border: "2px solid rgba(217,119,6,0.25)", borderTopColor: "#D97706",
      borderRadius: "50%", animation: "spin 0.7s linear infinite",
      verticalAlign: "middle",
    }} />
  );
}

function CollapsePanel({ label, count, accentColor = "#D97706", children }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: 6 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          background: "none", border: "none", cursor: "pointer",
          color: accentColor, fontSize: 11, fontWeight: 600,
          letterSpacing: "0.04em", padding: 0,
          display: "flex", alignItems: "center", gap: 5,
        }}
      >
        <span style={{ fontSize: 8 }}>{open ? "▼" : "▶"}</span>
        {label}
        {count !== undefined && (
          <span style={{
            background: "rgba(217,119,6,0.15)", borderRadius: 10,
            padding: "1px 6px", fontSize: 10, fontWeight: 700,
          }}>
            {count}
          </span>
        )}
      </button>
      {open && (
        <div style={{
          marginTop: 8, borderLeft: `2px solid ${accentColor}`,
          paddingLeft: 10,
        }}>
          {children}
        </div>
      )}
    </div>
  );
}

// ─── Sources panel — supports multiple files ──────────────────────────────────

function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <CollapsePanel label="Sources" count={sources.length} accentColor="#D97706">
      {sources.map((s, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <div style={{
            display: "flex", alignItems: "center", gap: 6, marginBottom: 3,
          }}>
            <span style={{ fontSize: 13 }}>📄</span>
            <span style={{ fontSize: 12, fontWeight: 700, color: "#FCD34D" }}>
              {s.filename}
            </span>
          </div>
          {s.snippet && (
            <div style={{
              fontSize: 11, color: "rgba(231,229,228,0.5)",
              fontStyle: "italic", lineHeight: 1.55,
              background: "rgba(255,255,255,0.03)",
              borderRadius: 6, padding: "5px 8px",
            }}>
              "{s.snippet.slice(0, 240)}{s.snippet.length > 240 ? "…" : ""}"
            </div>
          )}
        </div>
      ))}
    </CollapsePanel>
  );
}

// ─── Eval / latency panel ─────────────────────────────────────────────────────

function EvalPanel({ evalData }) {
  if (!evalData || Object.keys(evalData).length === 0) return null;
  const lat = evalData.latency || {};
  const rows = [
    ["Model", evalData.model],
    ["Intent", evalData.intent],
    ["Context", evalData.context_chars != null ? `${evalData.context_chars} chars` : null],
    ["Intent latency", lat.intent_classification_s != null ? `${lat.intent_classification_s}s` : null],
    ["Retrieval", lat.retrieval_s != null ? `${lat.retrieval_s}s` : null],
    ["Generation", lat.generation_s != null ? `${lat.generation_s}s` : null],
    ["Total", lat.total_s != null ? `${lat.total_s}s` : null],
  ].filter(([, v]) => v != null);

  return (
    <CollapsePanel label="Performance" accentColor="#6366F1">
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: "flex", gap: 8, fontSize: 11 }}>
            <span style={{ color: "#A5B4FC", width: 90, flexShrink: 0 }}>{k}</span>
            <span style={{ color: "rgba(231,229,228,0.6)" }}>{v}</span>
          </div>
        ))}
      </div>
    </CollapsePanel>
  );
}

// ─── Message bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex", justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 20, gap: 10, alignItems: "flex-start",
    }}>
      {!isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: "50%",
          background: "linear-gradient(135deg,#D97706,#92400E)",
          flexShrink: 0, display: "flex", alignItems: "center",
          justifyContent: "center", fontSize: 13, marginTop: 2,
        }}>A</div>
      )}

      <div style={{ maxWidth: "72%", minWidth: 60 }}>
        <div style={{
          background: isUser ? "#292524" : "#1C1917",
          border: isUser
            ? "1px solid rgba(217,119,6,0.25)"
            : "1px solid rgba(255,255,255,0.06)",
          borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
          padding: "10px 14px",
          fontSize: 14, lineHeight: 1.65, color: "#E7E5E4",
          whiteSpace: "pre-wrap", wordBreak: "break-word",
        }}>
          {msg.text}
        </div>

        {!isUser && (
          <div style={{ paddingLeft: 4, paddingTop: 3 }}>
            <SourcesPanel sources={msg.sources} />
            <EvalPanel evalData={msg.eval} />
          </div>
        )}
      </div>

      {isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: "50%",
          background: "#292524", border: "1px solid rgba(217,119,6,0.3)",
          flexShrink: 0, display: "flex", alignItems: "center",
          justifyContent: "center", fontSize: 13, color: "#D97706", marginTop: 2,
        }}>U</div>
      )}
    </div>
  );
}

// ─── Suggested prompts ────────────────────────────────────────────────────────

const SUGGESTED = [
  "Summarise my uploaded documents",
  "What are the key findings?",
  "Compare topics across my files",
  "Find statistics in my data",
];

// ─── Main component ───────────────────────────────────────────────────────────

const MODELS = ["gemini", "llama", "deepseek"];

export default function Chat({ chatId, onTitleUpdate }) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [input, setInput] = useState("");
  const [model, setModel] = useState("gemini");

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // Load persisted history for this chat
  useEffect(() => {
    async function load() {
      setLoading(true);
      const data = await getMessages(chatId);
      setMessages(data);
      setLoading(false);
    }
    load();
  }, [chatId]);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, [input]);

  async function handleSend(text = input) {
    const prompt = text.trim();
    if (!prompt || sending) return;
    setInput("");

    // Optimistic user bubble
    const tmpUser = { id: `tmp-u-${Date.now()}`, role: "user", text: prompt, sources: [], eval: {} };
    setMessages((prev) => [...prev, tmpUser]);
    setSending(true);

    try {
      const result = await queryLLM(prompt, model, chatId);

      const assistantMsg = {
        id: `tmp-a-${Date.now()}`,
        role: "assistant",
        text: result.answer ?? "No response.",
        sources: result.sources ?? [],
        eval: result.eval ?? {},
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // If the backend auto-titled the chat, notify parent
      if (result._chat_title) {
        onTitleUpdate?.(chatId, result._chat_title);
      }
    } catch {
      setMessages((prev) => [...prev, {
        id: `err-${Date.now()}`, role: "assistant",
        text: "Something went wrong. Please try again.",
        sources: [], eval: {},
      }]);
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const isEmpty = !loading && messages.length === 0;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100%", minWidth: 0 }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Message list */}
      <div style={{ flex: 1, overflowY: "auto", padding: "28px 0", display: "flex", flexDirection: "column" }}>
        <div style={{ maxWidth: 720, width: "100%", margin: "0 auto", padding: "0 20px" }}>

          {loading && (
            <div style={{ textAlign: "center", marginTop: 60, color: "rgba(231,229,228,0.3)", fontSize: 13 }}>
              <Spinner /> &nbsp;Loading conversation…
            </div>
          )}

          {isEmpty && (
            <div style={{ textAlign: "center", marginTop: 60 }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>📚</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: "#E7E5E4", marginBottom: 6 }}>Archivum</div>
              <div style={{ fontSize: 14, color: "rgba(231,229,228,0.4)", marginBottom: 32 }}>
                Ask anything about your documents
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
                {SUGGESTED.map((s) => (
                  <button key={s} onClick={() => handleSend(s)} style={{
                    background: "#1C1917", border: "1px solid rgba(217,119,6,0.25)",
                    borderRadius: 20, padding: "7px 14px", fontSize: 13,
                    color: "#D4C5A0", cursor: "pointer",
                  }}>{s}</button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => <MessageBubble key={msg.id} msg={msg} />)}

          {sending && (
            <div style={{ display: "flex", gap: 10, alignItems: "flex-start", marginBottom: 18 }}>
              <div style={{
                width: 28, height: 28, borderRadius: "50%",
                background: "linear-gradient(135deg,#D97706,#92400E)",
                flexShrink: 0, display: "flex", alignItems: "center",
                justifyContent: "center", fontSize: 13,
              }}>A</div>
              <div style={{
                background: "#1C1917", border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: "18px 18px 18px 4px", padding: "10px 16px",
                color: "rgba(231,229,228,0.45)", fontSize: 13,
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <Spinner /> Thinking…
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar */}
      <div style={{
        borderTop: "1px solid rgba(255,255,255,0.07)",
        padding: "14px 20px 18px", background: "#0C0A09",
      }}>
        <div style={{
          maxWidth: 720, margin: "0 auto", background: "#1C1917",
          border: "1px solid rgba(255,255,255,0.1)", borderRadius: 14,
          display: "flex", flexDirection: "column", overflow: "hidden",
        }}>
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your documents…"
            style={{
              background: "transparent", border: "none", outline: "none",
              resize: "none", color: "#E7E5E4", fontSize: 14, lineHeight: 1.6,
              padding: "12px 14px 4px", fontFamily: "inherit", overflow: "hidden",
            }}
          />
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "6px 10px 10px",
          }}>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{
                background: "#292524", border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 8, color: "#A8A29E", fontSize: 12,
                padding: "4px 8px", cursor: "pointer", outline: "none",
              }}
            >
              {MODELS.map((m) => (
                <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>
              ))}
            </select>

            <button
              onClick={() => handleSend()}
              disabled={!input.trim() || sending}
              style={{
                background: input.trim() && !sending
                  ? "linear-gradient(135deg,#D97706,#B45309)"
                  : "#292524",
                border: "none", borderRadius: 10,
                width: 34, height: 34,
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: input.trim() && !sending ? "pointer" : "default",
                color: input.trim() && !sending ? "#fff" : "rgba(255,255,255,0.25)",
                fontSize: 16, transition: "background 0.2s",
              }}
            >↑</button>
          </div>
        </div>
        <div style={{
          textAlign: "center", marginTop: 8, fontSize: 11,
          color: "rgba(231,229,228,0.25)",
        }}>
          Shift+Enter for newline · Enter to send
        </div>
      </div>
    </div>
  );
}