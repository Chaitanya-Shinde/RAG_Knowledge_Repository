export default function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="msg-row msg-row-user">
        <div className="msg-bubble msg-user">
          {message.text}
        </div>
        <style>{`
          .msg-row { display: flex; margin-bottom: 4px; }
          .msg-row-user { justify-content: flex-end; }
          .msg-row-bot { justify-content: flex-start; }
          .msg-bubble { max-width: 72%; padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.6; word-break: break-word; font-family: 'DM Sans', 'Segoe UI', sans-serif; }
          .msg-user { background: #D97706; color: #1C1917; border-bottom-right-radius: 4px; font-weight: 500; }
          .msg-bot { background: #1C1917; border: 1px solid rgba(255,255,255,0.07); color: #E7E5E4; border-bottom-left-radius: 4px; }
          .msg-sources { margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); }
          .msg-sources-label { font-size: 11px; color: #57534E; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
          .msg-source-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #78716C; margin-top: 3px; }
        `}</style>
      </div>
    );
  }

  return (
    <div className="msg-row msg-row-bot">
      <div className="msg-bubble msg-bot">
        <div style={{ whiteSpace: "pre-wrap" }}>{message.text}</div>
        {message.sources && message.sources.length > 0 && (
          <div className="msg-sources">
            <div className="msg-sources-label">Sources</div>
            {message.sources.map((s, i) => (
              <div key={i} className="msg-source-item">
                <svg width="11" height="13" viewBox="0 0 11 13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M6.5 1H1.5A1.5 1.5 0 0 0 0 2.5v8A1.5 1.5 0 0 0 1.5 12h8A1.5 1.5 0 0 0 11 10.5V5.5L6.5 1z" />
                  <polyline points="6.5 1 6.5 5.5 11 5.5" />
                </svg>
                {s.filename}
              </div>
            ))}
          </div>
        )}
      </div>
      <style>{`
        .msg-row { display: flex; margin-bottom: 4px; }
        .msg-row-user { justify-content: flex-end; }
        .msg-row-bot { justify-content: flex-start; }
        .msg-bubble { max-width: 72%; padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.6; word-break: break-word; font-family: 'DM Sans', 'Segoe UI', sans-serif; }
        .msg-user { background: #D97706; color: #1C1917; border-bottom-right-radius: 4px; font-weight: 500; }
        .msg-bot { background: #1C1917; border: 1px solid rgba(255,255,255,0.07); color: #E7E5E4; border-bottom-left-radius: 4px; }
        .msg-sources { margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); }
        .msg-sources-label { font-size: 11px; color: #57534E; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
        .msg-source-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #78716C; margin-top: 3px; }
      `}</style>
    </div>
  );
}
