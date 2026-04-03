export default function Login() {
  function handleGoogleLogin() {
    window.location.href = "http://localhost:8000/auth/login";
  }

  return (
    <div className="login-root">
      <div className="login-bg">
        <div className="login-grid" aria-hidden="true" />
      </div>

      <div className="login-card">
        <div className="login-logo">
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
            <rect width="36" height="36" rx="10" fill="#D97706" />
            <path d="M10 26L18 10L26 26" stroke="#1C1917" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M13 21H23" stroke="#1C1917" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
          <span className="login-brand">Archivum</span>
        </div>

        <h1 className="login-title">Your knowledge,<br />at your fingertips.</h1>
        <p className="login-sub">Upload documents. Ask questions. Get answers grounded in your repository.</p>

        <button className="google-btn" onClick={handleGoogleLogin}>
          <svg width="18" height="18" viewBox="0 0 18 18">
            <path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z" />
            <path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" />
            <path fill="#FBBC05" d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" />
            <path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z" />
          </svg>
          Continue with Google
        </button>

        <p className="login-footer">
          Your documents stay private. Only you have access.
        </p>
      </div>

      <style>{`
        .login-root {
          min-height: 100vh;
          background: #0C0A09;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: 'DM Sans', 'Segoe UI', sans-serif;
          position: relative;
          overflow: hidden;
        }
        .login-bg {
          position: absolute;
          inset: 0;
          pointer-events: none;
        }
        .login-grid {
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(217,119,6,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(217,119,6,0.06) 1px, transparent 1px);
          background-size: 48px 48px;
          mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black, transparent);
        }
        .login-card {
          position: relative;
          z-index: 10;
          background: #1C1917;
          border: 1px solid rgba(217,119,6,0.2);
          border-radius: 20px;
          padding: 3rem 2.5rem;
          width: 100%;
          max-width: 420px;
          box-shadow: 0 0 80px rgba(217,119,6,0.06);
        }
        .login-logo {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 2.5rem;
        }
        .login-brand {
          font-size: 20px;
          font-weight: 600;
          color: #F5F5F4;
          letter-spacing: -0.02em;
        }
        .login-title {
          font-size: 28px;
          font-weight: 700;
          color: #F5F5F4;
          line-height: 1.25;
          margin: 0 0 12px;
          letter-spacing: -0.03em;
        }
        .login-sub {
          font-size: 14px;
          color: #78716C;
          line-height: 1.6;
          margin: 0 0 2rem;
        }
        .google-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          background: #F5F5F4;
          color: #1C1917;
          border: none;
          border-radius: 10px;
          padding: 13px 20px;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s, transform 0.1s;
          letter-spacing: -0.01em;
        }
        .google-btn:hover { background: #fff; }
        .google-btn:active { transform: scale(0.98); }
        .login-footer {
          text-align: center;
          font-size: 12px;
          color: #57534E;
          margin: 1.5rem 0 0;
        }
      `}</style>
    </div>
  );
}
