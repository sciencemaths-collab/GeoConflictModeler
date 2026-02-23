// GeoConflictModeler site JS (no frameworks)
// Works on Render: geoconflictmodeler-site.onrender.com -> geoconflictmodeler-api.onrender.com -> geoconflictmodeler-app.onrender.com

(function () {
  "use strict";

  // -----------------------------
  // Helpers
  // -----------------------------
  function $(id) { return document.getElementById(id); }
  function show(el, yes = true) { if (!el) return; el.style.display = yes ? "block" : "none"; }
  function setText(idOrEl, txt) {
    const el = typeof idOrEl === "string" ? $(idOrEl) : idOrEl;
    if (!el) return;
    el.textContent = (txt ?? "").toString();
  }
  function normalizeUrl(u) { return (u || "").replace(/\/$/, ""); }

  // -----------------------------
  // Render-aware base URLs
  // -----------------------------
  function apiBase() {
    const saved = localStorage.getItem("gcm_api_base");
    if (saved) return normalizeUrl(saved);

    // query override ?api=https://...
    try {
      const qs = new URLSearchParams(location.search);
      const qapi = qs.get("api");
      if (qapi) {
        localStorage.setItem("gcm_api_base", qapi);
        return normalizeUrl(qapi);
      }
    } catch { }

    // local dev
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
      return "http://localhost:9010";
    }

    // Render auto-detect:
    // geoconflictmodeler-site.onrender.com -> geoconflictmodeler-api.onrender.com
    if (location.hostname.endsWith(".onrender.com")) {
      if (location.hostname.includes("-site")) {
        return normalizeUrl(`https://${location.hostname.replace("-site", "-api")}`);
      }
      return "https://geoconflictmodeler-api.onrender.com";
    }

    // custom domain later
    return normalizeUrl(window.GCM_API_BASE || "https://geoconflictmodeler-api.onrender.com");
  }

  function appBase() {
    if (window.GCM_APP_BASE) return normalizeUrl(window.GCM_APP_BASE);

    // Render auto-detect:
    // geoconflictmodeler-site.onrender.com -> geoconflictmodeler-app.onrender.com
    if (location.hostname.endsWith(".onrender.com") && location.hostname.includes("-site")) {
      return normalizeUrl(`https://${location.hostname.replace("-site", "-app")}`);
    }
    return "https://geoconflictmodeler-app.onrender.com";
  }

  // -----------------------------
  // Auth token helpers
  // -----------------------------
  function getAuthToken() { return localStorage.getItem("gcm_auth_token") || ""; }
  function setAuthToken(tok) { if (tok) localStorage.setItem("gcm_auth_token", tok); }
  function clearAuth() {
    localStorage.removeItem("gcm_auth_token");
    localStorage.removeItem("gcm_user_email");
  }
  function setUserEmail(email) { if (email) localStorage.setItem("gcm_user_email", email); }
  function getUserEmail() { return localStorage.getItem("gcm_user_email") || ""; }

  // -----------------------------
  // Endpoint map (matches your API docs)
  // -----------------------------
  const EP = {
    health: ["/health"],
    register: ["/auth/register"],
    login: ["/auth/login"],
    me: ["/me"],
    accessStatus: ["/access/status"],
    accessMint: ["/access/mint"],
    accessVerify: ["/access/verify"],
    checkout: ["/billing/create-checkout"],
    // webhook is server-to-server only:
    // /billing/webhook
  };

  // -----------------------------
  // UI state
  // -----------------------------
  function setStatus(msg) {
    setText("status", msg);
    setText("account_status", msg);
    setText("acct_status", msg);
  }

  function setError(msg) {
    setText("error", msg);
    setText("msg", msg);
    setText("message", msg);
    const el = $("error_box") || $("msg_box");
    if (el) el.textContent = msg;
  }
  function clearError() { setError(""); }

  // -----------------------------
  // API wrapper (CORS-safe)
  // KEY FIX: do NOT set content-type on GET requests
  // -----------------------------
  async function api(path, opts = {}) {
    const method = (opts.method || "GET").toUpperCase();
    const headers = Object.assign({ "accept": "application/json" }, opts.headers || {});

    const tok = getAuthToken();
    if (tok) headers["authorization"] = `Bearer ${tok}`;

    // Only send JSON content-type when we actually send a body
    const hasBody = opts.body != null;
    if (hasBody && !headers["content-type"]) {
      headers["content-type"] = "application/json";
    }

    const url = apiBase() + path;

    let res;
    try {
      res = await fetch(url, Object.assign({}, opts, { method, headers }));
    } catch (e) {
      // This is where Firefox shows: "NetworkError when attempting to fetch resource"
      throw new Error(
        `NetworkError: cannot reach API.\n\n` +
        `API base: ${apiBase()}\n` +
        `URL: ${url}\n\n` +
        `Fix:\n` +
        `1) Confirm API is up: ${apiBase()}/health\n` +
        `2) In Render (API service), set SITE_ORIGIN to: https://geoconflictmodeler-site.onrender.com\n` +
        `3) Then redeploy API.\n`
      );
    }

    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; }
    catch { body = { raw: text }; }

    if (!res.ok) {
      const msg = (body && (body.detail || body.error || body.message)) || `Request failed (${res.status})`;
      const err = new Error(msg);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  async function postJSON(path, payload) {
    return api(path, { method: "POST", body: JSON.stringify(payload || {}) });
  }

  async function tryFirst(paths, payload, method = "GET") {
    let lastErr = null;
    for (const p of paths) {
      try {
        if (method === "POST") return await postJSON(p, payload);
        return await api(p, { method });
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error("All endpoints failed");
  }

  // -----------------------------
  // Auth flows
  // -----------------------------
  function extractToken(resp) {
    return (resp && (resp.token || resp.access_token || resp.jwt)) || "";
  }

  async function loginOrRegister(mode) {
    clearError();

    const email = ($("email") && $("email").value.trim()) || getUserEmail();
    const password = ($("password") && $("password").value) || "";

    if (!email) {
      setError("Enter your email.");
      return;
    }

    const payload = { email, password };

    try {
      const resp = await tryFirst(mode === "register" ? EP.register : EP.login, payload, "POST");
      const tok = extractToken(resp);
      if (tok) setAuthToken(tok);
      setUserEmail(email);

      await refreshAccountStatus();

      // Default redirect
      if (location.pathname.endsWith("/login.html") || location.pathname.endsWith("/signup.html")) {
        location.href = "pricing.html";
      }
    } catch (e) {
      setError(e.message || "Login failed.");
    }
  }

  async function logout() {
    clearError();
    clearAuth();
    setStatus("Not logged in.");
    toggleAuthUI(false, false);
  }

  // -----------------------------
  // Subscription / access status
  // -----------------------------
  async function refreshAccountStatus() {
    const tok = getAuthToken();
    if (!tok) {
      setStatus("Not logged in.");
      toggleAuthUI(false, false);
      return { logged_in: false, active: false };
    }

    // Prefer access/status for subscription truth
    try {
      const st = await tryFirst(EP.accessStatus, null, "GET");
      const active = !!(st && (st.active || st.subscription_active || st.is_active));
      setStatus(active ? "Logged in. Subscription: Active." : "Logged in. Subscription: Inactive.");
      toggleAuthUI(true, active);
      return { logged_in: true, active };
    } catch (_e) {
      // fallback: /me (may not contain subscription info)
      try {
        await tryFirst(EP.me, null, "GET");
        setStatus("Logged in.");
        toggleAuthUI(true, false);
        return { logged_in: true, active: false };
      } catch (e2) {
        // token probably invalid
        clearAuth();
        setStatus("Not logged in.");
        toggleAuthUI(false, false);
        return { logged_in: false, active: false };
      }
    }
  }

  function toggleAuthUI(loggedIn, active) {
    show($("btn_login"), !loggedIn);
    show($("btn_register"), !loggedIn);
    show($("btn_logout"), loggedIn);

    show($("login_block"), !loggedIn);
    show($("loggedin_block"), loggedIn);

    show($("btn_launch"), loggedIn);
    show($("launch_block"), loggedIn);

    if (loggedIn) {
      show($("subscribe_block"), !active);
      show($("already_active_block"), !!active);
    } else {
      show($("subscribe_block"), false);
      show($("already_active_block"), false);
    }
  }

  // -----------------------------
  // Stripe Checkout
  // -----------------------------
  async function startCheckout() {
    clearError();
    if (!getAuthToken()) {
      setError("You must be logged in before subscribing.");
      return;
    }

    try {
      const resp = await tryFirst(EP.checkout, {
        success_url: window.location.origin + "/success.html",
        cancel_url: window.location.origin + "/cancel.html",
      }, "POST");

      const url = (resp && (resp.url || resp.checkout_url || resp.redirect_url)) || "";
      if (!url) throw new Error("Checkout URL missing from API response.");
      window.location.href = url;
    } catch (e) {
      setError(e.message || "Failed to start checkout.");
    }
  }

  // -----------------------------
  // Launch App (mint token + redirect)
  // -----------------------------
  async function launchApp() {
    clearError();
    if (!getAuthToken()) {
      setError("Not logged in. Please log in first.");
      return;
    }

    // First: confirm subscription is active
    const st = await refreshAccountStatus();
    if (!st.active) {
      setError("Subscription required. Go to Pricing and subscribe.");
      return;
    }

    try {
      const resp = await tryFirst(EP.accessMint, { app_base: appBase() }, "POST");
      const tok = extractToken(resp) || (resp && resp.token) || "";
      const target = tok
        ? `${appBase()}/?token=${encodeURIComponent(tok)}`
        : `${appBase()}/`;

      window.location.href = target;
    } catch (e) {
      // If mint fails, try direct (only works if paywall is still off)
      window.location.href = `${appBase()}/`;
    }
  }

  // -----------------------------
  // Bind
  // -----------------------------
  function bind() {
    const bl = $("btn_login"); if (bl) bl.addEventListener("click", () => loginOrRegister("login"));
    const br = $("btn_register"); if (br) br.addEventListener("click", () => loginOrRegister("register"));
    const bo = $("btn_logout"); if (bo) bo.addEventListener("click", logout);

    const sub = $("btn_subscribe"); if (sub) sub.addEventListener("click", startCheckout);
    const launch = $("btn_launch"); if (launch) launch.addEventListener("click", launchApp);

    const form = $("auth_form");
    if (form) {
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const mode = (form.dataset && form.dataset.mode) || "login";
        loginOrRegister(mode);
      });
    }

    refreshAccountStatus().catch(() => { });

    setText("api_base", apiBase());
    setText("app_base", appBase());
  }

  document.addEventListener("DOMContentLoaded", bind);

  // Expose for console debug
  window.GCM = {
    apiBase,
    appBase,
    api,
    setApiBase: (u) => { localStorage.setItem("gcm_api_base", u); },
    clearApiBase: () => { localStorage.removeItem("gcm_api_base"); },
    logout
  };
})();
