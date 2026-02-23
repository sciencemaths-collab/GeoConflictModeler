// GeoConflictModeler site JS (no frameworks)
// Works on Render: geoconflictmodeler-site.onrender.com -> geoconflictmodeler-api.onrender.com -> geoconflictmodeler-app.onrender.com

(function(){
  "use strict";

  // -----------------------------
  // Helpers
  // -----------------------------
  function $(id){ return document.getElementById(id); }
  function show(el, yes=true){ if(!el) return; el.style.display = yes ? "block" : "none"; }
  function setText(idOrEl, txt){
    const el = typeof idOrEl === "string" ? $(idOrEl) : idOrEl;
    if(!el) return;
    el.textContent = (txt ?? "").toString();
  }
  function setHTML(idOrEl, html){
    const el = typeof idOrEl === "string" ? $(idOrEl) : idOrEl;
    if(!el) return;
    el.innerHTML = html ?? "";
  }
  function normalizeUrl(u){ return (u || "").replace(/\/$/, ""); }

  // -----------------------------
  // API Base Detection
  // -----------------------------
  function apiBase(){
    // Preferred: user sets it once in localStorage after you bind domains.
    const saved = localStorage.getItem("gcm_api_base");
    if(saved) return normalizeUrl(saved);

    // Allow query override: ?api=https://...
    const qs = new URLSearchParams(location.search);
    const qapi = qs.get("api");
    if(qapi){
      localStorage.setItem("gcm_api_base", qapi);
      return normalizeUrl(qapi);
    }

    // Local dev
    if(location.hostname === "localhost" || location.hostname === "127.0.0.1"){
      return "http://localhost:9010";
    }

    // Render auto-detect:
    // geoconflictmodeler-site.onrender.com -> geoconflictmodeler-api.onrender.com
    if(location.hostname.endsWith(".onrender.com")){
      if(location.hostname.includes("-site")){
        return `https://${location.hostname.replace("-site", "-api")}`;
      }
      // fallback to your known API (safe)
      return "https://geoconflictmodeler-api.onrender.com";
    }

    // Custom domain case (later)
    // You can set window.GCM_API_BASE in HTML or localStorage gcm_api_base.
    return normalizeUrl(window.GCM_API_BASE || "https://geoconflictmodeler-api.onrender.com");
  }

  function appBase(){
    // Preferred: explicit
    if(window.GCM_APP_BASE) return normalizeUrl(window.GCM_APP_BASE);

    // Render auto-detect:
    // geoconflictmodeler-site.onrender.com -> geoconflictmodeler-app.onrender.com
    if(location.hostname.endsWith(".onrender.com") && location.hostname.includes("-site")){
      return `https://${location.hostname.replace("-site", "-app")}`;
    }
    // fallback
    return "https://geoconflictmodeler-app.onrender.com";
  }

  // -----------------------------
  // Auth token helpers
  // -----------------------------
  function getAuthToken(){ return localStorage.getItem("gcm_auth_token") || ""; }
  function setAuthToken(tok){ if(tok) localStorage.setItem("gcm_auth_token", tok); }
  function clearAuth(){
    localStorage.removeItem("gcm_auth_token");
    localStorage.removeItem("gcm_user_email");
  }
  function setUserEmail(email){ if(email) localStorage.setItem("gcm_user_email", email); }
  function getUserEmail(){ return localStorage.getItem("gcm_user_email") || ""; }

  // -----------------------------
  // API wrapper
  // -----------------------------
  async function api(path, opts={}){
    const headers = Object.assign({"content-type":"application/json"}, opts.headers || {});
    const tok = getAuthToken();
    if(tok) headers["authorization"] = `Bearer ${tok}`;

    const url = apiBase() + path;
    const res = await fetch(url, Object.assign({}, opts, {headers}));
    const text = await res.text();
    let body = null;
    try{ body = text ? JSON.parse(text) : null; }catch{ body = {raw:text}; }

    if(!res.ok){
      const msg = (body && (body.detail || body.error || body.message)) || `Request failed (${res.status})`;
      const e = new Error(msg);
      e.status = res.status;
      e.body = body;
      throw e;
    }
    return body;
  }

  async function postJSON(path, payload){
    return api(path, {method:"POST", body: JSON.stringify(payload || {})});
  }

  async function tryFirst(paths, payload, method="POST"){
    let lastErr = null;
    for(const p of paths){
      try{
        if(method === "POST") return await postJSON(p, payload);
        return await api(p, {method});
      }catch(e){
        lastErr = e;
      }
    }
    throw lastErr || new Error("All endpoints failed");
  }

  // -----------------------------
  // Endpoint candidates (robust to small route differences)
  // -----------------------------
  const EP = {
    health: ["/health", "/ping"],
    me: ["/me", "/auth/me", "/account/me", "/user/me"],
    login: ["/auth/login", "/login", "/account/login"],
    register: ["/auth/register", "/register", "/auth/signup", "/signup", "/account/register"],
    checkout: ["/billing/create-checkout-session", "/create-checkout-session", "/billing/checkout", "/checkout/session"],
    access: ["/access/token", "/billing/access-token", "/app/token", "/token", "/access"],
    logout: ["/auth/logout", "/logout"]
  };

  // -----------------------------
  // UI state updates (no UI changes required)
  // -----------------------------
  function setStatus(msg){
    // Common ids used across pages
    setText("status", msg);
    setText("account_status", msg);
    setText("acct_status", msg);
  }

  function setError(msg){
    setText("error", msg);
    setText("msg", msg);
    setText("message", msg);
    const el = $("error_box") || $("msg_box");
    if(el) el.textContent = msg;
  }

  function clearError(){ setError(""); }

  // -----------------------------
  // Auth flows
  // -----------------------------
  function extractToken(resp){
    // Support common response shapes
    return (resp && (resp.token || resp.access_token || resp.jwt || resp.app_token)) || "";
  }

  async function loginOrRegister(mode){
    clearError();

    const email = ($("email") && $("email").value.trim()) || getUserEmail();
    const password = ($("password") && $("password").value) || "";

    if(!email){
      setError("Enter your email.");
      return;
    }
    // Password is optional depending on how your API is implemented.
    // If your API requires password and it's missing, you'll get a clear error.
    const payload = { email, password };

    try{
      const resp = await tryFirst(mode === "register" ? EP.register : EP.login, payload, "POST");
      const tok = extractToken(resp);
      if(tok) setAuthToken(tok);
      setUserEmail(email);

      // Optional: update status if page has it
      await refreshAccountStatus();

      // Redirect hints: if there's a next= query param, go there
      const qs = new URLSearchParams(location.search);
      const next = qs.get("next");
      if(next){
        location.href = next;
        return;
      }

      // Default redirect: pricing page if exists, else launch
      if(location.pathname.endsWith("/login.html") || location.pathname.endsWith("/signup.html")){
        location.href = "pricing.html";
      }
    }catch(e){
      setError(e.message || "Login failed.");
    }
  }

  async function logout(){
    clearError();
    try{
      // best effort logout on API
      await tryFirst(EP.logout, {}, "POST").catch(()=>{});
    }finally{
      clearAuth();
      setStatus("Not logged in.");
      // If there is a visible button state, update it
      toggleAuthUI();
    }
  }

  // -----------------------------
  // Subscription / access status
  // -----------------------------
  async function refreshAccountStatus(){
    const tok = getAuthToken();
    if(!tok){
      setStatus("Not logged in.");
      toggleAuthUI();
      return { logged_in:false };
    }

    // Try /me first (some APIs put subscription flags there)
    try{
      const me = await tryFirst(EP.me, null, "GET");
      // Try to infer status fields
      const active =
        !!(me && (me.subscription_active || me.is_active || (me.subscription && me.subscription.active)));

      if(active){
        setStatus("Logged in. Subscription: Active.");
      }else{
        // Might not include subscription info; fallback to access endpoint probe
        setStatus("Logged in.");
      }
      toggleAuthUI(active);
      return { logged_in:true, active };
    }catch(_e){
      // If /me fails, still show logged in (token exists), but unknown state
      setStatus("Logged in (status unknown).");
      toggleAuthUI(false);
      return { logged_in:true, active:false };
    }
  }

  function toggleAuthUI(active=false){
    // These ids are optional. If not present, nothing breaks.
    const loggedIn = !!getAuthToken();

    // Buttons/sections commonly used
    show($("btn_login"), !loggedIn);
    show($("btn_register"), !loggedIn);
    show($("btn_logout"), loggedIn);

    show($("login_block"), !loggedIn);
    show($("loggedin_block"), loggedIn);

    // Launch button: show if logged in; if you want stricter gating, only show when active=true
    show($("btn_launch"), loggedIn);
    show($("launch_block"), loggedIn);

    // Subscribe block: show if logged in but not active (best effort)
    if(loggedIn){
      show($("subscribe_block"), !active);
      show($("already_active_block"), !!active);
    }else{
      show($("subscribe_block"), false);
      show($("already_active_block"), false);
    }
  }

  // -----------------------------
  // Checkout (Stripe)
  // -----------------------------
  async function startCheckout(){
    clearError();
    if(!getAuthToken()){
      setError("You must be logged in before subscribing.");
      return;
    }

    try{
      const resp = await tryFirst(EP.checkout, {
        // API may ignore these; they help some implementations
        success_url: window.location.origin + "/success.html",
        cancel_url: window.location.origin + "/cancel.html"
      }, "POST");

      // Common: {url: "..."} or {checkout_url: "..."}
      const url = (resp && (resp.url || resp.checkout_url || resp.redirect_url)) || "";
      if(!url){
        throw new Error("Checkout URL missing from API response.");
      }
      window.location.href = url;
    }catch(e){
      setError(e.message || "Failed to start checkout.");
    }
  }

  // -----------------------------
  // Launch app (mint token + redirect)
  // -----------------------------
  async function launchApp(){
    clearError();
    if(!getAuthToken()){
      setError("Not logged in. Please log in first.");
      return;
    }

    try{
      // Ask API for a short-lived app token (preferred)
      const resp = await tryFirst(EP.access, {
        // Some APIs use this to bind redirect origin
        app_base: appBase()
      }, "POST");

      const tok = extractToken(resp) || (resp && resp.token) || "";
      const target = tok
        ? `${appBase()}/?token=${encodeURIComponent(tok)}`
        : `${appBase()}/`;

      window.location.href = target;
    }catch(e){
      // If API doesn't support minting yet, still allow open launch (when paywall off)
      window.location.href = `${appBase()}/`;
    }
  }

  // -----------------------------
  // Page wiring
  // -----------------------------
  function bind(){
    // Wire buttons if they exist (no UI change required)
    const bl = $("btn_login"); if(bl) bl.addEventListener("click", ()=>loginOrRegister("login"));
    const br = $("btn_register"); if(br) br.addEventListener("click", ()=>loginOrRegister("register"));
    const bo = $("btn_logout"); if(bo) bo.addEventListener("click", logout);

    const sub = $("btn_subscribe"); if(sub) sub.addEventListener("click", startCheckout);
    const launch = $("btn_launch"); if(launch) launch.addEventListener("click", launchApp);

    // Form submit support (if your login page uses a form)
    const form = $("auth_form");
    if(form){
      form.addEventListener("submit", (ev)=>{
        ev.preventDefault();
        const mode = (form.dataset && form.dataset.mode) || "login";
        loginOrRegister(mode);
      });
    }

    // Auto-refresh status on pages that likely need it
    refreshAccountStatus().catch(()=>{});

    // Display API base somewhere if you have a debug slot
    setText("api_base", apiBase());
    setText("app_base", appBase());
  }

  // Run
  document.addEventListener("DOMContentLoaded", bind);

  // Expose a couple helpers for quick debugging in console (optional)
  window.GCM = {
    apiBase,
    appBase,
    setApiBase: (u)=>{ localStorage.setItem("gcm_api_base", u); },
    clearApiBase: ()=>{ localStorage.removeItem("gcm_api_base"); },
    logout
  };
})();
