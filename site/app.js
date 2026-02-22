// GeoConflictModeler site JS (no frameworks)

function apiBase(){
  // Preferred: user sets it once in localStorage after you bind domains.
  const saved = localStorage.getItem('gcm_api_base');
  if(saved) return saved.replace(/\/$/, '');

  // Sensible defaults
  if(location.hostname === 'localhost' || location.hostname === '127.0.0.1'){
    return 'http://localhost:9010';
  }

  // If you set a custom domain, this should be your API subdomain.
  return (window.GCM_API_BASE || 'https://api.geoconflictmodeler.com').replace(/\/$/, '');
}

function getAuthToken(){
  return localStorage.getItem('gcm_auth_token') || '';
}

function setAuthToken(tok){
  if(tok) localStorage.setItem('gcm_auth_token', tok);
}

function clearAuth(){
  localStorage.removeItem('gcm_auth_token');
}

async function api(path, opts={}){
  const headers = Object.assign({'content-type':'application/json'}, opts.headers || {});
  const tok = getAuthToken();
  if(tok) headers['authorization'] = `Bearer ${tok}`;

  const res = await fetch(apiBase()+path, Object.assign({}, opts, {headers}));
  const text = await res.text();
  let body = null;
  try{ body = text ? JSON.parse(text) : null; }catch{ body = {raw:text}; }
  if(!res.ok){
    const msg = (body && (body.detail || body.error)) || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return body;
}

function $(id){ return document.getElementById(id); }

function show(el, yes=true){ if(!el) return; el.style.display = yes ? 'block' : 'none'; }

