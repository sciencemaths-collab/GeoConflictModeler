# GeoConflictModeler

A geography-aware, Monte Carlo adjudicated conflict toy model with a synchronized chess “decision layer”.

- **Monte Carlo = adjudicator (truth prior)**
- **Chess = decision layer (influencer)**
- **Battle Losses = cinematic run snapshot**

Contact: **bessuman.academia@gmail.com**

> Educational/cinematic toy model. Not a real-world predictor.

---

## Folder map

- `app/` Streamlit app (`app.py`) + simulation engine (`engine.py`) + bundled permissive UCI chess engine.
- `api/` FastAPI service for accounts + Stripe subscription + access tokens (optional).
- `site/` Static website (landing + docs + login + pricing + launch).
- `docs/` Deployment notes and environment variables.

---

## Run locally (no paywall)

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Turn on paid access (optional)

This does **not** change the simulator UI. It only blocks access when `GCM_REQUIRE_PAID=1`.

1) Start the API:

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL='postgresql://...'
export JWT_SECRET='change-me'
export APP_TOKEN_SECRET='change-me'
export APP_BASE_URL='https://app.YOURDOMAIN.com'
uvicorn main:app --host 0.0.0.0 --port 9010
```

2) Start Streamlit with the gate enabled:

```bash
cd app
source .venv/bin/activate
export GCM_REQUIRE_PAID=1
export GCM_API_BASE_URL='https://api.YOURDOMAIN.com'
streamlit run app.py
```

3) Serve `site/` (static). The site provides login + subscribe + launch flow.

---

## Deploy on Render (recommended)

Use the included `render.yaml` Blueprint. See `docs/DEPLOY_RENDER.md`.

---

## Coming soon

- Users upload their own datasets (inventories, budgets, geography friction)
- Improved geographic distance (great-circle) and chokepoint friction
- Scenario templates / presets
