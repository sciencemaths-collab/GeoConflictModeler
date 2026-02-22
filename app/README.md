# GeoConflictModeler Streamlit App

Run locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data
Place required files in `app/data/`.

Required:
- `military_capability_merged_scored.xlsx` (sheets: `DATA_Master`, `Name_Mapping`)

## Chess engine
Bundled UCI engine: `geoconflict_uci_engine.py` (MIT).

Optional override:
- `GEOCONFLICT_UCI_ENGINE` or `WARCOLOR_UCI_ENGINE` can point to a custom UCI command.

## Paid gate (optional)
Enabled only when:
- `GCM_REQUIRE_PAID=1`
- `GCM_API_BASE_URL=https://api.yourdomain.com`

Then the app expects `?token=...` in the URL.
