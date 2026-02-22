# GeoConflictModeler (UI Rebuild)
# Streamlit UI: Scenario Builder (left), Chess + Phase Separation (center/right), Loss Table (bottom)

from __future__ import annotations

import os
import json
import math
import time
import sys
import subprocess
import atexit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from engine import (
    load_data, Scenario, simulate_cinematic,
    monte_carlo_fast, monte_carlo_integrated_fast,
    allocate_losses_to_countries,
    dense_default_cfg, fit_dense_cfg, fit_integrated_cfg,
)


# ----------------------------
# Chess ↔ Monte Carlo fusion helpers (no UI changes)
# ----------------------------

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))

def _cp_to_pwin(cp_white_pov: float, scale: float = 250.0) -> float:
    """Convert centipawns (White POV) into P(White wins vs Black) on [0,1] via a logistic curve."""
    cp = float(np.clip(cp_white_pov, -3000.0, 3000.0))
    return float(_sigmoid(cp / float(scale)))

def _fuse_mc_with_chess(pB: float, pR: float, pD: float, cp_white_pov: Optional[float], lam: float = 0.35) -> Tuple[float,float,float,float,float]:
    """
    3-way fusion:
      - Keep Pdraw from MC as the 'truth' draw channel.
      - Fuse Blue vs Red odds using: logit(Pfinal)=logit(PmcBR)+lam*logit(Pchess)
      - Then re-attach draw mass: (1-Pdraw) split by PfinalBR.

    Returns: (pB_f, pR_f, pD_f, p_final_BR, p_chess_BR)
    """
    pB = float(pB or 0.0); pR = float(pR or 0.0); pD = float(pD or 0.0)
    tot = pB + pR + pD
    if tot > 1e-9:
        pB, pR, pD = pB/tot, pR/tot, pD/tot
    else:
        pB, pR, pD = 0.5, 0.5, 0.0

    # Blue vs Red conditional (ignore draw mass)
    denom = max(1e-9, (pB + pR))
    p_mc_br = float(np.clip(pB / denom, 1e-6, 1.0 - 1e-6))

    cpw = float(cp_white_pov or 0.0)
    p_chess = float(np.clip(_cp_to_pwin(cpw), 1e-6, 1.0 - 1e-6))

    # Reduce chess influence when draw channel is large (keeps chess from 'flipping reality')
    lam_eff = float(lam) * float(np.clip(1.0 - pD, 0.15, 1.0))

    p_final_br = float(_sigmoid(_logit(p_mc_br) + lam_eff * _logit(p_chess)))

    pD_f = pD
    rem = float(max(0.0, 1.0 - pD_f))
    pB_f = rem * p_final_br
    pR_f = rem * (1.0 - p_final_br)
    return float(pB_f), float(pR_f), float(pD_f), float(p_final_br), float(p_chess)

def _winner_label(pB: float, pR: float, pD: float) -> str:
    pB = float(pB or 0.0); pR = float(pR or 0.0); pD = float(pD or 0.0)
    if pD >= 0.34:
        return "Draw"
    if abs(pB - pR) < 0.03:
        return "Draw"
    return "Blue" if pB > pR else "Red"

def _fen_from_cs(stc: ChessState) -> str:
    """Convert internal ChessState -> FEN."""
    rows = []
    for r in range(8):
        empt = 0
        s = []
        for c in range(8):
            p = stc.board[r][c]
            if p == ".":
                empt += 1
            else:
                if empt:
                    s.append(str(empt))
                    empt = 0
                s.append(p)
        if empt:
            s.append(str(empt))
        rows.append("".join(s))
    side = "w" if stc.white else "b"
    castle = stc.castle if stc.castle else "-"
    ep = "-"
    if stc.ep is not None:
        rr, cc = stc.ep
        ep = f"{'abcdefgh'[cc]}{8-rr}"
    return f"{'/'.join(rows)} {side} {castle} {ep} {int(stc.halfmove)} {int(stc.fullmove)}"

def _mv_from_uci(stc: ChessState, uci: str):
    """Find our internal move tuple corresponding to a UCI string."""
    uci = (uci or "").strip()
    for mv in _legal_moves(stc):
        if _uci(mv) == uci:
            return mv
    return None


class _UCIClient:
    """Lightweight UCI bridge (kept internal; no UI changes)."""
    def __init__(self, cmd: List[str]):
        self.cmd = list(cmd)
        self.p = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._handshake()

    def _w(self, s: str) -> None:
        try:
            assert self.p.stdin is not None
            self.p.stdin.write(s + "\n")
            self.p.stdin.flush()
        except Exception:
            pass

    def _r(self) -> str:
        try:
            assert self.p.stdout is not None
            return self.p.stdout.readline()
        except Exception:
            return ""

    def _drain_until(self, token: str, max_lines: int = 5000) -> None:
        for _ in range(max_lines):
            line = self._r()
            if not line:
                break
            if token in line:
                break

    def _handshake(self) -> None:
        self._w("uci")
        self._drain_until("uciok")
        self._w("isready")
        self._drain_until("readyok")

    def newgame(self) -> None:
        self._w("ucinewgame")
        self._w("isready")
        self._drain_until("readyok")

    def setoption(self, name: str, value: str) -> None:
        self._w(f"setoption name {name} value {value}")
        self._w("isready")
        self._drain_until("readyok")

    def bestmove(self, fen: str, depth: int, movetime_ms: int) -> Tuple[str, Optional[int]]:
        """Returns (bestmove_uci, cp_side_to_move) where cp is from side-to-move POV (UCI convention)."""
        self._w(f"position fen {fen}")
        self._w(f"go depth {int(depth)} movetime {int(movetime_ms)}")
        cp = None
        mate = None
        best = "0000"
        # Read until bestmove
        for _ in range(20000):
            line = self._r()
            if not line:
                break
            line = line.strip()
            if line.startswith("info") and "score" in line:
                # parse: ... score cp X ... OR score mate X
                parts = line.split()
                try:
                    si = parts.index("score")
                    typ = parts[si + 1]
                    val = parts[si + 2]
                    if typ == "cp":
                        cp = int(val)
                        mate = None
                    elif typ == "mate":
                        mate = int(val)
                except Exception:
                    pass
            if line.startswith("bestmove"):
                try:
                    best = line.split()[1].strip()
                except Exception:
                    best = "0000"
                break

        if mate is not None:
            # Map mate score to a large cp so logistic saturates but doesn't overflow.
            cp = int(np.clip(12000 * (1 if mate > 0 else -1), -12000, 12000))
        return best, cp

    def close(self) -> None:
        try:
            self._w("quit")
        except Exception:
            pass
        try:
            if self.p.poll() is None:
                self.p.terminate()
        except Exception:
            pass


def _get_uci_client() -> Optional[_UCIClient]:
    """Get or start a UCI engine process (cached in session_state)."""
    eng = st.session_state.get("_uci_client", None)
    if eng is not None:
        try:
            if eng.p is not None and eng.p.poll() is None:
                return eng
        except Exception:
            pass
        st.session_state["_uci_client"] = None

    # Engine command: env override, else python_engine.py in CWD
    cmd_env = os.environ.get("WARCOLOR_UCI_ENGINE", "").strip()
    if cmd_env:
        # allow: "stockfish" or "python -u python_engine.py"
        cmd = cmd_env.split()
    else:
        # prefer local python_engine.py
        cand = None
        for path in ["python_engine.py", os.path.join(os.path.dirname(__file__), "python_engine.py") if "__file__" in globals() else None]:
            if path and os.path.exists(path):
                cand = path
                break
        if cand is None:
            # no engine available
            return None
        cmd = [sys.executable, "-u", cand]

    try:
        eng = _UCIClient(cmd)
        st.session_state["_uci_client"] = eng
        # Ensure cleanup on app exit
        try:
            atexit.register(lambda: eng.close())
        except Exception:
            pass
        return eng
    except Exception:
        st.session_state["_uci_client"] = None
        return None


# ----------------------------
# Page + Theme
# ----------------------------
st.set_page_config(page_title="GeoConflictModeler", layout="wide")

st.markdown("""
<style>
/* ----------------------------
   GeoConflictModeler HUD Skin
   Goal: match reference screenshot as closely as Streamlit allows.
---------------------------- */

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 700px at 65% 25%, rgba(80,120,255,0.11), rgba(0,0,0,0) 55%),
    radial-gradient(900px 550px at 85% 55%, rgba(255,60,60,0.10), rgba(0,0,0,0) 62%),
    linear-gradient(180deg, #05070c, #070b12 30%, #060a10);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}

.block-container { padding: 0.55rem 0.75rem 1.00rem 0.75rem; max-width: 100%; }

/* Hide Streamlit chrome */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
section.main > div { padding-top: 0rem; }

/* --- HUD typography --- */
h1, h2, h3, h4 { letter-spacing: 0.3px; }
small, .small { opacity: 0.82; }

.panel-h { font-size: 1.05rem; font-weight: 850; margin: 0 0 0.35rem 0; }
.panel-sub { font-size: 0.88rem; opacity: 0.72; margin-bottom: 0.60rem; }

/* --- Panel look --- */
.hud-panel {
  background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.028));
  border: 1px solid rgba(255,255,255,0.105);
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.50);
  backdrop-filter: blur(8px);
}
.hud-title { font-size: 1.25rem; font-weight: 800; opacity: 0.95; margin-bottom: 0.35rem; }
.hud-sub   { font-size: 0.92rem; opacity: 0.78; margin-bottom: 0.65rem; }

.hud-frame {
  background: rgba(0,0,0,0.28);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 10px;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05), 0 14px 45px rgba(0,0,0,0.45);
}

/* --- Top bar --- */
.hud-topbar {
  background: linear-gradient(180deg, rgba(255,255,255,0.060), rgba(255,255,255,0.030));
  border: 1px solid rgba(255,255,255,0.105);
  border-radius: 16px;
  padding: 10px 12px;
  box-shadow: 0 14px 44px rgba(0,0,0,0.45);
}

#topbar { margin-bottom: 0.55rem; }

/* --- Button skins (global) --- */
div.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  padding: 0.55rem 0.9rem !important;
  font-weight: 780 !important;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92) !important;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
}

/* Run/Reset match screenshot (top bar + builder bottom) */
#topbar div.stButton:nth-of-type(1) > button,
#builder-actions div.stButton:nth-of-type(1) > button {
  background: linear-gradient(180deg, rgba(40,200,110,0.92), rgba(15,120,65,0.92)) !important;
  border: 1px solid rgba(70,255,170,0.22) !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.45), inset 0 0 0 1px rgba(255,255,255,0.10);
}

#topbar div.stButton:nth-of-type(2) > button,
#builder-actions div.stButton:nth-of-type(2) > button {
  background: linear-gradient(180deg, rgba(255,70,70,0.92), rgba(140,15,20,0.92)) !important;
  border: 1px solid rgba(255,120,120,0.22) !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.45), inset 0 0 0 1px rgba(255,255,255,0.10);
}

/* --- Inputs (BaseWeb) --- */
label { color: rgba(255,255,255,0.78) !important; }

div[data-baseweb="select"] > div {
  background: rgba(0,0,0,0.25) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
}

div[data-baseweb="input"] input {
  background: rgba(0,0,0,0.20) !important;
  border-radius: 12px !important;
}

/* Sliders */
div[data-baseweb="slider"] [role="slider"] { box-shadow: 0 0 0 3px rgba(90,170,255,0.15); }
div[data-baseweb="slider"] div[aria-label="rail"] { opacity: 0.85; }

/* Checkbox */
div[data-testid="stCheckbox"] label { gap: 0.55rem; }

/* --- Matplotlib --- */
canvas { outline: none; }

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Robust data load
# ----------------------------
@st.cache_data(show_spinner=True)
def _load_bundle(data_dir_hint: str):
    base = (data_dir_hint or os.environ.get("WARCOLOR_DATA_DIR", "") or "data").strip() or "data"
    candidates = [base, os.path.join(base, "data"), "data", os.path.join("data", "data"), "."]
    last_err = None
    for cand in candidates:
        try:
            return load_data(cand), cand
        except Exception as e:
            last_err = e
    raise last_err if last_err else FileNotFoundError("Could not locate data directory.")

bundle, data_dir = _load_bundle(os.environ.get("WARCOLOR_DATA_DIR", "data"))
df = bundle.countries.copy()

countries = df.sort_values("Country_Display")["Country_Display"].tolist()
iso_by_name = dict(zip(df["Country_Display"], df["ISO3"]))
name_by_iso = dict(zip(df["ISO3"], df["Country_Display"]))
# Ensure session values are valid for widgets (prevents Streamlit key/value warnings)
if countries:
    if st.session_state.get("battleground") not in countries:
        st.session_state["battleground"] = countries[0]
    # filter selections to existing options
    st.session_state["blue_names"] = [x for x in st.session_state.get("blue_names", []) if x in countries]
    st.session_state["red_names"] = [x for x in st.session_state.get("red_names", []) if x in countries]
    # Neutral/support lists should exclude selected belligerents.
    blue_set = set(st.session_state.get("blue_names", []) or [])
    red_set = set(st.session_state.get("red_names", []) or [])
    eligible_neutral = [x for x in countries if (x not in blue_set and x not in red_set)]

    st.session_state["neutral_names"] = [x for x in st.session_state.get("neutral_names", []) if x in eligible_neutral]
    st.session_state["support_blue"] = [x for x in st.session_state.get("support_blue", []) if x in eligible_neutral]
    st.session_state["support_red"] = [x for x in st.session_state.get("support_red", []) if x in eligible_neutral]

    # Prevent a neutral from "supporting" both sides at once.
    ov = set(st.session_state.get("support_blue", []) or []) & set(st.session_state.get("support_red", []) or [])
    if ov:
        st.session_state["support_red"] = [x for x in (st.session_state.get("support_red", []) or []) if x not in ov]

    # If a country is set as a supporter, treat it as neutral automatically (keeps model + UI coherent).
    sup_union = list(dict.fromkeys((st.session_state.get("neutral_names", []) or []) + (st.session_state.get("support_blue", []) or []) + (st.session_state.get("support_red", []) or [])))
    st.session_state["neutral_names"] = [x for x in sup_union if x in eligible_neutral]


def _iso_list(names: List[str]) -> List[str]:
    return [iso_by_name[n] for n in names if n in iso_by_name]


def _auto_theatre_mix(bg_row: pd.Series, scenario_type: str = "General") -> Tuple[float, float, float]:
    """Compute an auto Air/Sea/Land theatre mix from battleground geography + A2/AD, with light scenario nudges.
    Returns (air, sea, land) shares that sum to 1.
    """
    try:
        A = float(bg_row.get("area_km2", 0.0) or 0.0)
        P = float(bg_row.get("perimeter_km", 0.0) or 0.0)
        a2ad = float(bg_row.get("a2ad", 50.0) or 50.0)
        terrain = float(bg_row.get("terrain_score_0_100", 50.0) or 50.0)
        transport = float(bg_row.get("transport_score_0_100", 50.0) or 50.0)
    except Exception:
        A, P, a2ad, terrain, transport = 0.0, 0.0, 50.0, 50.0, 50.0

    # Coastline / fragmentation proxy: 1 - compactness
    compact = 0.35
    if A > 0 and P > 0:
        compact = (4.0 * math.pi * A) / (P * P)
    compact = float(np.clip(compact, 0.0, 1.0))
    littoral = 1.0 - compact  # 0 inland compact, 1 very littoral/fragmented

    # Base mix (simple and stable)
    sea = 0.25 + 0.55 * littoral
    air = 0.25 + 0.25 * (a2ad / 100.0) + 0.15 * (transport / 100.0)
    # terrain pushes toward land friction (more land share), but keep subtle
    air *= (1.0 - 0.08 * (terrain / 100.0))
    sea *= (1.0 - 0.05 * (terrain / 100.0))

    land = max(0.15, 1.0 - air - sea)

    # Scenario nudges
    stype = (scenario_type or "General").lower()
    if "amphib" in stype or "invasion" in stype or "island" in stype or "landing" in stype:
        sea += 0.10
        land += 0.03
        air -= 0.05
    elif "blockade" in stype or "port" in stype:
        sea += 0.15
        air -= 0.05
        land -= 0.08
    elif "air" in stype:
        air += 0.15
        sea -= 0.05
        land -= 0.10
    elif "land" in stype or "ground" in stype:
        land += 0.15
        sea -= 0.05
        air -= 0.10

    arr = np.array([air, sea, land], dtype=float)
    arr = np.clip(arr, 0.02, None)
    arr /= arr.sum()
    return float(arr[0]), float(arr[1]), float(arr[2])


def _fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"


def _render_losses_table(loss_table: pd.DataFrame) -> str:
    """Render the Battle Losses table as HUD HTML (closer to the reference image than st.dataframe)."""
    # Expected columns: Losses, Blue Today, Blue Total, Red Today, Red Total
    rows_html = []
    for _, r in loss_table.iterrows():
        loss = str(r.get("Losses", ""))
        bt = r.get("Blue Today", "-")
        bT = r.get("Blue Total", "-")
        rt = r.get("Red Today", "-")
        rT = r.get("Red Total", "-")

        def _cell(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "-"
            # Always display whole numbers in the Battle Losses HUD
            try:
                if isinstance(v, (int, np.integer)):
                    return f"{int(v):,}"
                if isinstance(v, (float, np.floating)):
                    return _fmt_int(v)
                # strings like '1575.0' or '1,575'
                vv = str(v).replace(",", "").strip()
                if vv in ("-", ""):
                    return "-"
                return _fmt_int(float(vv))
            except Exception:
                return str(v)

        rows_html.append(
            f"<tr>"
            f"<td class='wc-loss'>{loss}</td>"
            f"<td class='wc-blue'>{_cell(bt)}</td>"
            f"<td class='wc-blue'>{_cell(bT)}</td>"
            f"<td class='wc-red'>{_cell(rt)}</td>"
            f"<td class='wc-red'>{_cell(rT)}</td>"
            f"</tr>"
        )

    css = """
    <style>
      .loss-wrap{
        border-radius:16px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,0.12);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05);
        background: rgba(0,0,0,0.24);
      }
      table.loss{
        width:100%;
        border-collapse:collapse;
        font-size: 0.92rem;
      }
      table.loss th, table.loss td{ padding: 10px 12px; border-bottom:1px solid rgba(255,255,255,0.09); }
      table.loss thead th{ font-weight:850; letter-spacing:0.3px; }
      .h-loss{ text-align:left; background: rgba(255,255,255,0.04); }
      .h-blue{ text-align:center; background: linear-gradient(90deg, rgba(40,110,255,0.16), rgba(0,0,0,0.0)); color: rgba(150,200,255,0.95); }
      .h-red { text-align:center; background: linear-gradient(90deg, rgba(0,0,0,0.0), rgba(255,60,60,0.16)); color: rgba(255,170,170,0.95); }
      .h-sub{ font-weight:800; opacity:0.78; font-size:0.82rem; }

      td.wc-loss{ text-align:left; background: rgba(255,255,255,0.03); font-weight:650; }
      td.wc-blue{ text-align:center; background: rgba(40,110,255,0.08); color: rgba(145,205,255,0.98); font-weight:820; }
      td.wc-red { text-align:center; background: rgba(255,60,60,0.08);  color: rgba(255,165,165,0.98); font-weight:820; }

      table.loss tbody tr:last-child td{ border-bottom:none; }
    </style>
    """

    html = (
        "<div class='loss-wrap'>" +
        "<table class='loss'>"
        "<thead>"
        "<tr>"
        "<th class='h-loss' rowspan='2'>Battle Losses</th>"
        "<th class='h-blue' colspan='2'>Blue Forces</th>"
        "<th class='h-red' colspan='2'>Red Forces</th>"
        "</tr>"
        "<tr>"
        "<th class='h-blue h-sub'>Today's</th>"
        "<th class='h-blue h-sub'>Total</th>"
        "<th class='h-red h-sub'>Today's</th>"
        "<th class='h-red h-sub'>Total</th>"
        "</tr>"
        "</thead>"
        "<tbody>" + "".join(rows_html) + "</tbody>"
        "</table>"
        "</div>"
    )
    return css + html

# ----------------------------
# Chess Engine (rules-correct enough for duel sync)
# Supports: castling, en passant, 50-move, threefold repetition, stalemate/checkmate, insufficient material
# ----------------------------
_PIECE = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟︎",
    ".": "",
}

@dataclass
class ChessState:
    board: List[List[str]]
    white: bool
    castle: str  # subset of "KQkq"
    ep: Optional[Tuple[int,int]]  # (r,c) target square for en passant capture
    halfmove: int
    fullmove: int
    rep: Dict[str,int]
    result: str = ""
    last_uci: Optional[str] = None
    moves: List[str] = None

def _start_board():
    return [
        list("rnbqkbnr"),
        list("pppppppp"),
        list("........"),
        list("........"),
        list("........"),
        list("........"),
        list("PPPPPPPP"),
        list("RNBQKBNR"),
    ]

def _start_chess_state() -> ChessState:
    b = _start_board()
    st = ChessState(board=b, white=True, castle="KQkq", ep=None, halfmove=0, fullmove=1, rep={}, moves=[])
    k = _pos_key(st)
    st.rep[k] = 1
    return st

def _inb(r,c): return 0<=r<8 and 0<=c<8
def _isW(p): return p.isupper()
def _isB(p): return p.islower()

def _pos_key(st: ChessState) -> str:
    # repetition key: board + side + castling + ep-file
    rows = ["".join(r) for r in st.board]
    epf = "-" if st.ep is None else "abcdefgh"[st.ep[1]]
    return "/".join(rows) + (" w " if st.white else " b ") + "".join(sorted(st.castle)) + " " + epf

def _find_king(b, white: bool):
    k = "K" if white else "k"
    for r in range(8):
        for c in range(8):
            if b[r][c] == k:
                return (r,c)
    return None

def _attacked(b, attacker_white: bool, tr: int, tc: int) -> bool:
    enemy = _isB if attacker_white else _isW
    # pawns
    up = -1 if attacker_white else 1
    pr = tr - up
    for dc in (-1,1):
        pc = tc - dc
        if _inb(pr,pc):
            p = b[pr][pc]
            if p != "." and (attacker_white == _isW(p)) and p.upper()=="P":
                return True
    # knights
    for dr,dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        r,c = tr+dr, tc+dc
        if _inb(r,c):
            p=b[r][c]
            if p!="." and (attacker_white==_isW(p)) and p.upper()=="N":
                return True
    # sliders
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for dr,dc in dirs:
        r,c = tr+dr, tc+dc
        while _inb(r,c):
            p=b[r][c]
            if p!=".":
                if attacker_white==_isW(p):
                    u=p.upper()
                    if (dr==0 or dc==0) and u in ("R","Q"):
                        return True
                    if (dr!=0 and dc!=0) and u in ("B","Q"):
                        return True
                    if u=="K" and abs(r-tr)<=1 and abs(c-tc)<=1:
                        return True
                break
            r+=dr; c+=dc
    return False

def _in_check(b, white: bool) -> bool:
    kp = _find_king(b, white)
    if not kp:
        return True
    return _attacked(b, attacker_white=not white, tr=kp[0], tc=kp[1])

def _insufficient_material(b) -> bool:
    pieces=[]
    bishops=[]
    for r in range(8):
        for c in range(8):
            p=b[r][c]
            if p=="." or p.upper()=="K": 
                continue
            pieces.append(p)
            if p.upper()=="B":
                bishops.append((r+c)%2)
    # any pawn/rook/queen means sufficient
    for p in pieces:
        if p.upper() in ("P","R","Q"):
            return False
    # only minor pieces
    if len(pieces)==0:
        return True
    if len(pieces)==1 and pieces[0].upper() in ("B","N"):
        return True
    if len(pieces)==2 and all(p.upper()=="B" for p in pieces) and len(set(bishops))==1:
        return True
    return False

def _pseudo_moves(st: ChessState, r: int, c: int, attacks_only: bool=False):
    b=st.board; white=st.white
    p=b[r][c]
    if p=="." or (white != _isW(p)): 
        return []
    enemy = _isB if white else _isW
    moves=[]
    up = -1 if white else 1

    def add(rr,cc,promo=None,castle=None,ep=False):
        if not _inb(rr,cc): 
            return
        tgt=b[rr][cc]
        if tgt=="." or enemy(tgt):
            moves.append((r,c,rr,cc,promo,castle,ep))

    u=p.upper()
    if u=="P":
        # captures
        for dc in (-1,1):
            rr,cc=r+up,c+dc
            if _inb(rr,cc) and enemy(b[rr][cc]):
                if (white and rr==0) or ((not white) and rr==7):
                    for pr in ("Q","R","B","N"):
                        add(rr,cc,pr,None,False)
                else:
                    add(rr,cc,None,None,False)
        # en passant
        if st.ep is not None:
            er,ec=st.ep
            if er==r+up and abs(ec-c)==1:
                add(er,ec,None,None,True)
        if attacks_only:
            return moves
        # forward
        rr,cc=r+up,c
        if _inb(rr,cc) and b[rr][cc]==".":
            if (white and rr==0) or ((not white) and rr==7):
                for pr in ("Q","R","B","N"):
                    add(rr,cc,pr,None,False)
            else:
                add(rr,cc,None,None,False)
            # double
            rr2=r+2*up
            if (white and r==6) or ((not white) and r==1):
                if _inb(rr2,cc) and b[rr2][cc]==".":
                    add(rr2,cc,None,None,False)
        return moves

    if u=="N":
        for dr,dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            add(r+dr,c+dc)
        return moves

    if u in ("B","R","Q"):
        dirs=[]
        if u in ("B","Q"):
            dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
        if u in ("R","Q"):
            dirs += [(-1,0),(1,0),(0,-1),(0,1)]
        for dr,dc in dirs:
            rr,cc=r+dr,c+dc
            while _inb(rr,cc):
                tgt=b[rr][cc]
                if tgt==".":
                    moves.append((r,c,rr,cc,None,None,False))
                else:
                    if enemy(tgt):
                        moves.append((r,c,rr,cc,None,None,False))
                    break
                rr+=dr; cc+=dc
        return moves

    if u=="K":
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: 
                    continue
                add(r+dr,c+dc)
        if attacks_only:
            return moves
        # castling
        if white:
            if "K" in st.castle:
                if b[7][5]=="." and b[7][6]=="." and (not _in_check(b,True)) and (not _attacked(b,False,7,5)) and (not _attacked(b,False,7,6)):
                    moves.append((7,4,7,6,None,"K",False))
            if "Q" in st.castle:
                if b[7][1]=="." and b[7][2]=="." and b[7][3]=="." and (not _in_check(b,True)) and (not _attacked(b,False,7,3)) and (not _attacked(b,False,7,2)):
                    moves.append((7,4,7,2,None,"Q",False))
        else:
            if "k" in st.castle:
                if b[0][5]=="." and b[0][6]=="." and (not _in_check(b,False)) and (not _attacked(b,True,0,5)) and (not _attacked(b,True,0,6)):
                    moves.append((0,4,0,6,None,"k",False))
            if "q" in st.castle:
                if b[0][1]=="." and b[0][2]=="." and b[0][3]=="." and (not _in_check(b,False)) and (not _attacked(b,True,0,3)) and (not _attacked(b,True,0,2)):
                    moves.append((0,4,0,2,None,"q",False))
        return moves

    return moves

def _apply_move(st: ChessState, mv):
    b=[row[:] for row in st.board]
    r,c,r2,c2,promo,castle,ep = mv
    piece=b[r][c]
    tgt=b[r2][c2]
    white=st.white
    castle_rights=set(st.castle)

    # halfmove clock
    half = st.halfmove + 1
    if piece.upper()=="P" or tgt!="." or ep:
        half = 0

    # clear ep by default
    ep_sq=None

    # move piece
    b[r][c]="."

    # en passant capture
    if ep and piece.upper()=="P":
        cap_r = r
        cap_c = c2
        b[cap_r][cap_c]="."

    # castling rook move
    if castle:
        if castle in ("K","k"):
            # rook h -> f
            rr = 7 if castle=="K" else 0
            b[rr][5]=b[rr][7]; b[rr][7]="."
        if castle in ("Q","q"):
            rr = 7 if castle=="Q" else 0
            b[rr][3]=b[rr][0]; b[rr][0]="."

    # promotion
    if promo:
        b[r2][c2] = promo if white else promo.lower()
    else:
        b[r2][c2] = piece

    # update castling rights if king/rook moved or rook captured
    if piece=="K":
        castle_rights.discard("K"); castle_rights.discard("Q")
    if piece=="k":
        castle_rights.discard("k"); castle_rights.discard("q")
    if piece=="R":
        if r==7 and c==0: castle_rights.discard("Q")
        if r==7 and c==7: castle_rights.discard("K")
    if piece=="r":
        if r==0 and c==0: castle_rights.discard("q")
        if r==0 and c==7: castle_rights.discard("k")
    if tgt=="R":
        if r2==7 and c2==0: castle_rights.discard("Q")
        if r2==7 and c2==7: castle_rights.discard("K")
    if tgt=="r":
        if r2==0 and c2==0: castle_rights.discard("q")
        if r2==0 and c2==7: castle_rights.discard("k")

    # set en passant target for pawn double
    if piece.upper()=="P" and abs(r2-r)==2:
        ep_sq = ((r+r2)//2, c)

    # next side
    white2 = not white
    full = st.fullmove + (0 if white else 1)

    st2 = ChessState(board=b, white=white2, castle="".join(sorted(castle_rights)), ep=ep_sq, halfmove=half, fullmove=full, rep=dict(st.rep), moves=list(st.moves), result=st.result, last_uci=st.last_uci)
    k=_pos_key(st2)
    st2.rep[k]=st2.rep.get(k,0)+1
    return st2

def _legal_moves(st: ChessState):
    b=st.board; white=st.white
    out=[]
    for r in range(8):
        for c in range(8):
            if b[r][c]=="." or (white != _isW(b[r][c])):
                continue
            for mv in _pseudo_moves(st,r,c,attacks_only=False):
                st2=_apply_move(st,mv)
                if not _in_check(st2.board, white):
                    out.append(mv)
    return out


# piece-square tables (simple, from white POV)
_PST = {
    "P": np.array([
        [0,0,0,0,0,0,0,0],
        [5,5,5,5,5,5,5,5],
        [1,1,2,3,3,2,1,1],
        [0.5,0.5,1,2.5,2.5,1,0.5,0.5],
        [0,0,0,2,2,0,0,0],
        [0.5,-0.5,-1,0,0,-1,-0.5,0.5],
        [0.5,1,1,-2,-2,1,1,0.5],
        [0,0,0,0,0,0,0,0],
    ], dtype=float),
    "N": np.array([
        [-5,-4,-3,-3,-3,-3,-4,-5],
        [-4,-2,0,0,0,0,-2,-4],
        [-3,0,1,1.5,1.5,1,0,-3],
        [-3,0.5,1.5,2,2,1.5,0.5,-3],
        [-3,0,1.5,2,2,1.5,0,-3],
        [-3,0.5,1,1.5,1.5,1,0.5,-3],
        [-4,-2,0,0.5,0.5,0,-2,-4],
        [-5,-4,-3,-3,-3,-3,-4,-5],
    ], dtype=float),
    "B": np.array([
        [-2,-1,-1,-1,-1,-1,-1,-2],
        [-1,0,0,0,0,0,0,-1],
        [-1,0,0.5,1,1,0.5,0,-1],
        [-1,0.5,0.5,1,1,0.5,0.5,-1],
        [-1,0,1,1,1,1,0,-1],
        [-1,1,1,1,1,1,1,-1],
        [-1,0.5,0,0,0,0,0.5,-1],
        [-2,-1,-1,-1,-1,-1,-1,-2],
    ], dtype=float),
    "R": np.zeros((8,8),dtype=float),
    "Q": np.zeros((8,8),dtype=float),
    "K": np.zeros((8,8),dtype=float),
}

_PVAL = {"P": 1.0, "N": 3.2, "B": 3.25, "R": 5.0, "Q": 9.0, "K": 0.0}

def _uci(mv) -> str:
    r1,c1,r2,c2,promo,castle,ep = mv
    f = "abcdefgh"
    s = f"{f[c1]}{8-r1}{f[c2]}{8-r2}"
    if promo:
        s += str(promo).lower()
    return s

def _eval(st: ChessState, wW: float = 1.0, wB: float = 1.0) -> float:
    """Fast eval from White POV. Positive means White/Blue is better."""
    b = st.board
    score = 0.0
    for r in range(8):
        for c in range(8):
            p = b[r][c]
            if p == ".":
                continue
            u = p.upper()
            v = _PVAL.get(u, 0.0)
            pst = _PST.get(u)
            pstv = 0.0
            if pst is not None:
                if _isW(p):
                    pstv = float(pst[r, c])
                else:
                    pstv = float(pst[7-r, c])
            if _isW(p):
                score += v + 0.06 * pstv
            else:
                score -= v + 0.06 * pstv

    # Mobility (cheap): count legal moves for side to move and opponent
    try:
        cur_moves = len(_legal_moves(st))
        st2 = ChessState(
            board=[row[:] for row in st.board],
            white=not st.white,
            castle=st.castle,
            ep=st.ep,
            halfmove=st.halfmove,
            fullmove=st.fullmove,
            rep=dict(st.rep),
            moves=list(st.moves),
            result=st.result,
            last_uci=st.last_uci,
        )
        opp_moves = len(_legal_moves(st2))
        mob = (cur_moves - opp_moves) if st.white else (opp_moves - cur_moves)
        score += 0.02 * float(mob) * (1.0 if st.white else -1.0)
    except Exception:
        pass

    # Check pressure
    try:
        if _in_check(st.board, white=not st.white):
            score += 0.18 if st.white else -0.18
    except Exception:
        pass

    # Bias for "playing to win" based on weights
    score *= float(wW if st.white else wB)
    return float(score)

def _terminal_result(st: ChessState) -> str:
    # 50-move rule
    if st.halfmove >= 100:
        return "Draw (50-move rule)"
    # repetition
    k = _pos_key(st)
    if st.rep.get(k, 0) >= 3:
        return "Draw (threefold repetition)"
    # insufficient material
    if _insufficient_material(st.board):
        return "Draw (insufficient material)"
    # legal moves
    lm = _legal_moves(st)
    if len(lm) == 0:
        if _in_check(st.board, st.white):
            # side to move is mated
            return "Red wins (checkmate)" if st.white else "Blue wins (checkmate)"
        return "Draw (stalemate)"
    return ""

def _minimax(st: ChessState, depth: int, alpha: float, beta: float, wW: float, wB: float):
    """Alpha-beta minimax. Returns (best_score, best_move, terminal_string)."""
    term = _terminal_result(st)
    if term:
        # Big terminal scores
        if term.startswith("Blue wins"):
            return (9e6, None, term)
        if term.startswith("Red wins"):
            return (-9e6, None, term)
        return (0.0, None, term)

    if depth <= 0:
        return (_eval(st, wW, wB), None, "")

    best_mv = None
    if st.white:
        best = -1e18
        for mv in _legal_moves(st):
            st2 = _apply_move(st, mv)
            sc, _, t = _minimax(st2, depth-1, alpha, beta, wW, wB)
            if sc > best:
                best, best_mv = sc, mv
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return (float(best), best_mv, "")
    else:
        best = 1e18
        for mv in _legal_moves(st):
            st2 = _apply_move(st, mv)
            sc, _, t = _minimax(st2, depth-1, alpha, beta, wW, wB)
            if sc < best:
                best, best_mv = sc, mv
            beta = min(beta, best)
            if beta <= alpha:
                break
        return (float(best), best_mv, "")


def _render_board_html(board: List[List[str]], last_uci: Optional[str]) -> str:
    """Render chessboard with the original (pre-fixed_7) look: big pieces, glossy squares, rounded frame."""
    hl = set()
    try:
        if last_uci and len(last_uci) >= 4:
            files = "abcdefgh"
            # last_uci like e2e4 or e7e8q
            c1 = files.index(last_uci[0]); r1 = 8 - int(last_uci[1])
            c2 = files.index(last_uci[2]); r2 = 8 - int(last_uci[3])
            hl = {(r1, c1), (r2, c2)}
    except Exception:
        hl = set()

    css = """
    <style>
      .wc-wrap {
        display: inline-block;
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 16px 45px rgba(0,0,0,0.45);
        background: rgba(0,0,0,0.22);
      }
      .wc-board { border-collapse: collapse; margin: 0; }
      .wc-board td {
        width: 58px; height: 58px;
        text-align: center; vertical-align: middle;
        font-size: 38px; line-height: 1;
        user-select: none;
      }
      .wc-light { background: linear-gradient(180deg, #dde5f5, #cfd8ee); }
      .wc-dark  { background: linear-gradient(180deg, #7d8aa2, #6f7c95); }
      .wc-hl { box-shadow: inset 0 0 0 3px rgba(90,170,255,0.95); }
    </style>
    """

    rows = []
    for r in range(8):
        tds = []
        for c in range(8):
            cls = "wc-light" if (r + c) % 2 == 0 else "wc-dark"
            if (r, c) in hl:
                cls += " wc-hl"
            p = board[r][c]
            tds.append(f"<td class='{cls}'>{_PIECE.get(p, '')}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    html = "<div class='wc-wrap'><table class='wc-board'>" + "".join(rows) + "</table></div>"
    return css + html



def _weights_from_payload(payload: dict) -> Tuple[float, float, float, float, float]:
    """Return (wWhite, wBlack, pBlue, pRed, pDraw) from MC payload.
    IMPORTANT: Monte Carlo is the arbiter. Chess is the play-by-play.
    We bias the chess eval slightly toward the MC edge so chess and phase stay consistent.
    """
    wp = (payload or {}).get("win_prob", {}) or {}
    try:
        pB = float(wp.get("Blue", 0.5) or 0.5)
        pR = float(wp.get("Red", 0.5) or 0.5)
        pD = float(wp.get("Draw", max(0.0, 1.0 - pB - pR)) or 0.0)
    except Exception:
        pB, pR, pD = 0.5, 0.5, 0.0
    tot = pB + pR + pD
    if tot > 1e-9:
        pB, pR, pD = pB / tot, pR / tot, pD / tot

    # Stronger bias than before so chess storyline matches MC signal
    edge = float(np.clip(pB - pR, -0.75, 0.75))
    wW = 1.0 + 0.35 * edge
    wB = 1.0 - 0.35 * edge
    return float(wW), float(wB), float(pB), float(pR), float(pD)


def _pct_int_triplet(pB: float, pR: float, pD: float) -> Tuple[int, int, int]:
    """Round to whole-number percents that sum to 100."""
    pB = float(np.clip(pB, 0.0, 1.0))
    pR = float(np.clip(pR, 0.0, 1.0))
    pD = float(np.clip(pD, 0.0, 1.0))
    tot = pB + pR + pD
    if tot > 1e-9:
        pB, pR, pD = pB / tot, pR / tot, pD / tot
    vals = np.array([pB, pR, pD], dtype=float) * 100.0
    ints = np.floor(vals).astype(int)
    rem = int(100 - ints.sum())
    if rem > 0:
        frac = vals - np.floor(vals)
        for i in np.argsort(-frac)[:rem]:
            ints[i] += 1
    return int(ints[0]), int(ints[1]), int(ints[2])


def _mc_winner(pB: float, pR: float, pD: float) -> str:
    """Return the most-likely outcome label from probabilities."""
    pB = float(pB); pR = float(pR); pD = float(pD)
    if pD >= pB and pD >= pR:
        return "Draw"
    return "Blue" if pB >= pR else "Red"


# ----------------------------
# Phase Separation (Ising + Kawasaki swaps, buoyancy tied to advantage)
# ----------------------------
@dataclass
class PhaseState:
    spins: np.ndarray  # int8 in {-1,+1}
    rng: np.random.Generator
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    u: np.ndarray      # vertical coordinate in [-1,1], shape (H,1)

def _make_phase_state(seed0: int, H: int = 140, W: int = 248) -> PhaseState:
    rng = np.random.default_rng(int(seed0))
    n = H * W
    arr = np.ones(n, dtype=np.int8)
    arr[: n//2] = -1
    rng.shuffle(arr)
    spins = arr.reshape(H, W)

    pairs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    # Horizontal neighbor pairs
    for parity in (0, 1):
        cols = np.arange(parity, W, 2, dtype=np.int32)
        rA = np.repeat(np.arange(H, dtype=np.int32), cols.size)
        cA = np.tile(cols, H)
        pairs[f"h{parity}"] = (rA, cA, rA, (cA + 1) % W)
    # Vertical neighbor pairs
    for parity in (0, 1):
        rows = np.arange(parity, H, 2, dtype=np.int32)
        rA = np.repeat(rows, W)
        cA = np.tile(np.arange(W, dtype=np.int32), rows.size)
        pairs[f"v{parity}"] = (rA, cA, (rA + 1) % H, cA)

    u = np.linspace(1.0, -1.0, H, dtype=float).reshape(H, 1)  # +1 at top row (0), -1 at bottom
    return PhaseState(spins=spins, rng=rng, pairs=pairs, u=u)

def _neighbor_sum(spins: np.ndarray) -> np.ndarray:
    s = spins.astype(np.int16)
    return (np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1))

def _kawasaki_sweep(state: PhaseState, T: float, adv: float, n_swaps: int, J: float = 1.0, buoy: float = 1.15):
    """Kawasaki dynamics via neighbor-pair flips (swap opposite spins = flip both)."""
    spins = state.spins
    H, W = spins.shape
    # pick a parity + direction (checkerboard pairing)
    parity = int(state.rng.integers(0, 2))
    direction = "h" if int(state.rng.integers(0, 2)) == 0 else "v"
    rA, cA, rB, cB = state.pairs[f"{direction}{parity}"]
    n_pairs = rA.size
    n_take = int(min(max(1, n_swaps), n_pairs))
    idx = state.rng.choice(n_pairs, size=n_take, replace=False)

    r1 = rA[idx]; c1 = cA[idx]; r2 = rB[idx]; c2 = cB[idx]
    s1 = spins[r1, c1].astype(np.int16)
    s2 = spins[r2, c2].astype(np.int16)
    diff = (s1 != s2)
    if not np.any(diff):
        return

    sumN = _neighbor_sum(spins)
    n1 = sumN[r1, c1].astype(np.int16)
    n2 = sumN[r2, c2].astype(np.int16)

    # buoyancy field (winner rises): energy term +h*s where h = buoy*adv*u
    h1 = (buoy * float(adv)) * state.u[r1, 0]
    h2 = (buoy * float(adv)) * state.u[r2, 0]

    # ΔE for flipping both spins (correcting shared bond)
    dE = (2.0 * J) * (s1 * n1 + s2 * n2) + 2.0 * (h1 * s1 + h2 * s2) - (4.0 * J) * (s1 * s2)
    dE = dE.astype(float)

    # Only for opposite spins
    dE = np.where(diff, dE, 1e9)

    # Metropolis
    accept = (dE <= 0.0)
    if np.any(~accept):
        prob = np.exp(-dE / max(1e-6, float(T)))
        rnd = state.rng.random(size=prob.shape)
        accept = accept | (rnd < prob)

    acc = accept & diff
    if not np.any(acc):
        return

    rr1 = r1[acc]; cc1 = c1[acc]; rr2 = r2[acc]; cc2 = c2[acc]
    spins[rr1, cc1] *= -1
    spins[rr2, cc2] *= -1

def _buoyancy_swaps(state: PhaseState, adv: float, n: int, strength: float = 1.0):
    """Explicit buoyancy: winner color drifts upward by biased vertical swaps.
    This guarantees clear layering (winner on top) when the conflict is decisive.
    adv > 0 => Blue (+1) rises, adv < 0 => Red (-1) rises.
    """
    adv = float(adv)
    if n <= 0 or abs(adv) < 1e-6:
        return
    spins = state.spins
    H, W = spins.shape
    winner = 1 if adv > 0 else -1
    loser = -winner
    rng = state.rng
    rr = rng.integers(1, H, size=int(n))  # lower cell row index
    cc = rng.integers(0, W, size=int(n))
    up = spins[rr - 1, cc]
    lo = spins[rr, cc]
    mask = (lo == winner) & (up == loser)
    if not np.any(mask):
        return
    p = 0.55 + 0.35 * min(1.0, float(strength) * abs(adv))
    take = mask & (rng.random(size=mask.shape) < p)
    if not np.any(take):
        return
    r2 = rr[take]
    c2 = cc[take]
    spins[r2 - 1, c2] = winner
    spins[r2, c2] = loser


def _blur2d(x: np.ndarray, iters: int = 2) -> np.ndarray:
    a = x.astype(float)
    for _ in range(max(0, int(iters))):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def _render_phase(state: PhaseState, frac: float, payload: dict, seed: int, advance: bool = True):
    """Render the Monte Carlo phase panel.
    Visual contract:
      - early: mixed looks BLACK
      - later: domains form and saturate into RED/BLUE
      - winner (by MC edge) rises to the TOP; draw stays warmer/mixed
    """
    frac = float(np.clip(frac, 0.0, 1.0))
    wp = (payload or {}).get("win_prob", {}) or {}
    pB = float(wp.get("Blue", 0.5) or 0.5)
    pR = float(wp.get("Red", 0.5) or 0.5)
    pD = float((payload or {}).get("drawness", wp.get("Draw", max(0.0, 1.0 - pB - pR))) or 0.0)
    tot = pB + pR + pD
    if tot > 1e-9:
        pB, pR, pD = pB / tot, pR / tot, pD / tot

    # Monte Carlo edge drives buoyancy (arbiter).
    adv = (payload or {}).get("phase_adv", None)
    if adv is None:
        adv = pB - pR
    adv = float(np.clip(float(adv), -1.0, 1.0))

    # Draw keeps the mixture warm (less separation).
    drawness = float(np.clip((payload or {}).get("drawness", pD), 0.0, 1.0))

    # Buoyancy strength ∝ (Pwinner - Ploser) * (1 - Pdraw)
    # This gives clean layers when there's a clear winner, and preserves mixing when draws are likely.
    edge = abs(pB - pR) * (1.0 - drawness)
    sign = 1.0 if (pB - pR) >= 0 else -1.0
    adv_eff = float(sign * np.clip(edge * 1.80, 0.0, 1.0))

    # Cooling schedule (hot -> cool), but with a draw floor to preserve mixing
    T_hot = 3.8 * (1.0 - frac) ** 1.55 + 0.10
    T_floor = 0.55 + 0.95 * drawness
    T = float(max(T_hot, T_floor))

    if advance:
        H, W = state.spins.shape
        speed = float(st.session_state.get("sim_speed", 0.60))
        # more swaps later; speed increases update cost but keeps motion visible
        swaps = int((H * W) * (0.65 + 2.90 * frac) * (0.55 + 1.15 * speed))
        reps = 1 if frac < 0.10 else (2 if frac < 0.32 else 3)
        buoy = 1.6 + 2.8 * frac  # stronger buoyancy late so winner clearly rises
        for _ in range(reps):
            _kawasaki_sweep(state, T=T, adv=adv_eff, n_swaps=swaps, buoy=buoy)
        # Explicit buoyancy drift (winner rises) to guarantee a clear layered outcome
        buoy_swaps = int((H * W) * (0.10 + 0.85 * frac) * (0.35 + 1.10 * speed) * abs(adv_eff))
        _buoyancy_swaps(state, adv=adv_eff, n=buoy_swaps, strength=(1.0 + 2.0 * frac))

    s = state.spins.astype(float)
    # Lighter blur keeps boundaries visible
    m = _blur2d(s, iters=2)
    m = np.clip(m, -1.0, 1.0)

    # Mixed looks black; separation threshold decreases over time
    m0 = 0.58 * (1.0 - frac) ** 0.60 + 0.06
    sat = np.clip((np.abs(m) - m0) / max(1e-6, (1.0 - m0)), 0.0, 1.0)
    sat = sat ** 0.60

    blue = np.array([0.08, 0.36, 1.00], dtype=float)
    red  = np.array([1.00, 0.14, 0.14], dtype=float)

    rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=float)
    pos = (m >= 0)
    rgb[pos]  = sat[pos, None] * blue
    rgb[~pos] = sat[~pos, None] * red

    gain = 0.95 + 1.20 * frac
    rgb = np.clip(rgb * gain, 0.0, 1.0)
    # Downsample 2x to remove lattice dot artifacts and make domains look fluid
    H, W = rgb.shape[0], rgb.shape[1]
    if (H % 2 == 0) and (W % 2 == 0):
        rgb = rgb.reshape(H//2, 2, W//2, 2, 3).mean(axis=(1, 3))

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=140)
    # origin='upper' so row 0 is TOP (matches "winner rises to the top")
    ax.imshow(rgb, origin="upper", interpolation="bilinear")
    ax.set_axis_off()
    fig.patch.set_facecolor((0.02, 0.02, 0.03))
    ax.set_facecolor((0.02, 0.02, 0.03))
    fig.tight_layout(pad=0)
    return fig



# ----------------------------
# Session defaults
# ----------------------------
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("year", 2030)
_init("mode", "Limited War")
_init("intensity", "Campaign")
_init("max_days", 360)
_init("nukes_allowed", False)
_init("core_model", "Cinematic")
_init("scenario_type", "General")
_init("auto_integrated", True)

_init("override_mix", False)
_init("mix_blend", 0.85)
_init("mix_air", 0.33)
_init("mix_sea", 0.33)
_init("mix_land", 0.34)

# Dense knobs
_init("dense_embed_strength", 1.0)
_init("dense_fronts", 1)
_init("dense_air_mult", 1.00)
_init("dense_sea_mult", 1.00)
_init("dense_land_mult", 1.00)
_init("dense_missile_mult", 1.00)
_init("dense_contact_mult", 1.00)
_init("dense_civ_mult", 1.00)
_init("dense_commit_mult", 1.00)
_init("dense_fit_requested", False)
_init("dense_fit_iters", 150)
_init("dense_fit_days", 90)

# Integrated knobs
_init("ic_blue_detection_prob", 0.62)
_init("ic_missile_fraction_to_ships", 0.55)
_init("ic_scuttle_day", 2)
_init("ic_unsupplied_loss", 0.10)
_init("ic_k_blue_air", 0.0020)
_init("ic_k_red_air", 0.0020)
_init("ic_k_blue_land", 0.0010)
_init("ic_k_red_land", 0.0010)
_init("ic_enable_orders", True)
_init("ic_aggression_blue", 0.55)
_init("ic_aggression_red", 0.60)
_init("ic_use_roll_hits", False)
_init("ic_fit_requested", False)
_init("ic_fit_iters", 1200)


_init("battleground", "Ukraine" if "Ukraine" in countries else (countries[0] if countries else ""))
_init("blue_names", ["United States", "United Kingdom", "France"] if "United States" in countries else [])
_init("red_names", ["China", "Russia"] if "China" in countries and "Russia" in countries else [])
_init("neutral_names", [])
_init("support_blue", [])
_init("support_red", [])

_init("sanctions_blue", 0.0)
_init("sanctions_red", 0.0)

_init("iters", 200_000)
_init("seed", 7)

_init("sim_speed", 0.60)  # UI speed knob (0 slow -> 1 fast)
_init("duel_depth", 2)
_init("duel_plies", 96)

# ----------------------------
# Top bar (Run/Reset/Speed)
# ----------------------------
def _reset_all():
    keep = {"sim_speed"}  # keep speed if you want
    for k in list(st.session_state.keys()):
        if k.startswith("_") or k in keep:
            continue
        # preserve cache keys etc
        if k in ("_run", "_last_run_seed"):
            continue
        # wipe user settings keys
        if k in (
            "year","mode","intensity","max_days","nukes_allowed","core_model","scenario_type",
            "battleground","blue_names","red_names","neutral_names","support_blue","support_red",
            "sanctions_blue","sanctions_red","iters","seed","duel_depth","duel_plies",
            "last_summary","last_series","last_mc",
            # duel visuals
            "chess_state","phase_state"
        ):
            st.session_state.pop(k, None)
    st.rerun()

top = st.container()
with top:
    st.markdown("<div id='topbar' class='hud-topbar'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([0.12, 0.12, 0.52, 0.24], vertical_alignment="center")
    run_top = c1.button("Run", type="primary", use_container_width=True)
    reset_top = c2.button("Reset", use_container_width=True)
    sim_speed = c3.slider("Simulation Speed", 0.10, 1.00, step=0.01, key="sim_speed", label_visibility="visible")
    c4.markdown("**GeoConflictModeler**<br><span class='small'>Educational toy model</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if reset_top:
    _reset_all()

# ----------------------------
# Main layout (left builder, right arena)
# ----------------------------
left, arena = st.columns([0.34, 0.66], gap="large")

with left:
    st.markdown("<div class='hud-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='hud-title'>Scenario Builder</div>", unsafe_allow_html=True)
    st.markdown("<div class='hud-sub'>Choose parameters, then press Run.</div>", unsafe_allow_html=True)

    st.slider("Year", 2025, 2036, step=1, key="year")
    st.selectbox("War Mode", ["Limited War", "Total War"], key="mode")
    st.selectbox("Intensity", ["Skirmish", "Campaign", "All-out"], key="intensity")
    st.slider("Sim Horizon (days)", 30, 3650, step=10, key="max_days")
    st.checkbox("Allow nuclear escalation (macro-shock)", key="nukes_allowed")

    st.selectbox("Core Model", ["Cinematic", "Dense (latent)", "Integrated (4-engine)"], key="core_model")

    st.selectbox("Scenario Type", ["General", "Amphibious invasion / island landing", "Blockade / port strikes", "Air campaign", "Land campaign"], key="scenario_type")
    st.checkbox("Auto-use Integrated for invasion/blockade", key="auto_integrated")

    st.divider()
    st.selectbox("Battleground (theater anchor)", countries, key="battleground")

    st.markdown("<div class='panel-h'>Alliances</div>", unsafe_allow_html=True)
    st.multiselect("Blue Forces", countries, key="blue_names")
    st.multiselect("Red Forces", countries, key="red_names")

    with st.expander("Advanced Options", expanded=False):
        # Keep UI identical, but avoid Streamlit "default + session_state" warnings by not
        # passing explicit defaults when using a widget key.
        st.multiselect("Neutral countries (optional)", countries, key="neutral_names")
        # Supporters can be selected from any eligible neutral state (not already Blue/Red).
        # They will be added to the neutral list automatically via the sanitization logic above.
        eligible_neutral = [x for x in countries if (x not in (st.session_state.get("blue_names", []) or []) and x not in (st.session_state.get("red_names", []) or []))]
        st.multiselect("Neutrals supporting Blue", eligible_neutral, key="support_blue")
        st.multiselect("Neutrals supporting Red", eligible_neutral, key="support_red")


        st.divider()
        st.markdown("**Theatre mix (auto + override)**")
        # Battleground ISO3 should be derived from the battleground selection only.
        # (Avoid referencing alliance ISO lists here to prevent NameError before Run.)
        bg_iso_tmp = iso_by_name.get(
            st.session_state.get("battleground", countries[0] if countries else ""),
            (df["ISO3"].iloc[0] if len(df) > 0 else ""),
        )
        bg_row = df[df["ISO3"] == bg_iso_tmp].iloc[0] if len(df[df["ISO3"] == bg_iso_tmp])>0 else df.iloc[0]
        a_auto, s_auto, l_auto = _auto_theatre_mix(bg_row, st.session_state.get("scenario_type","General"))
        st.caption(f"Auto mix: Air {a_auto:.2f} • Sea {s_auto:.2f} • Land {l_auto:.2f}")

        st.checkbox("Override auto mix", key="override_mix")
        st.slider("Override strength", 0.0, 1.0, step=0.05, key="mix_blend", disabled=not bool(st.session_state.get("override_mix", False)))
        st.slider("Air share", 0.0, 1.0, step=0.01, key="mix_air", disabled=not bool(st.session_state.get("override_mix", False)))
        st.slider("Sea share", 0.0, 1.0, step=0.01, key="mix_sea", disabled=not bool(st.session_state.get("override_mix", False)))
        st.slider("Land share", 0.0, 1.0, step=0.01, key="mix_land", disabled=not bool(st.session_state.get("override_mix", False)))

        st.divider()
        st.markdown("**Dense knobs (latent)**")
        st.caption("Only used when Core Model is Dense (latent).")
        st.slider("Latent influence", 0.0, 2.5, step=0.05, key="dense_embed_strength")
        st.slider("Fronts (multi-theatre)", 1, 6, step=1, key="dense_fronts")
        st.slider("Air multiplier", 0.60, 1.70, step=0.02, key="dense_air_mult")
        st.slider("Sea multiplier", 0.60, 1.70, step=0.02, key="dense_sea_mult")
        st.slider("Land multiplier", 0.60, 1.70, step=0.02, key="dense_land_mult")
        st.slider("Missile multiplier", 0.60, 1.70, step=0.02, key="dense_missile_mult")
        st.slider("Contact/tempo multiplier", 0.60, 1.70, step=0.02, key="dense_contact_mult")
        st.slider("Civilian-risk multiplier", 0.60, 1.70, step=0.02, key="dense_civ_mult")
        st.slider("Force-commit multiplier", 0.60, 1.70, step=0.02, key="dense_commit_mult")

        st.markdown("**Auto-fit Dense cfg (optional)**")
        st.slider("Calibration horizon (days)", 30, 365, step=5, key="dense_fit_days")
        st.slider("Fit iterations (higher = slower)", 50, 400, step=10, key="dense_fit_iters")
        t_b_air = st.number_input("Target Blue aircraft lost", min_value=0, value=0, step=10)
        t_b_ship = st.number_input("Target Blue ships lost", min_value=0, value=0, step=5)
        t_b_kia = st.number_input("Target Blue KIA", min_value=0, value=0, step=100)
        t_r_air = st.number_input("Target Red aircraft lost", min_value=0, value=0, step=10)
        t_r_ship = st.number_input("Target Red ships lost", min_value=0, value=0, step=5)
        t_r_kia = st.number_input("Target Red KIA", min_value=0, value=0, step=100)
        req = st.button("Request Dense fit (runs on Run scenario)")
        if req:
            st.session_state["dense_fit_requested"] = True
            st.session_state["dense_fit_targets"] = {
                "days": int(st.session_state["dense_fit_days"]),
                "blue": {"aircraft_lost": float(t_b_air), "ships_lost": float(t_b_ship), "kia": float(t_b_kia)},
                "red": {"aircraft_lost": float(t_r_air), "ships_lost": float(t_r_ship), "kia": float(t_r_kia)},
            }
            st.success("Dense fit request stored. It will run when you click Run.")

        st.divider()
        st.markdown("**Integrated knobs (4-engine)**")
        st.caption("Only used when Core Model is Integrated (4-engine), or auto-switched for invasion/blockade.")
        st.slider("Blue detection probability", 0.25, 0.90, step=0.01, key="ic_blue_detection_prob")
        st.slider("Missile fraction aimed at ships", 0.25, 0.90, step=0.01, key="ic_missile_fraction_to_ships")
        st.slider("Port scuttle day", 1, 14, step=1, key="ic_scuttle_day")
        st.slider("Daily effectiveness loss if unsupplied", 0.02, 0.20, step=0.01, key="ic_unsupplied_loss")
        st.slider("Air attrition k (Blue)", 0.0006, 0.0060, step=0.0001, key="ic_k_blue_air")
        st.slider("Air attrition k (Red)", 0.0006, 0.0060, step=0.0001, key="ic_k_red_air")
        st.slider("Land attrition k (Blue)", 0.0003, 0.0045, step=0.0001, key="ic_k_blue_land")
        st.slider("Land attrition k (Red)", 0.0003, 0.0045, step=0.0001, key="ic_k_red_land")
        st.checkbox("Enable Orders AI (Blue + Red)", key="ic_enable_orders")
        st.slider("Blue aggression", 0.0, 1.0, step=0.05, key="ic_aggression_blue")
        st.slider("Red aggression", 0.0, 1.0, step=0.05, key="ic_aggression_red")
        st.checkbox("Use roll-based missile hit adjudication (fog-of-war)", key="ic_use_roll_hits")

        st.markdown("**Auto-fit Integrated cfg (optional)**")
        st.slider("Fit iterations", 50, 4000, step=50, key="ic_fit_iters")
        t_b_air2 = st.number_input("Target Blue aircraft lost (Integrated)", min_value=0, value=0, step=10)
        t_b_ship2 = st.number_input("Target Blue ships lost (Integrated)", min_value=0, value=0, step=5)
        t_r_air2 = st.number_input("Target Red aircraft lost (Integrated)", min_value=0, value=0, step=10)
        t_r_ship2 = st.number_input("Target Red ships lost (Integrated)", min_value=0, value=0, step=5)
        req2 = st.button("Request Integrated fit (runs on Run scenario)")
        if req2:
            st.session_state["ic_fit_requested"] = True
            st.session_state["ic_fit_targets"] = {
                "blue": {"aircraft_lost": float(t_b_air2), "ships_lost": float(t_b_ship2)},
                "red": {"aircraft_lost": float(t_r_air2), "ships_lost": float(t_r_ship2)},
            }
            st.success("Integrated fit request stored. It will run when you click Run.")
    st.markdown("<div class='panel-h'>Economics</div>", unsafe_allow_html=True)
    st.slider("Sanctions vs. Blue", 0.0, 1.0, step=0.05, key="sanctions_blue")
    st.slider("Sanctions vs. Red", 0.0, 1.0, step=0.05, key="sanctions_red")

    st.markdown("<div class='panel-h'>Monte Carlo settings</div>", unsafe_allow_html=True)
    st.number_input("Iterations", min_value=1_000, max_value=10_000_000, step=50_000, key="iters")
    st.number_input("RNG Seed", min_value=1, max_value=999_999, step=1, key="seed")

    st.markdown("<div class='panel-h'>Duel feel</div>", unsafe_allow_html=True)
    st.slider("Duel plies (chess moves)", 32, 240, step=8, key="duel_plies")
    st.slider("Duel depth", 1, 3, step=1, key="duel_depth")

    st.markdown("<div id='builder-actions'>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    run_bottom = b1.button("Run", type="primary", use_container_width=True, key="_run_bottom")
    reset_bottom = b2.button("Reset", use_container_width=True, key="_reset_bottom")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if reset_bottom:
    _reset_all()

# Validate (outside left panel)
blue_iso = _iso_list(st.session_state["blue_names"])
red_iso = _iso_list(st.session_state["red_names"])
neutral_iso = _iso_list(st.session_state.get("neutral_names", []))
support_blue_iso = _iso_list(st.session_state.get("support_blue", []))
support_red_iso = _iso_list(st.session_state.get("support_red", []))

can_run = True
if len(set(blue_iso) & set(red_iso)) > 0:
    st.error("A country cannot be on both Blue and Red. Remove overlaps.")
    can_run = False
if not blue_iso or not red_iso:
    st.warning("Pick at least 1 Blue country and 1 Red country.")
    can_run = False

# ----------------------------
# Arena placeholders (always visible)
# ----------------------------
with arena:
    st.markdown("<div class='hud-panel'>", unsafe_allow_html=True)

    # two visuals
    cA, cB = st.columns([0.50, 0.50], gap="medium")
    with cA:
        st.markdown("<div class='hud-frame'><div class='panel-h'>♟️ War Chess</div>", unsafe_allow_html=True)
        chess_slot = st.empty()
        chess_meta = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with cB:
        st.markdown("<div class='hud-frame'><div class='panel-h'>🌡️ Monte Carlo Phase Separation</div>", unsafe_allow_html=True)
        fluid_slot = st.empty()
        fluid_meta = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hud-frame'><div class='panel-h'>Battle Losses</div>", unsafe_allow_html=True)
    loss_table_slot = st.empty()
    loss_meta_slot = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='small' style='text-align:center; opacity:0.65; padding-top:6px;'>Educational, cinematic toy model. Not a real-world predictor.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Initial render if nothing run yet
if "chess_state" not in st.session_state:
    st.session_state["chess_state"] = _start_chess_state()
if "phase_state" not in st.session_state:
    st.session_state["phase_state"] = _make_phase_state(int(st.session_state["seed"]) + 1337)

chess_slot.markdown(_render_board_html(st.session_state["chess_state"].board, None), unsafe_allow_html=True)

# If we have previous results, keep them visible on reruns (so the HUD doesn’t revert to placeholders).
_last_mc = st.session_state.get("last_mc", None) or {}
_wp = (st.session_state.get("last_fused_win_prob", None) or _last_mc.get("win_prob", None) or {"Blue": 0.5, "Red": 0.5, "Draw": 0.0})
_payload0 = {"win_prob": dict(_wp)}

# Show a "final" looking phase snapshot if MC exists; otherwise show the hot/mixed baseline.
_frac0 = 1.0 if ("last_mc" in st.session_state) else 0.0
fig0 = _render_phase(st.session_state["phase_state"], _frac0, _payload0, int(st.session_state["seed"]), advance=False)
fluid_slot.pyplot(fig0, use_container_width=True)
plt.close(fig0)

_bi0, _ri0, _di0 = _pct_int_triplet(float(_wp.get("Blue", 0.5) or 0.5), float(_wp.get("Red", 0.5) or 0.5), float(_wp.get("Draw", 0.0) or 0.0))
fluid_meta.markdown(f"**Win mix:** Blue {_bi0}% • Red {_ri0}% • Draw {_di0}%")

def _build_loss_table_from_state(summary: dict, series: Optional[List[dict]]) -> pd.DataFrame:
    tb = (summary or {}).get("totals", {}).get("blue", {}) or {}
    tr = (summary or {}).get("totals", {}).get("red", {}) or {}

    b_today_kia = r_today_kia = 0
    b_today_air = r_today_air = 0
    b_today_ship = r_today_ship = 0
    b_today_tank = r_today_tank = 0
    b_today_art = r_today_art = 0
    b_today_miss = r_today_miss = 0

    try:
        s = pd.DataFrame(series or []).sort_values("day")
        if len(s) >= 2:
            a = s.iloc[-2].to_dict()
            b = s.iloc[-1].to_dict()
            b_today_kia = int(max(0, (b.get("blue_total_kia", 0) or 0) - (a.get("blue_total_kia", 0) or 0)))
            r_today_kia = int(max(0, (b.get("red_total_kia", 0) or 0) - (a.get("red_total_kia", 0) or 0)))

            b_today_air = int(max(0, round((a.get("blue_aircraft_remaining", 0) or 0) - (b.get("blue_aircraft_remaining", 0) or 0))))
            r_today_air = int(max(0, round((a.get("red_aircraft_remaining", 0) or 0) - (b.get("red_aircraft_remaining", 0) or 0))))
            b_today_ship = int(max(0, round((a.get("blue_ships_remaining", 0) or 0) - (b.get("blue_ships_remaining", 0) or 0))))
            r_today_ship = int(max(0, round((a.get("red_ships_remaining", 0) or 0) - (b.get("red_ships_remaining", 0) or 0))))
            b_today_tank = int(max(0, round((a.get("blue_tanks_remaining", 0) or 0) - (b.get("blue_tanks_remaining", 0) or 0))))
            r_today_tank = int(max(0, round((a.get("red_tanks_remaining", 0) or 0) - (b.get("red_tanks_remaining", 0) or 0))))
            b_today_miss = int(max(0, round((a.get("blue_missiles_remaining", 0) or 0) - (b.get("blue_missiles_remaining", 0) or 0))))
            r_today_miss = int(max(0, round((a.get("red_missiles_remaining", 0) or 0) - (b.get("red_missiles_remaining", 0) or 0))))
            b_today_miss = int(max(0, round((a.get("blue_missiles_remaining", 0) or 0) - (b.get("blue_missiles_remaining", 0) or 0))))
            r_today_miss = int(max(0, round((a.get("red_missiles_remaining", 0) or 0) - (b.get("red_missiles_remaining", 0) or 0))))
    except Exception:
        pass

    def _safe_int(v):
        try:
            if v is None:
                return 0
            return int(round(float(v)))
        except Exception:
            return 0

    return pd.DataFrame({
        "Losses": ["Personnel (KIA)", "Aircraft", "Tanks / AFVs", "Artillery", "Missiles Used"],
        "Blue Today": [b_today_kia, b_today_air, b_today_tank, b_today_art, b_today_miss],
        "Blue Total": [_safe_int(tb.get("kia", 0)), _safe_int(tb.get("aircraft", 0)), _safe_int(tb.get("tanks", 0)), _safe_int(tb.get("artillery", 0)), _safe_int(tb.get("missiles", 0))],
        "Red Today":  [r_today_kia, r_today_air, r_today_tank, r_today_art, r_today_miss],
        "Red Total":  [_safe_int(tr.get("kia", 0)), _safe_int(tr.get("aircraft", 0)), _safe_int(tr.get("tanks", 0)), _safe_int(tr.get("artillery", 0)), _safe_int(tr.get("missiles", 0))],
    })

if "last_summary" in st.session_state:
    lt = _build_loss_table_from_state(st.session_state.get("last_summary", {}) or {}, st.session_state.get("last_series", []) or [])
    loss_table_slot.markdown(_render_losses_table(lt), unsafe_allow_html=True)
    try:
        _ls = st.session_state.get("last_summary", {}) or {}
        _mc = st.session_state.get("last_mc", {}) or {}
        _wp2 = (_mc.get("win_prob", {}) or {})
        pB2 = float(_wp2.get("Blue", 0.5) or 0.5)
        pR2 = float(_wp2.get("Red", 0.5) or 0.5)
        pD2 = float(_wp2.get("Draw", 0.0) or max(0.0, 1.0 - pB2 - pR2))
        _bi2, _ri2, _di2 = _pct_int_triplet(pB2, pR2, pD2)
        _w2 = _mc_winner(pB2, pR2, pD2)
        fw = st.session_state.get("last_fused_win_prob", {}) or {}
        pBf2 = float(fw.get("Blue", pB2) or pB2)
        pRf2 = float(fw.get("Red", pR2) or pR2)
        pDf2 = float(fw.get("Draw", pD2) or pD2)
        _bf2, _rf2, _df2 = _pct_int_triplet(pBf2, pRf2, pDf2)
        loss_meta_slot.markdown(
            f"**Outcome:** {_w2}  ·  <span class='small'>MC: Blue {_bi2}% · Red {_ri2}% · Draw {_di2}%</span>"
            f"<br><span class='small'>Fused (MC ⊕ Chess): Blue {_bf2}% · Red {_rf2}% · Draw {_df2}%</span>"
            f"<br><span class='small'>Loss totals are from the sample path.</span>",
            unsafe_allow_html=True
        )
    except Exception:
        loss_meta_slot.markdown("")
else:
    _placeholder_losses = pd.DataFrame({
        "Losses": ["Personnel (KIA)", "Aircraft", "Tanks / AFVs", "Artillery", "Missiles Used"],
        "Blue Today": ["-", "-", "-", "-", "-"],
        "Blue Total": ["-", "-", "-", "-", "-"],
        "Red Today": ["-", "-", "-", "-", "-"],
        "Red Total": ["-", "-", "-", "-", "-"],
    })
    loss_table_slot.markdown(_render_losses_table(_placeholder_losses), unsafe_allow_html=True)
    loss_meta_slot.markdown("")


# ----------------------------
# Run scenario
# ----------------------------
run_now = (run_top or run_bottom)

if run_now and can_run:
    selected_core = str(st.session_state.get("core_model", "Cinematic"))
    scenario_type = str(st.session_state.get("scenario_type", "General"))
    auto_int = bool(st.session_state.get("auto_integrated", True))

    # Auto-switch to Integrated for invasion / blockade (keeps UI selection but uses Integrated under the hood)
    effective_core = selected_core
    if auto_int and scenario_type in ("Amphibious invasion / island landing", "Blockade / port strikes"):
        effective_core = "Integrated (4-engine)"

    core_model = "Integrated (4-engine)" if effective_core.startswith("Integrated") else ("Dense" if effective_core.startswith("Dense") else "Cinematic")

    bg_name = st.session_state["battleground"]
    bg_iso = iso_by_name.get(bg_name, (df["ISO3"].iloc[0] if len(df)>0 else ""))

    # Theatre mix (auto + override blend)
    bg_row = df[df["ISO3"] == bg_iso].iloc[0] if len(df[df["ISO3"] == bg_iso]) > 0 else df.iloc[0]
    a_auto, s_auto, l_auto = _auto_theatre_mix(bg_row, scenario_type)

    if bool(st.session_state.get("override_mix", False)):
        raw = np.array([float(st.session_state.get("mix_air", a_auto)),
                        float(st.session_state.get("mix_sea", s_auto)),
                        float(st.session_state.get("mix_land", l_auto))], dtype=float)
        raw = np.clip(raw, 1e-6, 1.0)
        raw = raw / float(raw.sum() + 1e-9)
        blend = float(np.clip(st.session_state.get("mix_blend", 0.85), 0.0, 1.0))
        mix = (1.0 - blend) * np.array([a_auto, s_auto, l_auto]) + blend * raw
        mix = mix / float(mix.sum() + 1e-9)
        a_sh, s_sh, l_sh = float(mix[0]), float(mix[1]), float(mix[2])
    else:
        a_sh, s_sh, l_sh = a_auto, s_auto, l_auto

    # Dense cfg
    dense_cfg = None
    if core_model == "Dense":
        dense_cfg = dense_default_cfg()
        dense_cfg.update(
            embed_strength=float(st.session_state.get("dense_embed_strength", 1.0)),
            fronts=int(st.session_state.get("dense_fronts", 1)),
            air_mult=float(st.session_state.get("dense_air_mult", 1.0)),
            sea_mult=float(st.session_state.get("dense_sea_mult", 1.0)),
            land_mult=float(st.session_state.get("dense_land_mult", 1.0)),
            missile_mult=float(st.session_state.get("dense_missile_mult", 1.0)),
            contact_mult=float(st.session_state.get("dense_contact_mult", 1.0)),
            civ_mult=float(st.session_state.get("dense_civ_mult", 1.0)),
            commit_mult=float(st.session_state.get("dense_commit_mult", 1.0)),
        )
        # let theatre mix subtly shape rates (keeps UI clean, behavior richer)
        dense_cfg["air_mult"] *= (0.80 + 0.60 * a_sh)
        dense_cfg["sea_mult"] *= (0.80 + 0.60 * s_sh)
        dense_cfg["land_mult"] *= (0.80 + 0.60 * l_sh)

    # Integrated cfg
    integrated_cfg = None
    if str(core_model).lower().startswith("integrated"):
        mf_auto = float(np.clip(0.40 + 0.55 * s_sh, 0.25, 0.90))
        det_auto = float(np.clip(0.35 + 0.50 * a_sh, 0.25, 0.90))
        integrated_cfg = dict(
            scenario_type=scenario_type,
            enable_orders=bool(st.session_state.get("ic_enable_orders", True)),
            aggression_blue=float(st.session_state.get("ic_aggression_blue", 0.55)),
            aggression_red=float(st.session_state.get("ic_aggression_red", 0.60)),

            blue_detection_prob=float(np.clip(0.70 * float(st.session_state.get("ic_blue_detection_prob", 0.62)) + 0.30 * det_auto, 0.15, 0.95)),
            missile_fraction_to_ships=float(np.clip(0.70 * float(st.session_state.get("ic_missile_fraction_to_ships", 0.55)) + 0.30 * mf_auto, 0.25, 0.90)),
            scuttle_day=int(st.session_state.get("ic_scuttle_day", 2)),
            use_roll_based_hits=bool(st.session_state.get("ic_use_roll_hits", False)),

            k_blue_air=float(st.session_state.get("ic_k_blue_air", 0.0020)),
            k_red_air=float(st.session_state.get("ic_k_red_air", 0.0020)),
            k_blue_land=float(st.session_state.get("ic_k_blue_land", 0.0010)),
            k_red_land=float(st.session_state.get("ic_k_red_land", 0.0010)),
        )

    sc = Scenario(
        scenario_type=scenario_type,
        year=int(st.session_state["year"]),
        intensity=str(st.session_state["intensity"]),
        mode=str(st.session_state["mode"]),
        max_days=int(st.session_state["max_days"]),
        battleground_iso3=str(bg_iso),
        sanctions_blue=float(st.session_state["sanctions_blue"]),
        sanctions_red=float(st.session_state["sanctions_red"]),
        nukes_allowed=bool(st.session_state["nukes_allowed"]),
        blue=blue_iso,
        red=red_iso,
        neutral_support_blue=support_blue_iso,
        neutral_support_red=support_red_iso,
        seed=int(st.session_state["seed"]),
        core_model=core_model,
        dense_cfg=dense_cfg,
        integrated_cfg=integrated_cfg,
    )

    # Optional model calibration (uses engine helpers)
    if sc.core_model == "Dense" and bool(st.session_state.get("dense_fit_requested", False)):
        try:
            targets = dict(st.session_state.get("dense_fit_targets", {}) or {})
            it_fit = int(st.session_state.get("dense_fit_iters", 150))
            fitted = fit_dense_cfg(bundle, sc, targets=targets, iters=it_fit, seed=int(sc.seed))
            sc.dense_cfg = fitted
        except Exception:
            pass

    if str(sc.core_model).lower().startswith("integrated") and bool(st.session_state.get("ic_fit_requested", False)):
        try:
            targets2 = dict(st.session_state.get("ic_fit_targets", {}) or {})
            it_fit2 = int(st.session_state.get("ic_fit_iters", 1200))
            best_cfg, _loss = fit_integrated_cfg(bundle, sc, targets=targets2, iters=it_fit2, mc=3, seed=int(sc.seed))
            if sc.integrated_cfg is None:
                sc.integrated_cfg = {}
            sc.integrated_cfg.update(best_cfg or {})
        except Exception:
            pass

    # Reset duel states for this run
    st.session_state["chess_state"] = _start_chess_state()
    st.session_state["phase_state"] = _make_phase_state(int(sc.seed) + 1337)

    # 1) Cinematic run -> populate the Battle Losses table immediately
    with st.spinner("Running cinematic scenario..."):
        summary, series = simulate_cinematic(bundle, sc)
    st.session_state["last_summary"] = summary
    st.session_state["last_series"] = series

    tb = summary["totals"]["blue"]
    tr = summary["totals"]["red"]

    # Today deltas (use last two days if available)
    b_today_kia = r_today_kia = 0
    b_today_air = r_today_air = 0
    b_today_ship = r_today_ship = 0
    b_today_tank = r_today_tank = 0
    b_today_art = r_today_art = 0
    b_today_miss = r_today_miss = 0

    try:
        s = pd.DataFrame(series).sort_values("day")
        if len(s) >= 2:
            a = s.iloc[-2].to_dict()
            b = s.iloc[-1].to_dict()
            b_today_kia = int(max(0, (b.get("blue_total_kia",0) or 0) - (a.get("blue_total_kia",0) or 0)))
            r_today_kia = int(max(0, (b.get("red_total_kia",0) or 0) - (a.get("red_total_kia",0) or 0)))
            # remaining -> losses today (approx)
            b_today_air = int(max(0, round((a.get("blue_aircraft_remaining",0) or 0) - (b.get("blue_aircraft_remaining",0) or 0))))
            r_today_air = int(max(0, round((a.get("red_aircraft_remaining",0) or 0) - (b.get("red_aircraft_remaining",0) or 0))))
            b_today_ship = int(max(0, round((a.get("blue_ships_remaining",0) or 0) - (b.get("blue_ships_remaining",0) or 0))))
            r_today_ship = int(max(0, round((a.get("red_ships_remaining",0) or 0) - (b.get("red_ships_remaining",0) or 0))))
            b_today_tank = int(max(0, round((a.get("blue_tanks_remaining",0) or 0) - (b.get("blue_tanks_remaining",0) or 0))))
            r_today_tank = int(max(0, round((a.get("red_tanks_remaining",0) or 0) - (b.get("red_tanks_remaining",0) or 0))))
            # missiles used today (if remaining is tracked)
            b_today_miss = int(max(0, round((a.get("blue_missiles_remaining",0) or 0) - (b.get("blue_missiles_remaining",0) or 0))))
            r_today_miss = int(max(0, round((a.get("red_missiles_remaining",0) or 0) - (b.get("red_missiles_remaining",0) or 0))))
            # no artillery daily series in cinematic; estimate 0
    except Exception:
        pass

    loss_table = pd.DataFrame({
        "Losses": ["Personnel (KIA)", "Aircraft", "Tanks / AFVs", "Artillery", "Missiles Used"],
        "Blue Today": [b_today_kia, b_today_air, b_today_tank, b_today_art, b_today_miss],
        "Blue Total": [tb.get("kia",0), tb.get("aircraft",0), tb.get("tanks",0), tb.get("artillery",0), int(tb.get("missiles",0))],
        "Red Today":  [r_today_kia, r_today_air, r_today_tank, r_today_art, r_today_miss],
        "Red Total":  [tr.get("kia",0), tr.get("aircraft",0), tr.get("tanks",0), tr.get("artillery",0), int(tr.get("missiles",0))],
    })
    st.session_state["pending_loss_table"] = loss_table
    st.session_state["pending_sample_summary"] = {"winner": summary.get("winner","Draw"), "end_reason": summary.get("end_reason","")}
    # Hold the Battle Losses panel until MC+duel finish (prevents partial / inconsistent readouts)
    loss_table_slot.markdown("<span class='small'>Running duel… Battle losses will appear when the duel completes.</span>", unsafe_allow_html=True)
    loss_meta_slot.markdown("", unsafe_allow_html=True)

    # 2) Monte Carlo with Live Duel
    iters = int(st.session_state["iters"])
    duel_plies = int(st.session_state["duel_plies"])
    duel_depth = int(st.session_state["duel_depth"])
    speed = float(st.session_state["sim_speed"])

    # Choose chunk count to get ~80-140 updates (smooth but not spammy)
    target_frames = int(60 + 80*speed)  # 60..140
    duel_chunk = int(max(2_000, min(200_000, iters // max(1, target_frames))))

    prog = st.progress(0.0)
    prog_txt = st.empty()

    # UCI chess engine (optional). If unavailable, we fall back to the built-in minimax.
    uci = _get_uci_client()
    try:
        if uci is not None:
            uci.newgame()
            # Hash sizing scales with Duel depth (UI unchanged).
            hash_mb = {1: 96, 2: 192, 3: 384}.get(int(st.session_state.get('duel_depth', 2)), 192)
            uci.setoption('Hash', str(hash_mb))
            # Optional: book/tablebases via env vars or local defaults
            book = os.environ.get('WARCOLOR_CHESS_BOOK', '').strip() or ('book.bin' if os.path.exists('book.bin') else '')
            tb = os.environ.get('WARCOLOR_SYZYGY', '').strip() or ('syzygy' if os.path.isdir('syzygy') else '')
            # Only send options if your engine supports them; your python_engine ignores unknown options safely.
            if book:
                uci.setoption('BookFile', book)
            if tb:
                uci.setoption('SyzygyPath', tb)
    except Exception:
        uci = None

    def _progress(done: int, total: int, payload: Optional[dict] = None):
        frac = 0.0 if total <= 0 else float(np.clip(done / total, 0.0, 1.0))
        prog.progress(frac)
        prog_txt.markdown(f"**Progress:** {frac*100:.1f}%  ({done:,}/{total:,})")

        # synth payload if missing
        if payload is None:
            rng = np.random.default_rng(int(sc.seed) + 202)
            drift = float(rng.normal(0.0, 0.08))
            pB = float(np.clip(0.50 + drift*(frac - 0.25), 0.10, 0.90))
            payload = {"win_prob": {"Blue": pB, "Red": 1.0-pB, "Draw": 0.0}}

        # Chess step to target ply
        cs = st.session_state["chess_state"]

        # Pull MC probabilities (truth channel)
        wp = (payload or {}).get("win_prob", {}) or {}
        pB_mc = float(wp.get("Blue", 0.5) or 0.5)
        pR_mc = float(wp.get("Red", 0.5) or 0.5)
        pD_mc = float((payload or {}).get("drawness", wp.get("Draw", max(0.0, 1.0 - pB_mc - pR_mc))) or 0.0)
        tot = pB_mc + pR_mc + pD_mc
        if tot > 1e-9:
            pB_mc, pR_mc, pD_mc = pB_mc/tot, pR_mc/tot, pD_mc/tot

        # Current chess eval (White POV) used for fusion
        cpw = float(st.session_state.get("_chess_cp_white", 0.0) or 0.0)

        # Fused win mix (small correction; MC stays dominant)
        pB_f, pR_f, pD_f, p_final_br, p_chess_br = _fuse_mc_with_chess(pB_mc, pR_mc, pD_mc, cpw, lam=0.35)

        target_ply = int(frac * duel_plies)

        # Cap moves per progress update to keep UI smooth
        sim_speed = float(st.session_state.get("sim_speed", 0.60))
        max_steps = 2 + int(6 * sim_speed)

        while (len(cs.moves) < target_ply) and (not cs.result) and (max_steps > 0):
            # Refresh fusion inputs each ply (keeps pacing coherent)
            cpw = float(st.session_state.get("_chess_cp_white", 0.0) or 0.0)
            pB_f, pR_f, pD_f, p_final_br, p_chess_br = _fuse_mc_with_chess(pB_mc, pR_mc, pD_mc, cpw, lam=0.35)

            mv = None
            best_uci = None

            if uci is not None:
                fen = _fen_from_cs(cs)

                # Base strength tier from the existing UI Duel depth (UI unchanged)
                base_depth = {1: 8, 2: 10, 3: 12}.get(int(duel_depth), 10)

                d_adj = 0
                try:
                    inten = str(sc.intensity).strip().lower()
                    if inten.startswith("skirmish"):
                        d_adj -= 1
                    if inten.startswith("all"):
                        d_adj += 1
                except Exception:
                    pass
                if sim_speed >= 0.85:
                    d_adj -= 1

                # Side-to-move catch-up: losing side gets a bit more compute
                p_side = p_final_br if cs.white else (1.0 - p_final_br)
                if p_side < 0.45:
                    d_adj += 1

                depth = int(np.clip(base_depth + d_adj, 6, 14))

                # Time budget derived from Simulation speed (fast slider -> smaller think time)
                movetime = (45 + 220 * (1.0 - sim_speed))
                movetime *= {1: 0.85, 2: 1.20, 3: 1.70}.get(int(duel_depth), 1.20)
                movetime *= (1.0 + 0.45 * max(0.0, 0.50 - p_side))  # losing => more time
                movetime_ms = int(np.clip(movetime, 25, 900))

                # If we need to jump several plies this tick, run a fast catch-up search
                if (target_ply - len(cs.moves)) > 2:
                    movetime_ms = int(max(25, movetime_ms * 0.38))
                    depth = int(max(6, depth - 2))

                best_uci, cp_stm = uci.bestmove(fen, depth=depth, movetime_ms=movetime_ms)
                mv = _mv_from_uci(cs, best_uci)

                # Store evaluation (convert from side-to-move POV to White POV)
                if cp_stm is not None:
                    st.session_state["_chess_cp_white"] = float(cp_stm if cs.white else -cp_stm)

            # Fallback: built-in minimax if no UCI engine or parse failed
            if mv is None:
                wW, wB, _pB0, _pR0, _pD0 = _weights_from_payload(payload)
                _, mv2, term = _minimax(cs, duel_depth, -1e9, 1e9, wW, wB)
                if mv2 is None:
                    cs.result = term or (_terminal_result(cs) or "Draw")
                    break
                mv = mv2
                best_uci = _uci(mv)

            cs = _apply_move(cs, mv)
            u = best_uci or _uci(mv)
            cs.moves.append(u)
            cs.last_uci = u
            t2 = _terminal_result(cs)
            if t2:
                cs.result = t2
            max_steps -= 1

        st.session_state["chess_state"] = cs

        chess_slot.markdown(_render_board_html(cs.board, cs.last_uci), unsafe_allow_html=True)
        last_line = " ".join(cs.moves[-14:]) if cs.moves else "(opening)"

        # Recompute MC probabilities (truth) + fused mix (small correction)
        wp = (payload or {}).get("win_prob", {}) or {}
        pB_mc = float(wp.get("Blue", 0.5) or 0.5)
        pR_mc = float(wp.get("Red", 0.5) or 0.5)
        pD_mc = float((payload or {}).get("drawness", wp.get("Draw", max(0.0, 1.0 - pB_mc - pR_mc))) or 0.0)
        tot = pB_mc + pR_mc + pD_mc
        if tot > 1e-9:
            pB_mc, pR_mc, pD_mc = pB_mc/tot, pR_mc/tot, pD_mc/tot

        cpw = float(st.session_state.get("_chess_cp_white", 0.0) or 0.0)
        pB_f, pR_f, pD_f, p_final_br, p_chess_br = _fuse_mc_with_chess(pB_mc, pR_mc, pD_mc, cpw, lam=0.35)

        edge_lbl_mc = _winner_label(pB_mc, pR_mc, pD_mc)
        edge_amt_mc = int(round(abs(pB_mc - pR_mc) * 100.0))
        chess_pawns = float(cpw) / 100.0
        chess_p_pct = int(round(p_chess_br * 100.0))

        chess_meta.markdown(
            f"**Plies:** {len(cs.moves)}/{duel_plies}  •  "
            f"<span class='small'>MC edge: {edge_lbl_mc} {edge_amt_mc}%</span>  •  "
            f"<span class='small'>Chess: {chess_pawns:+.2f} ({chess_p_pct}%)</span><br>"
            f"<span class='small'>{last_line}</span>",
            unsafe_allow_html=True
        )

        # Phase uses the fused mix for the visual, but draw channel remains MC truth.
        phase_payload = dict(payload or {})
        phase_payload["win_prob"] = {"Blue": float(pB_f), "Red": float(pR_f), "Draw": float(pD_f)}
        phase_payload["phase_adv"] = float(np.clip(pB_f - pR_f, -1.0, 1.0))
        phase_payload["drawness"] = float(np.clip(pD_mc, 0.0, 1.0))

        fig = _render_phase(st.session_state["phase_state"], frac, phase_payload, int(sc.seed) + 1337)
        fluid_slot.pyplot(fig, use_container_width=True)
        plt.close(fig)

        _bf, _rf, _df = _pct_int_triplet(pB_f, pR_f, pD_f)
        _bm, _rm, _dm = _pct_int_triplet(pB_mc, pR_mc, pD_mc)
        fluid_meta.markdown(f"**Win mix:** Blue {_bf}% • Red {_rf}% • Draw {_df}%  <span class='small'>(MC: {_bm}/{_rm}/{_dm})</span>", unsafe_allow_html=True)

        return {}

    with st.spinner("Running Monte Carlo distribution (synced to duel)..."):
        if core_model.lower().startswith("integrated"):
            mc = monte_carlo_integrated_fast(bundle, sc, iters, chunk=duel_chunk, seed=int(sc.seed), progress_callback=_progress)
        else:
            mc = monte_carlo_fast(bundle, sc, iters, chunk=duel_chunk, seed=int(sc.seed), progress_callback=_progress)

    # Force a final visual update (ensures the phase panel shows separation even if the last callback was skipped)
    try:
        _progress(int(iters), int(iters), {"win_prob": (mc.get("win_prob", {}) or {})})
    except Exception:
        pass

    prog.progress(1.0)
    prog_txt.markdown("**Progress:** 100%")
    st.session_state["last_mc"] = mc
    try:
        _wp3 = (mc.get("win_prob", {}) or {})
        pB3 = float(_wp3.get("Blue", 0.5) or 0.5)
        pR3 = float(_wp3.get("Red", 0.5) or 0.5)
        pD3 = float(_wp3.get("Draw", 0.0) or max(0.0, 1.0 - pB3 - pR3))
        _bi3, _ri3, _di3 = _pct_int_triplet(pB3, pR3, pD3)
        _w3 = _mc_winner(pB3, pR3, pD3)
        cpw3 = float(st.session_state.get("_chess_cp_white", 0.0) or 0.0)
        pBf3, pRf3, pDf3, _pbr3, _pch3 = _fuse_mc_with_chess(pB3, pR3, pD3, cpw3, lam=0.35)
        _bf3, _rf3, _df3 = _pct_int_triplet(pBf3, pRf3, pDf3)
        st.session_state["last_fused_win_prob"] = {"Blue": float(pBf3), "Red": float(pRf3), "Draw": float(pDf3)}

        # Now that MC+duel are done, render the Battle Losses table (whole numbers).
        _lt = st.session_state.get("pending_loss_table", None)
        if _lt is not None:
            loss_table_slot.markdown(_render_losses_table(_lt), unsafe_allow_html=True)

        # Single source of truth: Monte Carlo drives Outcome + phase buoyancy.
        loss_meta_slot.markdown(
            f"**Outcome:** {_w3}  ·  <span class='small'>MC: Blue {_bi3}% · Red {_ri3}% · Draw {_di3}%</span>"
            f"<br><span class='small'>Fused (MC ⊕ Chess): Blue {_bf3}% · Red {_rf3}% · Draw {_df3}%</span>"
            f"<br><span class='small'>Battle losses are from the cinematic sample path.</span>",
            unsafe_allow_html=True
        )
    except Exception:
        pass
