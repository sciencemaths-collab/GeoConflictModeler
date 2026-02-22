
# warcolor_wargame_global_realism/engine.py
# A cinematic + Monte Carlo wargame engine (fictional educational simulator).
# Design goals: realism-inspired constraints (projection, distance, A2/AD), daily timestep,
# inventory-driven loss proxies, sanctions + mobilization curves, tech leaps (2032/2036),
# optional nuclear escalation as macro-shock, alliance/country breakdowns.

from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd



# -----------------
# Job-Done / Logistics add-on helpers (145-country matrices)
# -----------------

_DAYS_PER_MONTH = 30.42  # smoother curves in long-term sims (avg days/month)

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", "and")
    for ch in [".", ",", "’", "'", "(", ")", "-", "—", "–"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

_NAME_ALIASES = {
    "usa": "united states",
    "u s a": "united states",
    "u s": "united states",
    "uk": "united kingdom",
    "u k": "united kingdom",
    "s korea": "south korea",
    "s. korea": "south korea",
    "korea": "south korea",
    "uae": "united arab emirates",
    "u a e": "united arab emirates",
    "drc": "dr congo",
    "dr congo": "congo dem rep",
    "czech republic": "czechia",
    "turkiye": "türkiye",
}

def _to_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _parse_days(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    s = s.replace("days", "").replace("day", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def _parse_km_token(tok: str) -> Optional[float]:
    tok = (tok or "").strip().lower()
    if not tok:
        return None
    mult = 1.0
    if tok.endswith("k"):
        mult = 1_000.0
        tok = tok[:-1].strip()
    if tok.endswith("m"):
        mult = 1_000_000.0
        tok = tok[:-1].strip()
    try:
        return float(tok) * mult
    except Exception:
        return None

def _parse_wmy(cell) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse strings like '10 / 45 / 540' or '4k / 18k / 218k' -> (W,M,Y)."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return (None, None, None)
    if isinstance(cell, (int, float)):
        # treat as yearly if only one number appears
        v = float(cell)
        return (None, None, v)
    s = str(cell).strip()
    parts = [p.strip() for p in s.split("/") if p.strip()]
    if len(parts) == 0:
        return (None, None, None)
    # allow "a / b / c"
    vals = [_parse_km_token(p) for p in parts[:3]]
    while len(vals) < 3:
        vals.append(None)
    return tuple(vals[:3])

def _parse_triplet(cell) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse '.95/.98/.99' -> (D,W,M)."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return (None, None, None)
    if isinstance(cell, (int, float)):
        v = float(cell)
        return (v, v, v)
    s = str(cell).strip()
    parts = [p.strip() for p in s.split("/") if p.strip()]
    vals = []
    for p in parts[:3]:
        try:
            vals.append(float(p))
        except Exception:
            vals.append(None)
    while len(vals) < 3:
        vals.append(None)
    return tuple(vals[:3])

def _safe_weighted_avg(df: pd.DataFrame, iso_list: List[str], col: str, wcol: str) -> float:
    sub = df[df["ISO3"].isin(iso_list)]
    if len(sub) == 0 or col not in sub.columns:
        return float("nan")
    x = pd.to_numeric(sub[col], errors="coerce")
    w = pd.to_numeric(sub.get(wcol), errors="coerce")
    if w is None:
        return float(np.nanmean(x))
    w = w.fillna(0.0)
    x = x.fillna(np.nan)
    if float(w.sum()) <= 1e-9:
        return float(np.nanmean(x))
    return float(np.nansum(x * w) / (np.nansum(w) + 1e-9))

def _merge_job_done_addons(df: pd.DataFrame, name_to_iso3: Dict[str, str], data_dir: str) -> pd.DataFrame:
    """Merge the 145-country Job-Done + Logistics matrices if present."""
    def map_country(name: str) -> Optional[str]:
        n = _norm_name(name)
        n = _NAME_ALIASES.get(n, n)
        if n in name_to_iso3:
            return name_to_iso3[n]
        # try a few simplifications
        n2 = n.replace("south ", "").replace("north ", "")
        if n2 in name_to_iso3:
            return name_to_iso3[n2]
        return None

    # Military Capability Analysis Framework
    mpath = os.path.join(data_dir, "Military Capability Analysis Framework.xlsx")
    if os.path.exists(mpath):
        m = pd.read_excel(mpath)
        m["ISO3"] = m["Country"].apply(map_country)
        for col, prefix in [
            ("Tanks (W/M/Y)", "jd_tanks"),
            ("Jets (W/M/Y)", "jd_jets"),
            ("Drones (W/M/Y)", "jd_drones"),
            ("Missiles (W/M/Y)", "jd_missiles"),
        ]:
            wmy = m[col].apply(_parse_wmy)
            m[f"{prefix}_W"] = [t[0] for t in wmy]
            m[f"{prefix}_M"] = [t[1] for t in wmy]
            m[f"{prefix}_Y"] = [t[2] for t in wmy]
            m[f"{prefix}_Pd_W"] = m[f"{prefix}_W"] / 7.0
            m[f"{prefix}_Pd_M"] = m[f"{prefix}_M"] / float(_DAYS_PER_MONTH)
            m[f"{prefix}_Pd_Y"] = m[f"{prefix}_Y"] / 365.0

        m["jd_Tech_0_10"] = pd.to_numeric(m.get("Tech"), errors="coerce")
        m["jd_Prod_0_10"] = pd.to_numeric(m.get("Prod"), errors="coerce")
        m["jd_CGJD_0_10"] = pd.to_numeric(m.get("CGJD"), errors="coerce")
        keep = ["ISO3", "jd_Tech_0_10", "jd_Prod_0_10", "jd_CGJD_0_10"] + [c for c in m.columns if c.startswith("jd_")]
        m = m[keep].dropna(subset=["ISO3"]).drop_duplicates("ISO3", keep="first")
        df = df.merge(m, on="ISO3", how="left")

    # Logistics & Endurance Matrix
    lpath = os.path.join(data_dir, "2026 Logistics & Endurance Matrix.xlsx")
    if os.path.exists(lpath):
        L = pd.read_excel(lpath)
        L["ISO3"] = L["Country"].apply(map_country)

        for col, prefix in [
            ("Ground (D/W/M)", "lmc_ground"),
            ("Air/Drone (D/W/M)", "lmc_air"),
            ("Naval (D/W/M)", "lmc_naval"),
            ("Cyber (D/W/M)", "lmc_cyber"),
        ]:
            tri = L[col].apply(_parse_triplet)
            L[f"{prefix}_D"] = [t[0] for t in tri]
            L[f"{prefix}_W"] = [t[1] for t in tri]
            L[f"{prefix}_M"] = [t[2] for t in tri]

        L["hub_score_0_10"] = pd.to_numeric(L.get("Hub Score"), errors="coerce")
        L["endurance_ammo_days"] = L.get("High-End Munitions (Days)").apply(_parse_days)
        L["endurance_fuel_days"] = L.get("Fuel / Energy (Days)").apply(_parse_days)
        L["endurance_parts_days"] = L.get("Spare Parts (Days)").apply(_parse_days)
        L["endurance_total_days"] = L.get("Total Endurance").apply(_parse_days)

        keep = ["ISO3","hub_score_0_10","endurance_ammo_days","endurance_fuel_days","endurance_parts_days","endurance_total_days"] + [c for c in L.columns if c.startswith("lmc_")]
        L = L[keep].dropna(subset=["ISO3"]).drop_duplicates("ISO3", keep="first")
        df = df.merge(L, on="ISO3", how="left")

    # Fill missing JD proxies from existing score columns (0..100)
    if "jd_Tech_0_10" not in df.columns:
        df["jd_Tech_0_10"] = np.nan
    if "jd_Prod_0_10" not in df.columns:
        df["jd_Prod_0_10"] = np.nan
    if "jd_CGJD_0_10" not in df.columns:
        df["jd_CGJD_0_10"] = np.nan

    # Proxies when missing
    tech_proxy = 1.0 + 0.09 * (0.6*pd.to_numeric(df.get("networks_score"), errors="coerce") + 0.4*pd.to_numeric(df.get("future_resources_score"), errors="coerce"))
    prod_proxy = 1.0 + 0.09 * (0.7*pd.to_numeric(df.get("economy_score"), errors="coerce") + 0.3*pd.to_numeric(df.get("future_resources_score"), errors="coerce"))
    cgjd_proxy = 1.0 + 0.09 * (0.4*pd.to_numeric(df.get("milcap_score"), errors="coerce") + 0.3*pd.to_numeric(df.get("logistics_score"), errors="coerce") + 0.3*pd.to_numeric(df.get("resilience_score"), errors="coerce"))

    df["jd_Tech_0_10"] = pd.to_numeric(df["jd_Tech_0_10"], errors="coerce").fillna(tech_proxy).clip(1.0, 10.0)
    df["jd_Prod_0_10"] = pd.to_numeric(df["jd_Prod_0_10"], errors="coerce").fillna(prod_proxy).clip(1.0, 10.0)
    df["jd_CGJD_0_10"] = pd.to_numeric(df["jd_CGJD_0_10"], errors="coerce").fillna(cgjd_proxy).clip(1.0, 10.0)

    # LMC proxies
    for pref in ["lmc_ground","lmc_air","lmc_naval","lmc_cyber"]:
        for t in ["D","W","M"]:
            col = f"{pref}_{t}"
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.85).clip(0.25, 1.0)

    if "hub_score_0_10" not in df.columns:
        df["hub_score_0_10"] = np.nan
    df["hub_score_0_10"] = pd.to_numeric(df["hub_score_0_10"], errors="coerce").fillna((pd.to_numeric(df.get("logistics_score"), errors="coerce")/10.0)).clip(0.0, 10.0)

    for col in ["endurance_ammo_days","endurance_fuel_days","endurance_parts_days","endurance_total_days"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(30.0).clip(1.0, 365.0)

    return df

def _jd_base_intensity(sc) -> float:
    base = {"Skirmish": 1.0, "Campaign": 2.5, "All-out": 4.5}.get(str(getattr(sc, "intensity", "Campaign")), 2.5)
    if str(getattr(sc, "mode", "Limited War")).lower().startswith("total"):
        base *= 1.10
    return float(np.clip(base, 0.5, 5.0))

def _jd_war_type(scenario_type: str) -> str:
    s = (scenario_type or "General").strip().lower()
    if "naval" in s or "blockade" in s or "port" in s:
        return "Naval War"
    if "air" in s or "drone" in s:
        return "Air / Drone"
    if "cyber" in s or "elect" in s:
        return "Cyber / Electronic"
    if "land" in s or "ground" in s or "amphib" in s or "invasion" in s:
        return "Ground War"
    return "Ground War"

def _jd_weights_from_sheet(data_dir: str) -> Dict[str, Tuple[float,float,float]]:
    # defaults
    out = {
        "Ground War": (0.50, 0.30, 0.20),
        "Naval War": (0.20, 0.25, 0.55),
        "Air / Drone": (0.30, 0.45, 0.25),
        "Cyber / Electronic": (0.05, 0.15, 0.80),
    }
    sp = os.path.join(data_dir, "Scenario-Adjusted Military Strength Algorithm.xlsx")
    if not os.path.exists(sp):
        return out
    try:
        s = pd.read_excel(sp)
        for _, r in s.iterrows():
            name = str(r.get("Scenario","")).strip()
            if name in out:
                out[name] = (float(r.get("Wq​ (Quantity)", r.get("Wq (Quantity)", out[name][0]))),
                             float(r.get("Wp​ (Production)", r.get("Wp (Production)", out[name][1]))),
                             float(r.get("Wt​ (Technology)", r.get("Wt (Technology)", out[name][2]))))
    except Exception:
        return out
    return out

def _jd_init_side(df: pd.DataFrame, iso_list: List[str], supporter_iso_list: Optional[List[str]] = None, wcol: str = "mil_exp_usd") -> dict:
    """Initialize Job-Done state for a side.

    We include neutral supporters at a reduced weight (mirrors _side_aggregate),
    and expose a has_supporters flag so hub "leach" bonuses only apply when allies exist.
    """
    core = [x for x in (iso_list or []) if x]
    sup = [x for x in (supporter_iso_list or []) if x and x not in core]
    has_supporters = bool(sup)

    # Weighted average across core + supporters (supporters contribute less)
    def wavg(col: str, default: float = 7.0) -> float:
        sub = df[df["ISO3"].isin(core + sup)].copy()
        if len(sub) == 0 or col not in sub.columns:
            return float(default)
        x = pd.to_numeric(sub[col], errors="coerce").fillna(np.nan)
        w = pd.to_numeric(sub.get(wcol), errors="coerce").fillna(0.0)
        mult = sub["ISO3"].isin(core).astype(float) * 1.0 + sub["ISO3"].isin(sup).astype(float) * 0.35
        ww = (w * mult).to_numpy(dtype=float)
        xx = x.to_numpy(dtype=float)
        denom = float(np.nansum(ww))
        if denom <= 1e-9:
            m = float(np.nanmean(xx))
            return float(m if np.isfinite(m) else default)
        m = float(np.nansum(xx * ww) / denom)
        return float(m if np.isfinite(m) else default)

    tech = wavg("jd_Tech_0_10", 7.0)
    prod = wavg("jd_Prod_0_10", 7.0)
    cgjd = wavg("jd_CGJD_0_10", 7.0)
    hub = wavg("hub_score_0_10", 5.0)

    # endurance (days)
    ammo = wavg("endurance_ammo_days", 30.0)
    fuel = wavg("endurance_fuel_days", 30.0)
    parts = wavg("endurance_parts_days", 30.0)
    etot = float(np.nanmin([ammo, fuel, parts])) if np.isfinite([ammo,fuel,parts]).any() else 30.0

    # production daily by timeframe (weighted sum)
    def sum_col(c):
        sub = df[df["ISO3"].isin(core + sup)].copy()
        if len(sub) == 0 or c not in sub.columns:
            return 0.0
        x = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)
        w = pd.to_numeric(sub.get(wcol), errors="coerce").fillna(0.0)
        mult = sub["ISO3"].isin(core).astype(float) * 1.0 + sub["ISO3"].isin(sup).astype(float) * 0.35
        ww = (w * mult)
        if float(ww.sum()) <= 1e-9:
            return float(x.sum())
        return float((x * ww).sum() / (ww.sum() + 1e-9))
    prods = {}
    for pref in ["jd_tanks","jd_jets","jd_drones","jd_missiles"]:
        for suf in ["Pd_W","Pd_M","Pd_Y"]:
            prods[f"{pref}_{suf}"] = sum_col(f"{pref}_{suf}")
    # lmc
    lmc = {}
    for pref in ["lmc_ground","lmc_air","lmc_naval","lmc_cyber"]:
        for suf in ["D","W","M"]:
            lmc[f"{pref}_{suf}"] = wavg(f"{pref}_{suf}", 0.85)

    return dict(
        tech=float(tech if np.isfinite(tech) else 7.0),
        prod=float(prod if np.isfinite(prod) else 7.0),
        cgjd=float(cgjd if np.isfinite(cgjd) else 7.0),
        hub=float(hub if np.isfinite(hub) else 5.0),
        ammo=float(ammo if np.isfinite(ammo) else 30.0),
        fuel=float(fuel if np.isfinite(fuel) else 30.0),
        parts=float(parts if np.isfinite(parts) else 30.0),
        ammo0=float(ammo if np.isfinite(ammo) else 30.0),
        fuel0=float(fuel if np.isfinite(fuel) else 30.0),
        parts0=float(parts if np.isfinite(parts) else 30.0),
        endurance0=float(etot if np.isfinite(etot) else 30.0),
        S=1.0,
        prods=prods,
        lmc=lmc,
        has_supporters=has_supporters,
    )

def _jd_timeframe(day: int) -> str:
    if day <= 28:
        return "short"
    if day <= 180:
        return "mid"
    return "long"

def _jd_pick_prod(jd: dict, key_base: str, day: int) -> float:
    tf = _jd_timeframe(day)
    if tf == "short":
        return float(jd["prods"].get(f"{key_base}_Pd_W", 0.0) or 0.0)
    if tf == "mid":
        return float(jd["prods"].get(f"{key_base}_Pd_M", 0.0) or 0.0)
    return float(jd["prods"].get(f"{key_base}_Pd_Y", 0.0) or 0.0)

def _jd_lmc_overall(jd: dict, scenario_type: str, day: int) -> float:
    tf = _jd_timeframe(day)
    suf = "D" if tf=="short" else ("W" if tf=="mid" else "M")
    l = jd["lmc"]
    g = float(l.get(f"lmc_ground_{suf}", 0.85) or 0.85)
    a = float(l.get(f"lmc_air_{suf}", 0.85) or 0.85)
    n = float(l.get(f"lmc_naval_{suf}", 0.85) or 0.85)
    c = float(l.get(f"lmc_cyber_{suf}", 0.85) or 0.85)
    # war type focus shifts weights
    wt = _jd_war_type(scenario_type)
    if wt == "Naval War":
        w = (0.20, 0.25, 0.40, 0.15)
    elif wt == "Air / Drone":
        w = (0.15, 0.45, 0.20, 0.20)
    elif wt == "Cyber / Electronic":
        w = (0.10, 0.25, 0.10, 0.55)
    else:
        w = (0.40, 0.20, 0.15, 0.25)
    return float(np.clip(w[0]*g + w[1]*a + w[2]*n + w[3]*c, 0.25, 1.05))

def _jd_pre_mult(jd: dict, day: int, base_id: float, remaining: dict, init_rem: dict, scenario_type: str) -> dict:
    # readiness proxy from remaining ratios + supply
    def ratio(k):
        a = float(init_rem.get(k, 1.0) or 1.0)
        b = float(remaining.get(k, 0.0) or 0.0)
        return float(np.clip(b / max(1e-9, a), 0.0, 1.0))
    air_r = ratio("aircraft")
    land_r = ratio("tanks")
    sea_r = ratio("ships") if float(init_rem.get("ships",0) or 0) > 0 else 0.65
    supply_r = float(np.clip(min(jd["ammo"]/max(1e-6,jd["ammo0"]), jd["fuel"]/max(1e-6,jd["fuel0"]), jd["parts"]/max(1e-6,jd["parts0"])), 0.0, 1.0))
    S = float(np.clip(0.45*air_r + 0.35*land_r + 0.20*sea_r, 0.0, 1.0))
    jd["S"] = float(np.clip(0.75*jd.get("S",1.0) + 0.25*S, 0.0, 1.0))

    Id = float(base_id)
    # Strained/collapse intensity shaping
    if 0.25 < jd["S"] < 0.50:
        Id = min(Id, 2.0)
    if jd["S"] < 0.10 or supply_r < 0.10:
        Id = 0.1
    Id = float(np.clip(Id, 0.1, 5.0))

    # multipliers for combat effectiveness
    lmc = _jd_lmc_overall(jd, scenario_type, day)
    drone_pd = _jd_pick_prod(jd, "jd_drones", day)
    drone_mult = 1.0 + 0.030 * math.log10(1.0 + max(0.0, drone_pd))
    readiness_mult = 0.55 + 0.45*jd["S"]
    intensity_mult = float(np.clip(Id/2.5, 0.15, 2.2))
    can_mult = float(np.clip(lmc * drone_mult * readiness_mult * intensity_mult, 0.10, 2.6))
    return {"Id": Id, "can_mult": can_mult, "drone_pd": drone_pd, "lmc": lmc, "supply_r": supply_r}

def _jd_apply_post(jd: dict, day: int, Id: float, remaining: dict, init_rem: dict, losses_today: dict, scenario_type: str):
    # Repair rate (Return-to-duty)
    tech = float(jd.get("tech", 7.0))
    rtd = 0.85 if tech > 8.5 else 1.0
    repair_rate = 0.15 if tech > 8.5 else 0.0

    # Production selection
    tanks_pd = _jd_pick_prod(jd, "jd_tanks", day)
    jets_pd = _jd_pick_prod(jd, "jd_jets", day)
    missiles_pd = _jd_pick_prod(jd, "jd_missiles", day)
    drones_pd = _jd_pick_prod(jd, "jd_drones", day)

    # Production multipliers
    prod_mult = 1.0
    if jd["S"] > 0.80:
        prod_mult *= 1.2
    # Hub "gravity": a strong hub helps output even alone, but the full +20% "leach" bonus
    # only triggers if neutral supporters exist (mirrors the design spec).
    if jd.get("hub", 0.0) > 7.0:
        prod_mult *= (1.2 if bool(jd.get("has_supporters", False)) else 1.10)
    # cyber decay
    tf = _jd_timeframe(day)
    suf = "D" if tf=="short" else ("W" if tf=="mid" else "M")
    cyber = float(jd["lmc"].get(f"lmc_cyber_{suf}", 0.85) or 0.85)
    if cyber < 0.5:
        prod_mult *= 0.85

    # supply factor gates production
    supply_r = float(np.clip(min(jd["ammo"]/max(1e-6,jd["ammo0"]), jd["fuel"]/max(1e-6,jd["fuel0"]), jd["parts"]/max(1e-6,jd["parts0"])), 0.0, 1.0))
    prod_mult *= (0.25 + 0.75*supply_r)

    # collapse mode
    if jd["S"] < 0.10 or supply_r < 0.10:
        prod_mult *= 0.2

    # attrition cliff
    if day >= 180 and jd["parts"] < 10:
        prod_mult *= 0.5

    # Apply repairs (restore fraction of losses)
    for k in ["aircraft", "tanks", "ships", "artillery"]:
        lost = float(losses_today.get(k, 0.0) or 0.0)
        if repair_rate > 0:
            remaining[k] = float(remaining.get(k, 0.0) or 0.0) + repair_rate*lost

    # Apply production
    remaining["tanks"] = float(remaining.get("tanks", 0.0) or 0.0) + prod_mult * float(max(0.0, tanks_pd))
    remaining["aircraft"] = float(remaining.get("aircraft", 0.0) or 0.0) + prod_mult * float(max(0.0, jets_pd))
    remaining["missiles"] = float(remaining.get("missiles", 0.0) or 0.0) + prod_mult * float(max(0.0, missiles_pd))
    # artillery proxy
    remaining["artillery"] = float(remaining.get("artillery", 0.0) or 0.0) + prod_mult * float(max(0.0, 0.35*tanks_pd))

    # cap to initial (can't exceed starting stock by huge factor in this toy model)
    for k in ["aircraft","tanks","ships","artillery"]:
        cap = float(init_rem.get(k, remaining.get(k,0.0)) or remaining.get(k,0.0))
        remaining[k] = float(np.clip(remaining[k], 0.0, cap*1.25 + 1.0))

    # consume endurance days
    burn = float(np.clip(Id/2.5, 0.05, 2.5))
    jd["ammo"] = float(max(0.0, jd["ammo"] - 0.85*burn))
    jd["fuel"] = float(max(0.0, jd["fuel"] - 0.45*burn))
    jd["parts"] = float(max(0.0, jd["parts"] - 0.30*burn))

    # update readiness after logistics & production
    def ratio(k):
        a = float(init_rem.get(k, 1.0) or 1.0)
        b = float(remaining.get(k, 0.0) or 0.0)
        return float(np.clip(b / max(1e-9, a), 0.0, 1.0))
    air_r = ratio("aircraft")
    land_r = ratio("tanks")
    sea_r = ratio("ships") if float(init_rem.get("ships",0) or 0) > 0 else 0.65
    supply_r2 = float(np.clip(min(jd["ammo"]/max(1e-6,jd["ammo0"]), jd["fuel"]/max(1e-6,jd["fuel0"]), jd["parts"]/max(1e-6,jd["parts0"])), 0.0, 1.0))
    jd["S"] = float(np.clip(0.45*air_r + 0.35*land_r + 0.20*sea_r, 0.0, 1.0) * (0.55 + 0.45*supply_r2))
    return {"prod_mult": prod_mult, "drones_pd": drones_pd, "rtd": rtd}


# -----------------------------
# Utilities
# -----------------------------

def _norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("&", "and").replace(".", "").replace(",", "")
    s = " ".join(s.split())
    return s

def _clip(x, lo, hi):
    return float(max(lo, min(hi, x)))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _log1p(x):
    return math.log1p(max(0.0, float(x)))

def _read_json(path: str) -> Optional[List[dict]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _read_optional_excel(path: str, sheet_name: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return None




def _num_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    """Safe numeric column accessor.

    - If column is missing, returns a Series filled with NaN (then filled with default).
    - Always returns a pandas Series (never a scalar), so .fillna/.clip are safe.
    """
    if col in df.columns:
        s = df[col]
    else:
        s = pd.Series(np.nan, index=df.index)
    return pd.to_numeric(s, errors="coerce").fillna(float(default))

def _resolve_path(data_dir: str, filename: str) -> str:
    """Resolve data file paths robustly.

    Supports projects where the zip extracts as data/data/<files>.
    Resolution order:
      1) <data_dir>/<filename>
      2) <data_dir>/data/<filename>
      3) <cwd>/<filename>
      4) <cwd>/data/<filename>

    Returns the first existing path; otherwise returns <data_dir>/<filename>.
    """
    data_dir = str(data_dir or "").strip() or "data"
    cands = [
        os.path.join(data_dir, filename),
        os.path.join(data_dir, "data", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "data", filename),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return os.path.join(data_dir, filename)

# -----------------------------
# Data bundle
# -----------------------------

@dataclass
class DataBundle:
    countries: pd.DataFrame
    name_to_iso3: Dict[str, str]
    data_dir: str = "data"
    # Latent embedding columns (computed from all numeric columns + missingness flags).
    latent_dim: int = 16
    z_cols: List[str] = field(default_factory=list)


def load_data(data_dir: str) -> DataBundle:
    """
    Required file:
      - data/military_capability_merged_scored.xlsx (sheet: DATA_Master, Name_Mapping)
    Optional:
      - data/military_equipment_by_country_supplement1.xlsx (sheet: Equipment_numeric)
      - data/largest-air-forces-in-the-world-2026.json
      - data/largest-navies-in-the-world-2026.json
      - data/military-size-by-country-2026.json
      - data/artillery-by-country-2026.json
    """
    master_path = _resolve_path(data_dir, "military_capability_merged_scored.xlsx")
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Missing required file: {master_path}")

    master = pd.read_excel(master_path, sheet_name="DATA_Master")
    name_map = pd.read_excel(master_path, sheet_name="Name_Mapping")

    # Base mapping name -> ISO3
    name_to_iso3: Dict[str, str] = {}

    for _, r in name_map.iterrows():
        nm = _norm_name(r.get("Original Country Name"))
        iso = str(r.get("ISO3") or "").strip()
        if nm and iso:
            name_to_iso3[nm] = iso

    for _, r in master.iterrows():
        nm = _norm_name(r.get("Country_Display"))
        iso = str(r.get("ISO3") or "").strip()
        if nm and iso:
            name_to_iso3[nm] = iso

    df = master.copy()
    if "ISO3" not in df.columns:
        raise ValueError("DATA_Master must contain ISO3 column.")

    df["ISO3"] = df["ISO3"].astype(str).str.strip()
    df["Country_Display"] = df.get("Country_Display", df["ISO3"]).astype(str)
    # Optional: terrain + transport proxy scores (drives theatre friction/defender edge).
    tt_path = _resolve_path(data_dir, "country_land_terrain_transport.xlsx")
    tt = _read_optional_excel(tt_path, sheet_name="Country_Data")
    if tt is not None and len(tt) > 0:
        tt = tt.copy()
        if "ISO3" not in tt.columns and "iso_a3" in tt.columns:
            tt["ISO3"] = tt["iso_a3"].astype(str).str.strip()

        if "ISO3" in tt.columns:
            keep = [c for c in ["ISO3", "terrain_score_0_100", "transport_score_0_100", "area_km2", "perimeter_km"] if c in tt.columns]
            if keep:
                tt = tt[keep].groupby("ISO3", as_index=False).first()
                df = df.merge(tt, on="ISO3", how="left")

    # Fill defaults so theatre modifiers always have a value
    df["terrain_score_0_100"] = _num_col(df, "terrain_score_0_100", 50.0).clip(0, 100)
    df["transport_score_0_100"] = _num_col(df, "transport_score_0_100", 50.0).clip(0, 100)

    # Ensure area/perimeter exist for battleground geometry heuristics (safe defaults)
    if "area_km2" not in df.columns:
        df["area_km2"] = 0.0
    if "perimeter_km" not in df.columns:
        df["perimeter_km"] = 0.0
    df["area_km2"] = pd.to_numeric(df.get("area_km2"), errors="coerce").fillna(0.0).clip(lower=0.0)
    df["perimeter_km"] = pd.to_numeric(df.get("perimeter_km"), errors="coerce").fillna(0.0).clip(lower=0.0)




    # Ensure inventory columns exist
    inv_cols = [
        "aircraft_total", "combat_aircraft", "attack_helos",
        "tanks_mbt",
        "naval_assets_total", "carriers", "helo_carriers",
        "submarines_nuclear", "submarines_non_nuclear",
        "artillery_total", "sp_artillery", "towed_artillery", "mlrs",
        "active_mil", "reserve_mil", "paramilitary",
    ]
    for c in inv_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Helper: merge a supplemental dataframe (already ISO3 keyed) and fill base columns.
    # Policy: if supplement provides a value, take max(base, supplement) to avoid losing a
    # conservative floor from another source.
    def _merge_fill_max(base: pd.DataFrame, sup: Optional[pd.DataFrame], cols: List[str], tag: str) -> pd.DataFrame:
        if sup is None or len(sup) == 0:
            return base
        sup = sup.copy()
        # Rename to avoid suffix surprises
        rename = {c: f"{c}__{tag}" for c in cols if c in sup.columns}
        if not rename:
            return base
        sup = sup.rename(columns=rename)
        keep = ["ISO3"] + list(rename.values())
        sup = sup[[c for c in keep if c in sup.columns]].groupby("ISO3", as_index=False).first()
        out = base.merge(sup, on="ISO3", how="left")
        has_sup = pd.Series(False, index=out.index)
        for c in cols:
            sc = f"{c}__{tag}"
            if sc not in out.columns:
                continue
            b = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0)
            s = pd.to_numeric(out.get(sc), errors="coerce")
            has_sup = has_sup | (s.notna() & (s.fillna(0.0) > 0))
            out[c] = np.where(s.notna(), np.maximum(b, s.fillna(0.0)), b)
            out.drop(columns=[sc], inplace=True)

        # Coverage flag (useful for UI)
        out[f"src_{tag}"] = has_sup.astype(int)
        return out




    # ---- Use any already-available inventory columns from the master workbook
    # (keeps the model useful even if the optional JSON supplements are missing)
    if "Total_Naval_Assets_2024" in df.columns:
        df["naval_assets_total"] = pd.to_numeric(df["Total_Naval_Assets_2024"], errors="coerce").fillna(df["naval_assets_total"])
    if "Aircraft_Carriers_2025" in df.columns:
        df["carriers"] = pd.to_numeric(df["Aircraft_Carriers_2025"], errors="coerce").fillna(df["carriers"])
    if "Helicopter_Carriers_2025" in df.columns:
        df["helo_carriers"] = pd.to_numeric(df["Helicopter_Carriers_2025"], errors="coerce").fillna(df["helo_carriers"])

    # WDMMW unit counts can act as a conservative naval floor
    if "Units_Tracked" in df.columns:
        ut = pd.to_numeric(df["Units_Tracked"], errors="coerce").fillna(0.0)
        df["naval_assets_total"] = np.where(df["naval_assets_total"] > 0, df["naval_assets_total"], ut)

    # ---- Fill terrain/transport proxies for territories missing in the master workbook (optional)
    terr_path = _resolve_path(data_dir, "country_land_terrain_transport.xlsx")
    terr = _read_optional_excel(terr_path, "Country_Data")
    if terr is not None and len(terr) > 0 and "iso_a3" in terr.columns:
        terr = terr.copy()
        terr["ISO3"] = terr["iso_a3"].astype(str).str.strip()
        fill_cols = ["terrain_score_0_100", "transport_score_0_100", "area_km2", "pop_density_per_km2", "gdp_per_cap_usd"]
        keep = ["ISO3"] + [c for c in fill_cols if c in terr.columns]
        terr2 = terr[keep].groupby("ISO3", as_index=False).first()
        df = df.merge(terr2, on="ISO3", how="left", suffixes=("", "_tt"))
        for c in fill_cols:
            tt = f"{c}_tt"
            if tt in df.columns:
                df[c] = pd.to_numeric(df.get(c), errors="coerce")
                df[tt] = pd.to_numeric(df[tt], errors="coerce")
                df[c] = df[c].fillna(df[tt])
                df.drop(columns=[tt], inplace=True)

    # ---- Merge equipment supplement (optional)
    equip_path = _resolve_path(data_dir, "military_equipment_by_country_supplement1.xlsx")
    equip = _read_optional_excel(equip_path, "Equipment_numeric")
    if equip is not None and len(equip) > 0 and "Country" in equip.columns:
        equip = equip.copy()
        equip["__name"] = equip["Country"].apply(_norm_name)
        equip["ISO3"] = equip["__name"].map(name_to_iso3)

        equip_cols = {
            "Main_battle_tanks": "tanks_mbt",
            "Combat_aircraft": "combat_aircraft",
            "Attack_helicopters": "attack_helos",
            "Aircraft_carriers": "carriers",
            "Nuclear_submarines": "submarines_nuclear",
            "Non_nuclear_submarines": "submarines_non_nuclear",
        }

        keep = ["ISO3"] + [k for k in equip_cols.keys() if k in equip.columns]
        equip2 = equip[keep].copy()
        equip2 = equip2.dropna(subset=["ISO3"])
        equip2 = equip2.groupby("ISO3", as_index=False).first()

        for src, dst in equip_cols.items():
            if src in equip2.columns:
                equip2[dst] = pd.to_numeric(equip2[src], errors="coerce")

        # Fill base inventories using the supplement (max policy)
        cols = [v for v in equip_cols.values() if v in equip2.columns]
        df = _merge_fill_max(df, equip2[["ISO3"] + cols], cols=cols, tag="equip")

    # ---- Merge air forces json (optional)
    air = _read_json(_resolve_path(data_dir, "largest-air-forces-in-the-world-2026.json"))
    if air:
        air_df = pd.DataFrame(air)
        if "country" in air_df.columns:
            air_df["__name"] = air_df["country"].apply(_norm_name)
            air_df["ISO3"] = air_df["__name"].map(name_to_iso3)
            air_df = air_df.dropna(subset=["ISO3"])
            air_df["aircraft_total"] = pd.to_numeric(air_df.get("TotalMilitaryAircraft_2025"), errors="coerce")
            air_df = air_df.groupby("ISO3", as_index=False).first()
            df = _merge_fill_max(df, air_df[["ISO3", "aircraft_total"]], cols=["aircraft_total"], tag="air")

    # ---- Merge navies json (optional)
    nav = _read_json(_resolve_path(data_dir, "largest-navies-in-the-world-2026.json"))
    if nav:
        nav_df = pd.DataFrame(nav)
        if "country" in nav_df.columns:
            nav_df["__name"] = nav_df["country"].apply(_norm_name)
            nav_df["ISO3"] = nav_df["__name"].map(name_to_iso3)
            nav_df = nav_df.dropna(subset=["ISO3"])
            nav_df["naval_assets_total"] = pd.to_numeric(nav_df.get("TotalNavalAssets_2024"), errors="coerce")
            nav_df["carriers"] = pd.to_numeric(nav_df.get("AircraftCarriers_2025"), errors="coerce").fillna(0.0)
            nav_df["helo_carriers"] = pd.to_numeric(nav_df.get("HelicopterCarriers_2025"), errors="coerce").fillna(0.0)
            nav_df = nav_df.groupby("ISO3", as_index=False).first()
            df = _merge_fill_max(
                df,
                nav_df[["ISO3", "naval_assets_total", "carriers", "helo_carriers"]],
                cols=["naval_assets_total", "carriers", "helo_carriers"],
                tag="nav",
            )

    # ---- Merge artillery json (optional)
    art = _read_json(_resolve_path(data_dir, "artillery-by-country-2026.json"))
    if art:
        art_df = pd.DataFrame(art)
        if "country" in art_df.columns:
            art_df["__name"] = art_df["country"].apply(_norm_name)
            art_df["ISO3"] = art_df["__name"].map(name_to_iso3)
            art_df = art_df.dropna(subset=["ISO3"])
            art_df["artillery_total"] = pd.to_numeric(art_df.get("TotalStocks_2024"), errors="coerce")
            art_df["sp_artillery"] = pd.to_numeric(art_df.get("SelfPropelledArtilleryStrength_2024"), errors="coerce")
            art_df["towed_artillery"] = pd.to_numeric(art_df.get("TowedArtilleryStrength_2024"), errors="coerce")
            art_df["mlrs"] = pd.to_numeric(art_df.get("MultipleLaunchRocketProjector_2024"), errors="coerce")
            art_df = art_df.groupby("ISO3", as_index=False).first()
            df = _merge_fill_max(
                df,
                art_df[["ISO3", "artillery_total", "sp_artillery", "towed_artillery", "mlrs"]],
                cols=["artillery_total", "sp_artillery", "towed_artillery", "mlrs"],
                tag="art",
            )

    # ---- Merge military size json (optional)
    size = _read_json(_resolve_path(data_dir, "military-size-by-country-2026.json"))
    if size:
        sz = pd.DataFrame(size)
        if "country" in sz.columns:
            sz["__name"] = sz["country"].apply(_norm_name)
            sz["ISO3"] = sz["__name"].map(name_to_iso3)
            sz = sz.dropna(subset=["ISO3"])
            sz["active_mil"] = pd.to_numeric(sz.get("MilitarySizeActiveMilitary_2025"), errors="coerce")
            sz["reserve_mil"] = pd.to_numeric(sz.get("MilitarySizeReserveMilitary_2025"), errors="coerce")
            sz["paramilitary"] = pd.to_numeric(sz.get("MilitarySizeParamilitary_2025"), errors="coerce")
            sz = sz.groupby("ISO3", as_index=False).first()
            df = _merge_fill_max(
                df,
                sz[["ISO3", "active_mil", "reserve_mil", "paramilitary"]],
                cols=["active_mil", "reserve_mil", "paramilitary"],
                tag="size",
            )

    
    # ---- Merge missile inventory / defense estimates (optional)
    # File: global_missile_numbers_only_top50.xlsx (sheet: Numbers_Only)
    # Columns include lower-bound offensive missile counts, lower-bound interceptor counts,
    # plus quality indices (Density/Saturation/Intercept/Breakthrough) in 0..100.
    miss_path = _resolve_path(data_dir, "global_missile_numbers_only_top50.xlsx")
    if not os.path.exists(miss_path):
        _alt = os.path.join(os.getcwd(), "global_missile_numbers_only_top50.xlsx")
        if os.path.exists(_alt):
            miss_path = _alt
    miss = _read_optional_excel(miss_path, sheet_name="Numbers_Only")
    if miss is not None and len(miss) > 0 and "Country" in miss.columns:
        miss = miss.copy()

        iso_set = set(df["ISO3"].astype(str).tolist())
        alias = {
            "united states": "USA", "u s": "USA", "us": "USA", "usa": "USA", "u.s.": "USA", "u.s": "USA",
            "russia": "RUS", "russian federation": "RUS",
            "china": "CHN", "people s republic of china": "CHN", "peoples republic of china": "CHN", "prc": "CHN",
            "south korea": "KOR", "korea south": "KOR", "republic of korea": "KOR",
            "north korea": "PRK", "korea north": "PRK", "dprk": "PRK",
            "iran": "IRN", "iran islamic republic of": "IRN",
            "turkey": "TUR",
            "united kingdom": "GBR", "uk": "GBR", "u k": "GBR", "great britain": "GBR",
        }

        def _missile_iso3(x) -> Optional[str]:
            s = str(x).strip()
            if len(s) == 3 and s.isalpha():
                code = s.upper()
                if code in iso_set:
                    return code
            nm = _norm_name(s)
            iso = name_to_iso3.get(nm)
            if iso:
                return iso
            return alias.get(nm)

        miss["ISO3"] = miss["Country"].apply(_missile_iso3)
        miss = miss.dropna(subset=["ISO3"]).copy()

        rename = {
            "Interceptors_Defense_Est_LB": "interceptors_defense_lb",
            "Offensive_Missiles_Est_LB": "missile_offensive_lb",
            "Interceptors_Known_Anchor": "interceptors_anchor",
            "Strategic_Missiles_Known_Anchor": "missile_strategic_anchor",
            "Shield_Rank": "missile_shield_rank",
            "Density_D": "missile_density_D",
            "Saturation_S": "missile_saturation_S",
            "Intercept_I": "missile_intercept_I",
            "Breakthrough_B": "missile_breakthrough_B",
            "Tier_Number": "missile_tier",
            "Rank": "missile_rank",
        }
        miss = miss.rename(columns={k: v for k, v in rename.items() if k in miss.columns})

        keep = ["ISO3"] + [v for v in rename.values() if v in miss.columns]
        miss2 = miss[keep].groupby("ISO3", as_index=False).first()

        df = df.merge(miss2, on="ISO3", how="left")

        # Coverage flag
        df["src_missile"] = _num_col(df, "missile_offensive_lb", 0.0).gt(0.0).astype(int)

# If aircraft_total missing but combat_aircraft exists, infer conservative total
    df["combat_aircraft"] = pd.to_numeric(df.get("combat_aircraft"), errors="coerce").fillna(0.0)
    df["aircraft_total"] = pd.to_numeric(df.get("aircraft_total"), errors="coerce").fillna(0.0)
    df.loc[df["aircraft_total"] <= 0, "aircraft_total"] = np.maximum(df["combat_aircraft"] * 1.35, 0.0)

    # Personnel fallback
    if "Armed forces personnel (total)" in df.columns:
        af = pd.to_numeric(df["Armed forces personnel (total)"], errors="coerce").fillna(0.0)
        df["active_mil"] = np.where(pd.to_numeric(df["active_mil"], errors="coerce").fillna(0.0) > 0, df["active_mil"], af)

    df["reserve_mil"] = pd.to_numeric(df["reserve_mil"], errors="coerce").fillna(0.0)
    df["paramilitary"] = pd.to_numeric(df["paramilitary"], errors="coerce").fillna(0.0)

    # Normalize key score columns (0-100 expected); fallback to 50
    score_cols = [
        "land_score", "air_score", "naval_score", "logistics_score", "economy_score",
        "population_score", "networks_score", "resilience_score", "future_resources_score", "milcap_score",
    ]
    for c in score_cols:
        if c not in df.columns:
            df[c] = 50.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(50.0).clip(0, 100)

    # Nuclear capability
    nuc = df.get("Estimated warheads (total inventory)")
    if nuc is None:
        df["warheads"] = 0.0
    else:
        df["warheads"] = pd.to_numeric(nuc, errors="coerce").fillna(0.0)
    df["is_nuclear"] = df["warheads"] > 0

    # Economic base / population
    df["gdp_usd"] = _num_col(df, "GDP (current US$)", 0.0)
    df["mil_spend_usd"] = _num_col(df, "Military expenditure (current US$)", 0.0)
    df["population"] = _num_col(df, "Population (people)", 0.0)

    # Conservative proxy inventories when explicit inventories are missing (prevents Day-1 collapse)
    spend_s = np.sqrt(np.maximum(df["mil_spend_usd"], 0.0)) / 1e5
    pop = np.maximum(df["population"], 0.0)

    air_proxy = 20.0 + 60.0 * spend_s + 0.0000008 * pop
    df["aircraft_total"] = np.where(df["aircraft_total"] > 0, df["aircraft_total"], air_proxy)
    # Allow very large air forces (dataset includes total military aircraft counts in the 5-figure range).
    df["aircraft_total"] = df["aircraft_total"].clip(0, 20000)

    df["combat_aircraft"] = np.where(df["combat_aircraft"] > 0, df["combat_aircraft"], df["aircraft_total"] * 0.65)
    df["combat_aircraft"] = np.minimum(df["combat_aircraft"], df["aircraft_total"]).clip(0, 15000)

    tank_proxy = 30.0 + 45.0 * spend_s + 0.0000015 * pop
    df["tanks_mbt"] = pd.to_numeric(df.get("tanks_mbt"), errors="coerce").fillna(0.0)
    df["tanks_mbt"] = np.where(df["tanks_mbt"] > 0, df["tanks_mbt"], tank_proxy)
    df["tanks_mbt"] = df["tanks_mbt"].clip(0, 25000)

    naval_proxy = 10.0 + 7.0 * spend_s + 0.0000002 * pop
    df["naval_assets_total"] = pd.to_numeric(df.get("naval_assets_total"), errors="coerce").fillna(0.0)
    df["naval_assets_total"] = np.where(df["naval_assets_total"] > 0, df["naval_assets_total"], naval_proxy)
    df["naval_assets_total"] = df["naval_assets_total"].clip(0, 5000)

    art_proxy = 60.0 + 90.0 * spend_s + 0.0000020 * pop
    df["artillery_total"] = pd.to_numeric(df.get("artillery_total"), errors="coerce").fillna(0.0)
    df["artillery_total"] = np.where(df["artillery_total"] > 0, df["artillery_total"], art_proxy)
    df["artillery_total"] = df["artillery_total"].clip(0, 120000)

    # Terrain + transport (if present)
    df["terrain_score_0_100"] = _num_col(df, "terrain_score_0_100", 50.0).clip(0, 100)
    df["transport_score_0_100"] = _num_col(df, "transport_score_0_100", 50.0).clip(0, 100)
    df["continent"] = (df["Continent"].astype(str) if "Continent" in df.columns else "")

    # A2/AD proxy: air+naval+networks+terrain
    df["a2ad"] = (
        0.35 * df["air_score"] +
        0.25 * df["naval_score"] +
        0.20 * df["networks_score"] +
        0.10 * df["terrain_score_0_100"] +
        0.10 * df["milcap_score"]
    ).clip(0, 100)

    # Projection: ability to fight far
    df["projection"] = (
        0.45 * df["naval_score"] +
        0.35 * df["air_score"] +
        0.20 * df["logistics_score"]
    ).clip(0, 100)

    # Missile + interceptor stocks: derived proxies
    spend_b = np.sqrt(np.maximum(df["mil_spend_usd"], 0.0)) / 1e6
    df["missile_stock"] = np.maximum(50.0, (120.0 * spend_b) * (0.6 + df["future_resources_score"] / 200.0) * (0.7 + df["networks_score"] / 200.0))
    df["interceptor_stock"] = np.maximum(80.0, (160.0 * spend_b) * (0.8 + df["air_score"] / 200.0) * (0.7 + df["networks_score"] / 250.0))

    # Override missile/interceptor proxies with lower-bound estimates (if present)
    miss_lb = _num_col(df, "missile_offensive_lb", 0.0)
    ints_lb = _num_col(df, "interceptors_defense_lb", 0.0)
    df["missile_stock"] = np.where(miss_lb > 0, np.maximum(df["missile_stock"], miss_lb), df["missile_stock"])
    df["interceptor_stock"] = np.where(ints_lb > 0, np.maximum(df["interceptor_stock"], ints_lb), df["interceptor_stock"])

    # Optional missile quality indices (0..100). Default to mid-values if missing.
    for _c, _dflt in [
        ("missile_density_D", 50.0),
        ("missile_saturation_S", 50.0),
        ("missile_intercept_I", 50.0),
        ("missile_breakthrough_B", 50.0),
    ]:
        df[_c] = pd.to_numeric(df.get(_c), errors="coerce").fillna(_dflt).clip(0, 100)

    df["missile_shield_rank"] = _num_col(df, "missile_shield_rank", 25.0).clip(1, 60)
    df["missile_tier"] = _num_col(df, "missile_tier", 3.0).clip(1, 10)
    df["missile_rank"] = _num_col(df, "missile_rank", 25.0).clip(1, 60)
    df["interceptors_anchor"] = _num_col(df, "interceptors_anchor", 0.0)
    df["missile_strategic_anchor"] = _num_col(df, "missile_strategic_anchor", 0.0)


    # Doctrine archetypes: coarse, derived from signature
    def doctrine_row(r):
        land = r["land_score"]; air = r["air_score"]; sea = r["naval_score"]; net = r["networks_score"]
        res = r["resilience_score"]; a2 = r["a2ad"]; proj = r["projection"]
        if net > 70 and (air > 65 or sea > 65):
            return "Joint Networked"
        if a2 > 70 and res > 60:
            return "Fortress A2AD"
        if land > 70 and res > 60:
            return "Mass Mobilization"
        if proj > 70 and sea > 65:
            return "Bluewater Expeditionary"
        if air > 72 and net > 55:
            return "Air-Centric"
        return "Balanced"

    df["doctrine"] = df.apply(doctrine_row, axis=1)

    df = df.replace([np.inf, -np.inf], 0.0)

    # Merge Job-Done / Logistics add-ons (145-country matrices) if present
    try:
        df = _merge_job_done_addons(df, name_to_iso3=name_to_iso3, data_dir=data_dir)
    except Exception:
        pass

    latent_dim = 16
    df, z_cols = _ensure_latent_embeddings(df, latent_dim=latent_dim)

    return DataBundle(countries=df, name_to_iso3=name_to_iso3, data_dir=data_dir, latent_dim=latent_dim, z_cols=z_cols)



# -----------------------------
# Dense latent model utilities
# -----------------------------

def _ensure_latent_embeddings(df: pd.DataFrame, latent_dim: int = 16) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute a compact latent embedding z0..z{latent_dim-1} for each country using:
      - ALL numeric columns in the dataframe (after merges)
      - Missingness flags for those numeric columns

    This provides a dense, multi-dimensional state representation without requiring extra libraries.
    """
    latent_dim = int(max(2, min(int(latent_dim), 64)))
    z_cols = [f"z{i}" for i in range(latent_dim)]
    if all(c in df.columns for c in z_cols):
        return df, z_cols

    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        # Fallback: create zeros
        for c in z_cols:
            df[c] = 0.0
        return df, z_cols

    num = num.replace([np.inf, -np.inf], np.nan)
    miss = num.isna().astype(float)

    med = num.median(numeric_only=True)
    num = num.fillna(med).fillna(0.0)

    X = num.values.astype(float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    Xs = (X - mu) / sd

    Xm = miss.values.astype(float)
    Xfeat = np.concatenate([Xs, Xm], axis=1)

    # PCA via SVD (deterministic)
    # Xfeat = U S V^T -> projection is Xfeat @ V[:k].T
    try:
        U, S, Vt = np.linalg.svd(Xfeat, full_matrices=False)
        k = min(latent_dim, Vt.shape[0])
        Z = Xfeat @ Vt[:k].T
    except Exception:
        # very rare fallback: random but deterministic projection
        rng = np.random.default_rng(1337)
        W = rng.normal(0.0, 1.0, size=(Xfeat.shape[1], latent_dim))
        Z = Xfeat @ W

    # Standardize Z
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-9)

    # If Z has fewer dims than requested, pad
    if Z.shape[1] < latent_dim:
        pad = np.zeros((Z.shape[0], latent_dim - Z.shape[1]), dtype=float)
        Z = np.concatenate([Z, pad], axis=1)

    for i, c in enumerate(z_cols):
        df[c] = Z[:, i].astype(float)

    return df, z_cols


def dense_default_cfg() -> dict:
    """Default knobs for the Dense model."""
    return dict(
        embed_strength=1.0,     # how strongly the latent vectors tilt advantage
        embed_seed=1337,        # stable projection for embed advantage
        fronts=1,               # multi-front penalty (>=1)
        # Global rate scaling (applied to both sides, then biased per-side)
        air_mult=1.00,
        sea_mult=1.00,
        land_mult=1.00,
        missile_mult=1.00,
        contact_mult=1.00,
        civ_mult=1.00,
        commit_mult=1.00,
    )


def _dense_side_vector(df: pd.DataFrame, iso3s: List[str], z_cols: List[str]) -> np.ndarray:
    if not z_cols:
        return np.zeros(16, dtype=float)
    sub = df[df["ISO3"].isin([x for x in iso3s if x])]
    if len(sub) == 0:
        return np.zeros(len(z_cols), dtype=float)
    return sub[z_cols].mean(axis=0).to_numpy(dtype=float)


def _dense_context(
    bundle: DataBundle,
    sc: Scenario,
    bg_row: dict,
    theatre: dict,
    blue_def: dict,
    red_def: dict,
    dist_blue: float,
    dist_red: float,
    pen_blue: float,
    pen_red: float,
) -> dict:
    """
    Return per-side multipliers for params + commit fractions.

    Philosophy:
      - Use interpretable scores (air/naval/land/networks/logistics/projection/a2ad)
      - Use latent vectors (z0..zK) to inject high-dimensional context from ALL columns
      - Include a simple multi-front penalty via cfg['fronts']
    """
    df = bundle.countries
    z_cols = bundle.z_cols or [c for c in df.columns if str(c).startswith("z")]

    cfg = dense_default_cfg()
    if getattr(sc, "dense_cfg", None):
        # user overrides
        for k, v in dict(sc.dense_cfg).items():
            cfg[k] = v

    fronts = int(max(1, min(int(cfg.get("fronts", 1)), 8)))
    front_friction = 1.0 + 0.08 * (fronts - 1)  # more fronts => slower tempo / higher logistics drag

    # Latent advantage (deterministic)
    zB = _dense_side_vector(df, sc.blue, z_cols)
    zR = _dense_side_vector(df, sc.red, z_cols)
    dz = zB - zR

    rng = np.random.default_rng(int(cfg.get("embed_seed", 1337)))
    w = rng.normal(0.0, 1.0, size=len(dz) if len(dz) > 0 else 16)
    if len(dz) == 0:
        dz = np.zeros_like(w)

    latent_adv = float(np.tanh(float(cfg.get("embed_strength", 1.0)) * (dz @ w) / (np.sqrt(len(w)) + 1e-9)))

    # Interpretable score advantages (scaled roughly to [-1, +1])
    air_adv = float((blue_def["air_score"] - red_def["air_score"]) / 40.0)
    sea_adv = float((blue_def["naval_score"] - red_def["naval_score"]) / 40.0)
    land_adv = float((blue_def["land_score"] - red_def["land_score"]) / 40.0)
    net_adv = float((blue_def["networks_score"] - red_def["networks_score"]) / 40.0)
    proj_adv = float((blue_def["projection"] - red_def["projection"]) / 50.0)
    log_adv = float((blue_def["logistics_score"] - red_def["logistics_score"]) / 45.0)

    # Parity (close fights tend to be bloodier + longer)
    parity = float(1.0 - min(1.0, abs(latent_adv) * 0.85))
    parity = float(_clip(parity, 0.05, 1.0))

    def _bias(base: float, adv: float) -> Tuple[float, float]:
        # Convert advantage -> +/- bias around base (bounded)
        adv = float(np.tanh(adv))
        # latent contributes as a smaller, general tilt
        adv2 = float(_clip(0.55 * adv + 0.45 * latent_adv, -1.0, 1.0))
        b = float(_clip(base * (1.0 + 0.18 * adv2), 0.55 * base, 1.85 * base))
        r = float(_clip(base * (1.0 - 0.18 * adv2), 0.55 * base, 1.85 * base))
        return b, r

    # Rate multipliers per side
    air_base = float(cfg.get("air_mult", 1.0)) / front_friction
    sea_base = float(cfg.get("sea_mult", 1.0)) / front_friction
    land_base = float(cfg.get("land_mult", 1.0)) / front_friction
    miss_base = float(cfg.get("missile_mult", 1.0)) / front_friction

    b_air, r_air = _bias(air_base, 0.65 * air_adv + 0.35 * net_adv)
    b_sea, r_sea = _bias(sea_base, 0.75 * sea_adv + 0.25 * proj_adv)
    b_land, r_land = _bias(land_base, 0.70 * land_adv + 0.30 * log_adv)
    b_miss, r_miss = _bias(miss_base, 0.45 * net_adv + 0.35 * proj_adv + 0.20 * air_adv)

    # Contact: higher with parity, more fronts, rough terrain, and limited penetration
    contact_base = float(cfg.get("contact_mult", 1.0)) * (1.0 + 0.22 * parity) * (1.0 + 0.06 * (fronts - 1))
    # attacker-specific: if you can penetrate, you can generate more contact
    b_contact = float(_clip(contact_base * (0.92 + 0.30 * pen_blue), 0.45, 2.20))
    r_contact = float(_clip(contact_base * (0.92 + 0.30 * pen_red), 0.45, 2.20))

    # Civilian risk rises with terrain friction + contact + penetration
    terrain = float(theatre.get("terrain", 50.0))
    transport = float(theatre.get("transport", 50.0))
    civ_env = (0.85 + 0.30 * (terrain / 100.0) + 0.12 * (1.0 - transport / 100.0))
    civ_base = float(cfg.get("civ_mult", 1.0)) * civ_env
    b_civ = float(_clip(civ_base * (0.88 + 0.34 * b_contact) * (0.92 + 0.25 * pen_blue), 0.35, 2.75))
    r_civ = float(_clip(civ_base * (0.88 + 0.34 * r_contact) * (0.92 + 0.25 * pen_red), 0.35, 2.75))

    # Commit fraction: projection/logistics vs distance vs multi-front tax
    commit_base = float(cfg.get("commit_mult", 1.0)) * (1.0 - 0.07 * (fronts - 1))

    b_commit = commit_base * (0.94 + 0.12 * (blue_def["projection"] / 100.0) + 0.10 * (blue_def["logistics_score"] / 100.0)) / (0.90 + 0.18 * max(0.0, dist_blue - 1.0))
    r_commit = commit_base * (0.94 + 0.12 * (red_def["projection"] / 100.0) + 0.10 * (red_def["logistics_score"] / 100.0)) / (0.90 + 0.18 * max(0.0, dist_red - 1.0))

    b_commit = float(_clip(b_commit, 0.65, 1.25))
    r_commit = float(_clip(r_commit, 0.65, 1.25))

    return dict(
        commit_blue_mult=b_commit,
        commit_red_mult=r_commit,
        blue_params_mult=dict(
            missile_rate=b_miss,
            air_rate=b_air,
            sea_rate=b_sea,
            land_rate=b_land,
            contact=b_contact,
            civ_factor=b_civ,
        ),
        red_params_mult=dict(
            missile_rate=r_miss,
            air_rate=r_air,
            sea_rate=r_sea,
            land_rate=r_land,
            contact=r_contact,
            civ_factor=r_civ,
        ),
        latent_adv=latent_adv,
        parity=parity,
        fronts=fronts,
    )


def fit_dense_cfg(
    bundle: DataBundle,
    sc: Scenario,
    targets: dict,
    iters: int = 150,
    seed: int = 42,
    progress_callback=None,
) -> dict:
    """
    Very small, generic calibration helper.

    You provide target totals like:
      targets = {
        "blue": {"aircraft_lost": 300, "ships_lost": 25, "kia": 5000},
        "red":  {"aircraft_lost": 200, "ships_lost": 120, "kia": 15000},
        "civ_total": 20000,     # optional (both sides combined)
        "days": 90              # optional: force calibration horizon
      }

    Returns a dense_cfg dict that best fits those targets under this toy model.

    NOTE: This is *not* a real-world predictor. It's just parameter fitting for a fictional simulator.
    """
    rng = np.random.default_rng(int(seed))

    # Build a short-horizon copy (faster fitting)
    days = int(targets.get("days", sc.max_days))
    sc_fit = Scenario(**{**sc.__dict__, "max_days": days, "core_model": "Dense"})

    KEYMAP = {
        "aircraft_lost": "aircraft",
        "ships_lost": "ships",
        "tanks_lost": "tanks",
        "artillery_lost": "artillery",
        "aircraft": "aircraft",
        "ships": "ships",
        "tanks": "tanks",
        "artillery": "artillery",
        "missiles_spent": "missiles_spent",
        "interceptors_spent": "interceptors_spent",
        "kia": "kia",
        "wia": "wia",
    }

    def _norm_side_dict(d: dict) -> dict:
        out = {}
        for k, v in dict(d or {}).items():
            mk = KEYMAP.get(str(k).strip().lower())
            if mk is None:
                continue
            try:
                out[mk] = float(v)
            except Exception:
                continue
        return out

    # Flatten targets into a vector for scoring
    target_vec = {}
    for side in ("blue", "red"):
        sd = _norm_side_dict(targets.get(side, {}))
        for k, v in sd.items():
            target_vec[f"{side}.{k}"] = float(v)
    if "civ_total" in targets:
        try:
            target_vec["civ_total"] = float(targets["civ_total"])
        except Exception:
            pass

    if not target_vec:
        return dense_default_cfg()

    def _extract(summary: dict) -> dict:
        out = {}
        totals = summary.get("totals", {})
        for side in ("blue", "red"):
            s = totals.get(side, {}) or {}
            for k, v in s.items():
                kk = KEYMAP.get(str(k).strip().lower(), None)
                if kk is None:
                    continue
                try:
                    out[f"{side}.{kk}"] = float(v)
                except Exception:
                    continue
        out["civ_total"] = float(summary.get("civ_total", 0.0))
        return out

    # Weights: heavier for aircraft/ships, moderate for KIA/civ
    def w_for(key: str) -> float:
        if key.endswith(".ships"):
            return 2.0
        if key.endswith(".aircraft"):
            return 2.0
        if key.endswith(".kia"):
            return 1.2
        if key == "civ_total":
            return 1.1
        return 1.0

    def score(pred: dict) -> float:
        err = 0.0
        for k, tv in target_vec.items():
            pv = float(pred.get(k, 0.0))
            denom = max(50.0, abs(tv))
            err += w_for(k) * ((pv - tv) / denom) ** 2
        return float(err)

    best_cfg = dense_default_cfg()
    best_err = float("inf")

    base = dense_default_cfg()
    base.update(dict(getattr(sc, "dense_cfg", None) or {}))

    iters = int(max(10, iters))

    for i in range(iters):
        cfg = dict(base)

        # jitter multipliers (log-normal)
        def j(key, sigma=0.18, lo=0.60, hi=1.70):
            cfg[key] = float(_clip(cfg.get(key, 1.0) * float(np.exp(rng.normal(0.0, sigma))), lo, hi))

        for k in ["air_mult", "sea_mult", "land_mult", "missile_mult", "contact_mult", "civ_mult", "commit_mult"]:
            j(k, sigma=0.16)

        cfg["embed_strength"] = float(_clip(cfg.get("embed_strength", 1.0) + rng.normal(0.0, 0.25), 0.0, 2.5))
        cfg["fronts"] = int(max(1, min(8, int(cfg.get("fronts", 1) + rng.integers(-1, 2)))))

        sc_try = Scenario(**{**sc_fit.__dict__, "dense_cfg": cfg})
        summary, _timeline = simulate_cinematic(bundle, sc_try)
        pred = _extract(summary)

        e = score(pred)
        if e < best_err:
            best_err = e
            best_cfg = cfg

        if progress_callback and (i % 10 == 0 or i == iters - 1):
            progress_callback(i + 1, iters, best_err)

    return best_cfg



# -----------------------------
# Scenario & simulation
# -----------------------------

@dataclass
class Scenario:
    year: int
    intensity: str          # "Skirmish" | "Campaign" | "All-out"
    mode: str               # "Limited War" | "Total War"
    max_days: int
    battleground_iso3: str  # theater anchor (defender homeland proxy)
    sanctions_blue: float   # 0..1
    sanctions_red: float    # 0..1
    nukes_allowed: bool
    blue: List[str]         # list of ISO3
    red: List[str]
    neutral_support_blue: List[str]
    neutral_support_red: List[str]
    # Scenario type is used by the Job-Done + LMC layers to shift weights (Ground/Naval/Air/Cyber).
    scenario_type: str = "General"
    core_model: str = "Cinematic"  # "Cinematic" | "Dense"
    dense_cfg: Optional[dict] = None  # advanced config for Dense model
    integrated_cfg: Optional[dict] = None  # advanced config for Integrated model
    seed: int = 7


def _continent_distance(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 1.2
    if a == b:
        return 0.7
    key = tuple(sorted([a, b]))
    table = {
        ("africa", "europe"): 1.0,
        ("africa", "asia"): 1.3,
        ("africa", "north america"): 1.5,
        ("africa", "south america"): 1.6,
        ("africa", "oceania"): 1.8,
        ("europe", "asia"): 1.2,
        ("europe", "north america"): 1.2,
        ("europe", "south america"): 1.5,
        ("europe", "oceania"): 1.7,
        ("asia", "north america"): 1.6,
        ("asia", "south america"): 1.7,
        ("asia", "oceania"): 1.1,
        ("north america", "south america"): 1.0,
        ("north america", "oceania"): 1.8,
        ("south america", "oceania"): 1.9,
    }
    return table.get(key, 1.6)


def _tech_leap_bonus(year: int, tech_score_0_100: float) -> float:
    t = tech_score_0_100 / 100.0
    bonus = 0.0
    if year >= 2032:
        bonus += 0.10 * t
    if year >= 2036:
        bonus += 0.12 * t
    return bonus


def _intensity_params(intensity) -> dict:
    """Return baseline intensity parameters.

    Accepts:
      - string labels: "Skirmish", "Campaign", "All-Out"
      - numeric scalar: 0.05..1.0 (smooth blend from Skirmish -> All-Out)

    The Streamlit UI uses the numeric form.
    """
    # Numeric slider support
    if isinstance(intensity, (int, float, np.floating)):
        x = float(intensity)
        x = float(_clip(x, 0.05, 1.0))
        t = (x - 0.05) / 0.95  # 0..1

        sk = dict(contact=0.25, missile_rate=0.012, air_rate=0.009, sea_rate=0.006, land_rate=0.007, civ_factor=0.25, shock=0.8)
        ao = dict(contact=0.75, missile_rate=0.035, air_rate=0.020, sea_rate=0.014, land_rate=0.018, civ_factor=0.70, shock=1.40)
        return {k: float(sk[k] + (ao[k] - sk[k]) * t) for k in sk}

    s = (str(intensity) if intensity is not None else "Campaign").strip().lower()
    if s.startswith("sk"):
        return dict(contact=0.25, missile_rate=0.012, air_rate=0.009, sea_rate=0.006, land_rate=0.007, civ_factor=0.25, shock=0.8)
    if s.startswith("all"):
        return dict(contact=0.75, missile_rate=0.035, air_rate=0.020, sea_rate=0.014, land_rate=0.018, civ_factor=0.70, shock=1.40)
    return dict(contact=0.50, missile_rate=0.022, air_rate=0.014, sea_rate=0.010, land_rate=0.012, civ_factor=0.45, shock=1.10)



def _theatre_modifiers(bg_row: dict) -> dict:
    """Return theatre-level friction/advantage modifiers from battleground attributes."""
    terrain = float(bg_row.get("terrain_score_0_100", 50.0) or 50.0)
    transport = float(bg_row.get("transport_score_0_100", 50.0) or 50.0)
    a2ad = float(bg_row.get("a2ad", 50.0) or 50.0)

    # Friction slows operational tempo (mountains/fragmented borders + weak transport)
    friction = 1.0 + 0.35 * (terrain / 100.0) + 0.20 * (1.0 - transport / 100.0)

    # Defender edge is mainly driven by local terrain + A2/AD saturation
    defender_edge = 1.0 + 0.20 * (a2ad / 100.0) + 0.15 * (terrain / 100.0)

    return dict(
        terrain=terrain,
        transport=transport,
        a2ad=a2ad,
        friction=float(_clip(friction, 0.85, 1.85)),
        defender_edge=float(_clip(defender_edge, 0.90, 1.55)),
    )


def _side_aggregate(df: pd.DataFrame, iso3s: List[str], sanctions: float, supporter_iso3s: List[str]) -> dict:
    iso3s = [x for x in iso3s if x]
    supporters = [x for x in supporter_iso3s if x and x not in iso3s]

    core = df[df["ISO3"].isin(iso3s)].copy()
    sup = df[df["ISO3"].isin(supporters)].copy()

    core_w = 1.0
    sup_w = 0.35

    def wsum(col):
        return core_w * core[col].sum() + sup_w * sup[col].sum()

    def wavg(col, default=50.0):
        denom = core_w * len(core) + sup_w * len(sup)
        if denom <= 0:
            return default
        return (core_w * core[col].sum() + sup_w * sup[col].sum()) / denom

    sanc = _clip(sanctions, 0.0, 1.0)
    sanc_mult = (1.0 - 0.45 * sanc)

    agg = dict(
        iso3=iso3s,
        supporters=supporters,
        population=wsum("population"),
        gdp_usd=wsum("gdp_usd") * (1.0 - 0.18 * sanc),
        mil_spend_usd=wsum("mil_spend_usd") * (1.0 - 0.15 * sanc),
        aircraft=wsum("aircraft_total"),
        combat_aircraft=wsum("combat_aircraft"),
        helos=wsum("attack_helos"),
        tanks=wsum("tanks_mbt"),
        naval=wsum("naval_assets_total"),
        carriers=wsum("carriers"),
        artillery=wsum("artillery_total"),
        mlrs=wsum("mlrs"),
        active=wsum("active_mil"),
        reserve=wsum("reserve_mil"),
        paramilitary=wsum("paramilitary"),
        missiles=wsum("missile_stock") * sanc_mult,
        interceptors=wsum("interceptor_stock") * sanc_mult,
        strategic_missiles=wsum("missile_strategic_anchor") * sanc_mult,
        missile_density=wavg("missile_density_D", 50.0),
        missile_saturation=wavg("missile_saturation_S", 50.0),
        missile_intercept=wavg("missile_intercept_I", 50.0),
        missile_breakthrough=wavg("missile_breakthrough_B", 50.0),
        missile_shield_rank=wavg("missile_shield_rank", 25.0),
        missile_tier=wavg("missile_tier", 3.0),
        land_score=wavg("land_score"),
        air_score=wavg("air_score"),
        naval_score=wavg("naval_score"),
        logistics_score=wavg("logistics_score") * sanc_mult,
        economy_score=wavg("economy_score") * (1.0 - 0.25 * sanc),
        networks_score=wavg("networks_score"),
        resilience_score=wavg("resilience_score"),
        future_resources_score=wavg("future_resources_score") * (1.0 - 0.20 * sanc),
        milcap_score=wavg("milcap_score"),
        a2ad=wavg("a2ad"),
        projection=wavg("projection") * sanc_mult,
        nukes=int(wsum("warheads") > 0),
        warheads=wsum("warheads"),
        doctrines=list(core["doctrine"].value_counts().index[:3]),
        sanc=sanc,
    )

    agg["tech_index"] = (0.55 * agg["future_resources_score"] + 0.45 * agg["networks_score"]) / 100.0
    return agg


def _commit_fraction(side: dict, target_a2ad: float, distance: float, mode: str) -> float:
    proj = side["projection"] / 100.0
    logi = side["logistics_score"] / 100.0
    risk = 0.55 + 0.25 * (side["resilience_score"] / 100.0)
    if mode.strip().lower().startswith("total"):
        risk += 0.10
    a2 = target_a2ad / 100.0
    dist_pen = 1.0 / (1.0 + 0.8 * (distance - 0.7))
    a2_pen = 1.0 - 0.45 * a2 * (1.0 - 0.75 * proj)
    frac = proj * logi * risk * dist_pen * a2_pen
    return _clip(frac, 0.08, 0.85)


def _penetration_chance(attacker: dict, defender: dict, year: int, distance: float, mode: str) -> float:
    proj = attacker["projection"] / 100.0
    atk_net = attacker["networks_score"] / 100.0
    def_a2 = defender["a2ad"] / 100.0
    dist_pen = 1.0 / (1.0 + 0.9 * (distance - 0.7))
    tech_bonus = _tech_leap_bonus(year, 100.0 * attacker["tech_index"])
    base = 0.35 + 0.40 * proj + 0.18 * atk_net - 0.35 * def_a2
    if mode.strip().lower().startswith("total"):
        base += 0.05
    p = base * dist_pen + tech_bonus * 0.25
    return _clip(p, 0.02, 0.95)


def _day_losses(attacker: dict, defender: dict, params: dict, year: int, commit_a: float, commit_d: float,
                can_penetrate: float, rng: np.random.Generator) -> dict:
    A_air = attacker["aircraft"] * commit_a
    D_air = defender["aircraft"] * commit_d
    A_land = (attacker["active"] + 0.25 * attacker["reserve"]) * commit_a
    D_land = (defender["active"] + 0.25 * defender["reserve"]) * commit_d
    A_sea = attacker["naval"] * commit_a
    D_sea = defender["naval"] * commit_d

    air_adv = (0.6 * (attacker["air_score"] / 100.0) + 0.4 * (attacker["networks_score"] / 100.0)) / max(0.35, (defender["air_score"] / 100.0))
    sea_adv = (0.6 * (attacker["naval_score"] / 100.0) + 0.4 * (attacker["projection"] / 100.0)) / max(0.35, (defender["naval_score"] / 100.0))
    land_adv = (0.7 * (attacker["land_score"] / 100.0) + 0.3 * (attacker["logistics_score"] / 100.0)) / max(0.35, (defender["land_score"] / 100.0))


    # Missile tempo shaped by saturation + density (from missile add-on dataset if present)
    satA = float(attacker.get("missile_saturation", 50.0) or 50.0) / 100.0
    denA = float(attacker.get("missile_density", 50.0) or 50.0) / 100.0
    brA  = float(attacker.get("missile_breakthrough", 50.0) or 50.0) / 100.0
    intAq = float(attacker.get("missile_intercept", 50.0) or 50.0) / 100.0

    satD = float(defender.get("missile_saturation", 50.0) or 50.0) / 100.0
    denD = float(defender.get("missile_density", 50.0) or 50.0) / 100.0
    brD  = float(defender.get("missile_breakthrough", 50.0) or 50.0) / 100.0
    intDq = float(defender.get("missile_intercept", 50.0) or 50.0) / 100.0

    launch_factor_A = float(np.clip((0.65 + 0.65 * satA) * (0.80 + 0.50 * denA), 0.45, 1.65))
    launch_factor_D = float(np.clip((0.65 + 0.65 * satD) * (0.80 + 0.50 * denD), 0.45, 1.65))

    missiles_launched_A = min(attacker["missiles"], max(0.0, params["missile_rate"] * attacker["missiles"] * (0.75 + 0.5 * rng.random()) * launch_factor_A))
    missiles_launched_D = min(defender["missiles"], max(0.0, params["missile_rate"] * defender["missiles"] * (0.75 + 0.5 * rng.random()) * launch_factor_D))


    techA = _tech_leap_bonus(year, 100.0 * attacker["tech_index"])
    techD = _tech_leap_bonus(year, 100.0 * defender["tech_index"])

    
    # Intercept effectiveness boosted by interceptor quality index and shield rank (lower rank => stronger shield)
    shield_rank_D = float(defender.get("missile_shield_rank", 25.0) or 25.0)
    shield_rank_A = float(attacker.get("missile_shield_rank", 25.0) or 25.0)
    shield_q_D = float(np.clip(1.0 - (shield_rank_D - 1.0) / 59.0, 0.0, 1.0))
    shield_q_A = float(np.clip(1.0 - (shield_rank_A - 1.0) / 59.0, 0.0, 1.0))

    intercept_rate_D = _clip((0.22 + 0.45 * (defender["a2ad"] / 100.0) + techD) * (0.78 + 0.55 * intDq) * (0.85 + 0.30 * shield_q_D), 0.08, 0.92)
    intercept_rate_A = _clip((0.22 + 0.45 * (attacker["a2ad"] / 100.0) + techA) * (0.78 + 0.55 * intAq) * (0.85 + 0.30 * shield_q_A), 0.08, 0.92)

    interceptors_spent_D = min(defender["interceptors"], missiles_launched_A * intercept_rate_D * (0.6 + 0.6 * rng.random()))
    interceptors_spent_A = min(attacker["interceptors"], missiles_launched_D * intercept_rate_A * (0.6 + 0.6 * rng.random()))

    
    # Penetration ("leakers") shaped by breakthrough index
    pen_A = missiles_launched_A * max(0.0, (1.0 - intercept_rate_D)) * can_penetrate * (0.85 + 0.35 * brA)
    pen_D = missiles_launched_D * max(0.0, (1.0 - intercept_rate_A)) * max(0.0, (1.0 - 0.15 * can_penetrate)) * (0.85 + 0.35 * brD)

    
    strike_factor_A = float(np.clip(
        0.40
        + 0.30 * (attacker["networks_score"] / 100.0)
        + 0.18 * (attacker["future_resources_score"] / 100.0)
        + 0.12 * brA
        + 0.08 * denA,
        0.35, 1.35
    ))
    strike_factor_D = float(np.clip(
        0.40
        + 0.30 * (defender["networks_score"] / 100.0)
        + 0.18 * (defender["future_resources_score"] / 100.0)
        + 0.12 * brD
        + 0.08 * denD,
        0.35, 1.35
    ))

    air_loss_D = params["air_rate"] * pen_A * (0.0007 + 0.0006 * strike_factor_A) * (air_adv) * (0.6 + 0.8 * rng.random())
    sea_loss_D = params["sea_rate"] * pen_A * (0.00025 + 0.00022 * strike_factor_A) * (sea_adv) * (0.6 + 0.8 * rng.random())
    land_loss_D = params["land_rate"] * pen_A * (0.00050 + 0.00045 * strike_factor_A) * (land_adv) * (0.6 + 0.8 * rng.random())

    air_loss_A = params["air_rate"] * pen_D * (0.0007 + 0.0006 * strike_factor_D) * (1.0 / max(0.55, air_adv)) * (0.6 + 0.8 * rng.random())
    sea_loss_A = params["sea_rate"] * pen_D * (0.00025 + 0.00022 * strike_factor_D) * (1.0 / max(0.55, sea_adv)) * (0.6 + 0.8 * rng.random())
    land_loss_A = params["land_rate"] * pen_D * (0.00050 + 0.00045 * strike_factor_D) * (1.0 / max(0.55, land_adv)) * (0.6 + 0.8 * rng.random())

    aircraft_lost_D = int(min(D_air, max(0.0, air_loss_D * D_air)))
    aircraft_lost_A = int(min(A_air, max(0.0, air_loss_A * A_air)))

    tanks_lost_D = int(min(defender["tanks"] * commit_d, max(0.0, land_loss_D * (defender["tanks"] * commit_d))))
    tanks_lost_A = int(min(attacker["tanks"] * commit_a, max(0.0, land_loss_A * (attacker["tanks"] * commit_a))))

    ships_lost_D = int(min(D_sea, max(0.0, sea_loss_D * D_sea)))
    ships_lost_A = int(min(A_sea, max(0.0, sea_loss_A * A_sea)))

    art_lost_D = int(min(defender["artillery"] * commit_d, max(0.0, land_loss_D * (defender["artillery"] * commit_d) * 0.75)))
    art_lost_A = int(min(attacker["artillery"] * commit_a, max(0.0, land_loss_A * (attacker["artillery"] * commit_a) * 0.75)))

    contact = params["contact"]
    land_contact = (rng.random() < contact)
    base_kia_rate = (0.000006 + 0.000010 * params["shock"])
    shock_kia = 0.0000015 * (pen_A + pen_D) * 0.001

    # A_land/D_land already include committed fraction; don't double-apply commit.
    kia_D = int(D_land * (base_kia_rate * (1.1 if land_contact else 0.55) * (0.7 + 0.8 * rng.random()) + shock_kia))
    kia_A = int(A_land * (base_kia_rate * (1.1 if land_contact else 0.55) * (0.7 + 0.8 * rng.random()) + shock_kia))

    wia_D = int(kia_D * (2.2 + 0.6 * rng.random()))
    wia_A = int(kia_A * (2.2 + 0.6 * rng.random()))

    civ_base = params["civ_factor"] * (0.0000008 + 0.0000012 * params["shock"])
    civ_D = int(min(defender["population"] * 0.0008, defender["population"] * civ_base * (0.7 + 0.9 * rng.random()) + pen_A * 2.0))
    civ_A = int(min(attacker["population"] * 0.0008, attacker["population"] * civ_base * (0.7 + 0.9 * rng.random()) + pen_D * 2.0))

    return dict(
        missiles_A=float(missiles_launched_A),
        missiles_D=float(missiles_launched_D),
        interceptors_A=float(interceptors_spent_A),
        interceptors_D=float(interceptors_spent_D),
        aircraft_lost_A=aircraft_lost_A,
        aircraft_lost_D=aircraft_lost_D,
        tanks_lost_A=tanks_lost_A,
        tanks_lost_D=tanks_lost_D,
        ships_lost_A=ships_lost_A,
        ships_lost_D=ships_lost_D,
        art_lost_A=art_lost_A,
        art_lost_D=art_lost_D,
        kia_A=kia_A,
        kia_D=kia_D,
        wia_A=wia_A,
        wia_D=wia_D,
        civ_A=civ_A,
        civ_D=civ_D,
    )


def simulate_cinematic(bundle: DataBundle, sc: Scenario) -> Tuple[dict, List[dict]]:
    df = bundle.countries
    if str(sc.core_model).lower().startswith("integrated"):
        return simulate_integrated(bundle, sc)
    rng = np.random.default_rng(sc.seed)

    bg = df[df["ISO3"] == sc.battleground_iso3]
    if len(bg) == 0:
        raise ValueError("Battleground ISO3 not found in dataset.")
    bg = bg.iloc[0].to_dict()
    theatre = _theatre_modifiers(bg)

    local_side = "none"
    if sc.battleground_iso3 in sc.blue:
        local_side = "blue"
    elif sc.battleground_iso3 in sc.red:
        local_side = "red"

    blue = _side_aggregate(df, sc.blue, sc.sanctions_blue, sc.neutral_support_blue)
    red = _side_aggregate(df, sc.red, sc.sanctions_red, sc.neutral_support_red)

    def avg_distance(iso3_list):
        sub = df[df["ISO3"].isin(iso3_list)]
        if len(sub) == 0:
            return 1.6
        return float(np.mean([_continent_distance(c, bg["continent"]) for c in sub["continent"].tolist()]))

    # Theatre effects: friction slows tempo; if battleground belongs to a coalition, that side gets a local defense edge.
    dist_mult_blue = 0.75 if local_side == "blue" else (1.15 if local_side == "red" else 1.0)
    dist_mult_red = 0.75 if local_side == "red" else (1.15 if local_side == "blue" else 1.0)

    dist_blue = avg_distance(sc.blue) * theatre["friction"] * dist_mult_blue
    dist_red = avg_distance(sc.red) * theatre["friction"] * dist_mult_red

    blue_def = dict(blue)
    red_def = dict(red)
    if local_side == "blue":
        blue_def["a2ad"] = min(100.0, blue_def["a2ad"] * theatre["defender_edge"])
    elif local_side == "red":
        red_def["a2ad"] = min(100.0, red_def["a2ad"] * theatre["defender_edge"])

    bg_a2 = float(bg.get("a2ad", 50.0) or 50.0)
    bg_a2_eff = min(100.0, bg_a2 * theatre["defender_edge"])

    if local_side == "blue":
        target_blue_a2 = 0.55 * bg_a2 + 0.45 * red["a2ad"]
        target_red_a2 = 0.80 * bg_a2_eff + 0.20 * blue_def["a2ad"]
    elif local_side == "red":
        target_red_a2 = 0.55 * bg_a2 + 0.45 * blue["a2ad"]
        target_blue_a2 = 0.80 * bg_a2_eff + 0.20 * red_def["a2ad"]
    else:
        target_blue_a2 = 0.70 * bg_a2 + 0.30 * red["a2ad"]
        target_red_a2 = 0.70 * bg_a2 + 0.30 * blue["a2ad"]

    commit_blue = _commit_fraction(blue, target_blue_a2, dist_blue, sc.mode)
    commit_red = _commit_fraction(red, target_red_a2, dist_red, sc.mode)

    if local_side == "blue":
        commit_blue = _clip(commit_blue * 1.10, 0.08, 0.90)
        commit_red = _clip(commit_red * 0.92, 0.08, 0.85)
    elif local_side == "red":
        commit_red = _clip(commit_red * 1.10, 0.08, 0.90)
        commit_blue = _clip(commit_blue * 0.92, 0.08, 0.85)

    pen_blue = _penetration_chance(blue, red_def, sc.year, dist_blue, sc.mode)
    pen_red = _penetration_chance(red, blue_def, sc.year, dist_red, sc.mode)

    params = dict(_intensity_params(sc.intensity))
    # Friction slows tempo; rugged terrain increases civilian risk a bit (dispersed infrastructure + close contact)
    params["contact"] = _clip(params["contact"] / theatre["friction"], 0.10, 0.85)
    for k in ("missile_rate", "air_rate", "sea_rate", "land_rate"):
        params[k] = params[k] / theatre["friction"]
    params["civ_factor"] = _clip(params["civ_factor"] * (0.85 + 0.30 * (theatre["terrain"] / 100.0)), 0.05, 1.25)


    # Per-attacker parameterization:
    # - Cinematic: both sides share the same params
    # - Dense: params are adjusted per side using latent embeddings (uses all numeric columns)
    params_blue = dict(params)
    params_red = dict(params)

    if str(getattr(sc, "core_model", "Cinematic")).lower().startswith("dense"):
        dense = _dense_context(bundle, sc, bg, theatre, blue_def, red_def, dist_blue, dist_red, pen_blue, pen_red)

        commit_blue = _clip(commit_blue * dense["commit_blue_mult"], 0.05, 0.95)
        commit_red = _clip(commit_red * dense["commit_red_mult"], 0.05, 0.95)

        for k, mlt in dense["blue_params_mult"].items():
            if k in params_blue:
                params_blue[k] = float(_clip(params_blue[k] * float(mlt), 0.0, 10.0))
        for k, mlt in dense["red_params_mult"].items():
            if k in params_red:
                params_red[k] = float(_clip(params_red[k] * float(mlt), 0.0, 10.0))

    state = dict(
        day=0,
        will_blue=100.0,
        will_red=100.0,
        remaining=dict(
            blue=dict(missiles=blue["missiles"], interceptors=blue["interceptors"], aircraft=blue["aircraft"], tanks=blue["tanks"], ships=blue["naval"], artillery=blue["artillery"]),
            red=dict(missiles=red["missiles"], interceptors=red["interceptors"], aircraft=red["aircraft"], tanks=red["tanks"], ships=red["naval"], artillery=red["artillery"]),
        ),
        totals=dict(
            blue=dict(kia=0, wia=0, civ=0, aircraft=0, tanks=0, ships=0, artillery=0, missiles=0, interceptors=0),
            red=dict(kia=0, wia=0, civ=0, aircraft=0, tanks=0, ships=0, artillery=0, missiles=0, interceptors=0),
        )
    )

    # Initial inventories snapshot (useful for downstream visualizations)
    init_snapshot = {
        "blue": dict(state["remaining"]["blue"]),
        "red": dict(state["remaining"]["red"]),
    }

    # Job-Done / Logistics sustainment overlay (optional; uses add-on matrices when present)
    scenario_type = str(getattr(sc, "scenario_type", None) or (getattr(sc, "integrated_cfg", None) or {}).get("scenario_type") or "General")
    data_dir = getattr(bundle, "data_dir", "data")
    _jd_weights = _jd_weights_from_sheet(data_dir)
    jd = {
        "base_id": _jd_base_intensity(sc),
        "scenario_type": scenario_type,
        "weights": _jd_weights,
        "blue": _jd_init_side(df, sc.blue, getattr(sc, "neutral_support_blue", []) or []),
        "red": _jd_init_side(df, sc.red, getattr(sc, "neutral_support_red", []) or []),
    }

    series: List[dict] = []

    cum_kia_blue = 0.0
    cum_kia_red = 0.0
    cum_wia_blue = 0.0
    cum_wia_red = 0.0
    winner = "Draw"
    end_reason = "Time limit reached"
    nukes_allowed = bool(sc.nukes_allowed and sc.mode.strip().lower().startswith("total"))

    def eff_power(side: dict, rem: dict) -> float:
        air = (rem["aircraft"] / max(1.0, side["aircraft"])) * (0.55 + 0.45 * (side["air_score"] / 100.0))
        sea = (rem["ships"] / max(1.0, side["naval"])) * (0.55 + 0.45 * (side["naval_score"] / 100.0))
        land = (rem["tanks"] / max(1.0, max(1.0, side["tanks"]))) * (0.55 + 0.45 * (side["land_score"] / 100.0))
        net = (side["networks_score"] / 100.0)
        proj = side["projection"] / 100.0
        sust = (side["economy_score"] / 100.0) * (0.8 + 0.2 * (side["resilience_score"] / 100.0))
        return 100.0 * (0.33 * air + 0.22 * sea + 0.33 * land + 0.12 * net) * (0.55 + 0.45 * proj) * (0.70 + 0.30 * sust)

    for day in range(1, sc.max_days + 1):
        state["day"] = day

        blue["missiles"] = state["remaining"]["blue"]["missiles"]
        blue["interceptors"] = state["remaining"]["blue"]["interceptors"]
        red["missiles"] = state["remaining"]["red"]["missiles"]
        red["interceptors"] = state["remaining"]["red"]["interceptors"]

        p_blue = eff_power(blue, state["remaining"]["blue"])
        p_red = eff_power(red, state["remaining"]["red"])

        # Job-Done multipliers (readiness + drones + LMC + intensity shaping)
        pre_b = _jd_pre_mult(jd["blue"], day, jd["base_id"], state["remaining"]["blue"], init_snapshot["blue"], jd["scenario_type"])
        pre_r = _jd_pre_mult(jd["red"],  day, jd["base_id"], state["remaining"]["red"],  init_snapshot["red"],  jd["scenario_type"])

        can_blue = pen_blue * (0.75 + 0.5 * rng.random()) * float(pre_b["can_mult"])
        can_red  = pen_red  * (0.75 + 0.5 * rng.random()) * float(pre_r["can_mult"])

        L = _day_losses(blue, red, params_blue, sc.year, commit_blue, commit_red, can_blue, rng)
        R = _day_losses(red, blue, params_red, sc.year, commit_red, commit_blue, can_red, rng)

        def apply(side_key: str, spend: dict, taken: dict):
            rem = state["remaining"][side_key]
            tot = state["totals"][side_key]

            rem["missiles"] = max(0.0, rem["missiles"] - spend["missiles"])
            rem["interceptors"] = max(0.0, rem["interceptors"] - spend["interceptors"])
            tot["missiles"] += float(spend["missiles"])
            tot["interceptors"] += float(spend["interceptors"])

            rem["aircraft"] = max(0.0, rem["aircraft"] - taken["aircraft_lost"])
            rem["tanks"] = max(0.0, rem["tanks"] - taken["tanks_lost"])
            rem["ships"] = max(0.0, rem["ships"] - taken["ships_lost"])
            rem["artillery"] = max(0.0, rem["artillery"] - taken["art_lost"])

            tot["aircraft"] += int(taken["aircraft_lost"])
            tot["tanks"] += int(taken["tanks_lost"])
            tot["ships"] += int(taken["ships_lost"])
            tot["artillery"] += int(taken["art_lost"])
            tot["kia"] += int(taken["kia"])
            tot["wia"] += int(taken["wia"])
            tot["civ"] += int(taken["civ"])

        # Exchange 1 (blue -> red)
        apply("blue", dict(missiles=L["missiles_A"], interceptors=L["interceptors_A"]),
              dict(aircraft_lost=L["aircraft_lost_A"], tanks_lost=L["tanks_lost_A"], ships_lost=L["ships_lost_A"], art_lost=L["art_lost_A"], kia=L["kia_A"], wia=L["wia_A"], civ=L["civ_A"]))
        apply("red", dict(missiles=L["missiles_D"], interceptors=L["interceptors_D"]),
              dict(aircraft_lost=L["aircraft_lost_D"], tanks_lost=L["tanks_lost_D"], ships_lost=L["ships_lost_D"], art_lost=L["art_lost_D"], kia=L["kia_D"], wia=L["wia_D"], civ=L["civ_D"]))

        # Exchange 2 (red -> blue)
        apply("red", dict(missiles=R["missiles_A"], interceptors=R["interceptors_A"]),
              dict(aircraft_lost=R["aircraft_lost_A"], tanks_lost=R["tanks_lost_A"], ships_lost=R["ships_lost_A"], art_lost=R["art_lost_A"], kia=R["kia_A"], wia=R["wia_A"], civ=R["civ_A"]))
        apply("blue", dict(missiles=R["missiles_D"], interceptors=R["interceptors_D"]),
              dict(aircraft_lost=R["aircraft_lost_D"], tanks_lost=R["tanks_lost_D"], ships_lost=R["ships_lost_D"], art_lost=R["art_lost_D"], kia=R["kia_D"], wia=R["wia_D"], civ=R["civ_D"]))

        # Job-Done / sustainment post-step: repairs + production + endurance drain
        blue_losses_today = {
            "aircraft": float(L["aircraft_lost_A"]) + float(R["aircraft_lost_D"]),
            "tanks": float(L["tanks_lost_A"]) + float(R["tanks_lost_D"]),
            "ships": float(L["ships_lost_A"]) + float(R["ships_lost_D"]),
            "artillery": float(L["art_lost_A"]) + float(R["art_lost_D"]),
        }
        red_losses_today = {
            "aircraft": float(L["aircraft_lost_D"]) + float(R["aircraft_lost_A"]),
            "tanks": float(L["tanks_lost_D"]) + float(R["tanks_lost_A"]),
            "ships": float(L["ships_lost_D"]) + float(R["ships_lost_A"]),
            "artillery": float(L["art_lost_D"]) + float(R["art_lost_A"]),
        }
        _jd_apply_post(jd["blue"], day, float(pre_b["Id"]), state["remaining"]["blue"], init_snapshot["blue"], blue_losses_today, jd["scenario_type"])
        _jd_apply_post(jd["red"],  day, float(pre_r["Id"]), state["remaining"]["red"],  init_snapshot["red"],  red_losses_today,  jd["scenario_type"])

        def will_drop(side_key: str) -> float:
            tot = state["totals"][side_key]
            res = (blue["resilience_score"] if side_key == "blue" else red["resilience_score"]) / 100.0
            eq = tot["aircraft"] * 0.015 + tot["ships"] * 0.060 + tot["tanks"] * 0.010 + tot["artillery"] * 0.006
            ppl = (tot["kia"] * 0.000020 + tot["civ"] * 0.000004)
            return (0.010 * params["shock"] * (eq + ppl)) * (1.05 - 0.55 * res)

        state["will_blue"] = max(0.0, state["will_blue"] - will_drop("blue"))
        state["will_red"] = max(0.0, state["will_red"] - will_drop("red"))

        def mobilize(side: dict, rem: dict, sanc: float):
            econ = side["economy_score"] / 100.0
            logi = side["logistics_score"] / 100.0
            k = (0.00020 + 0.00045 * econ) * (0.6 + 0.8 * logi) * (1.0 - 0.60 * sanc)
            rem["missiles"] += k * max(0.0, side["missiles"] - rem["missiles"])
            rem["interceptors"] += (k * 1.1) * max(0.0, side["interceptors"] - rem["interceptors"])
            rem["aircraft"] += (k * 0.08) * max(0.0, side["aircraft"] - rem["aircraft"])
            rem["ships"] += (k * 0.04) * max(0.0, side["naval"] - rem["ships"])
            rem["tanks"] += (k * 0.06) * max(0.0, side["tanks"] - rem["tanks"])
            rem["artillery"] += (k * 0.05) * max(0.0, side["artillery"] - rem["artillery"])

        mobilize(blue, state["remaining"]["blue"], blue["sanc"])
        mobilize(red, state["remaining"]["red"], red["sanc"])

        if nukes_allowed:
            def nuke_trigger(side: dict, will: float, power: float) -> float:
                if not side["nukes"]:
                    return 0.0
                return _clip(0.001 + 0.010 * max(0.0, (25.0 - will) / 25.0) + 0.008 * max(0.0, (35.0 - power) / 35.0), 0.0, 0.08)

            pN_blue = nuke_trigger(blue, state["will_blue"], p_blue)
            pN_red = nuke_trigger(red, state["will_red"], p_red)

            if rng.random() < pN_blue:
                shock = 0.12 + 0.08 * rng.random()
                state["totals"]["red"]["civ"] += int(red["population"] * shock * 0.020)
                state["totals"]["red"]["kia"] += int(red["active"] * shock * 0.010)
                state["remaining"]["red"]["aircraft"] *= (1.0 - (0.18 + 0.10 * rng.random()))
                state["remaining"]["red"]["ships"] *= (1.0 - (0.10 + 0.08 * rng.random()))
                state["will_red"] *= (1.0 - (0.22 + 0.15 * rng.random()))

            if rng.random() < pN_red:
                shock = 0.12 + 0.08 * rng.random()
                state["totals"]["blue"]["civ"] += int(blue["population"] * shock * 0.020)
                state["totals"]["blue"]["kia"] += int(blue["active"] * shock * 0.010)
                state["remaining"]["blue"]["aircraft"] *= (1.0 - (0.18 + 0.10 * rng.random()))
                state["remaining"]["blue"]["ships"] *= (1.0 - (0.10 + 0.08 * rng.random()))
                state["will_blue"] *= (1.0 - (0.22 + 0.15 * rng.random()))

        series.append(dict(
            day=day,
            blue_power=p_blue,
            red_power=p_red,
            blue_will=state["will_blue"],
            red_will=state["will_red"],
            blue_aircraft_remaining=state["remaining"]["blue"]["aircraft"],
            red_aircraft_remaining=state["remaining"]["red"]["aircraft"],
            blue_ships_remaining=state["remaining"]["blue"]["ships"],
            red_ships_remaining=state["remaining"]["red"]["ships"],
            blue_tanks_remaining=state["remaining"]["blue"]["tanks"],
            red_tanks_remaining=state["remaining"]["red"]["tanks"],
            blue_missiles_remaining=state["remaining"]["blue"]["missiles"],
            red_missiles_remaining=state["remaining"]["red"]["missiles"],
            blue_artillery_remaining=state["remaining"]["blue"]["artillery"],
            red_artillery_remaining=state["remaining"]["red"]["artillery"],
            blue_interceptors_remaining=state["remaining"]["blue"]["interceptors"],
            red_interceptors_remaining=state["remaining"]["red"]["interceptors"],
            blue_total_kia=state["totals"]["blue"]["kia"],
            red_total_kia=state["totals"]["red"]["kia"],
            blue_total_civ=state["totals"]["blue"]["civ"],
            red_total_civ=state["totals"]["red"]["civ"],
        ))

        if state["will_blue"] <= 6.0 or p_blue <= 12.0:
            winner = "Red"
            end_reason = "Blue will/capability collapsed"
            break
        if state["will_red"] <= 6.0 or p_red <= 12.0:
            winner = "Blue"
            end_reason = "Red will/capability collapsed"
            break

    def recovery_days(side: dict, rem: dict) -> int:
        econ = max(0.25, side["economy_score"] / 100.0)
        logi = max(0.20, side["logistics_score"] / 100.0)
        sanc = side["sanc"]
        baseline = side["aircraft"] + side["naval"] + side["tanks"] + side["artillery"]
        remaining = rem["aircraft"] + rem["ships"] + rem["tanks"] + rem["artillery"]
        loss_frac = 0.0 if baseline <= 0 else max(0.0, (baseline - remaining) / baseline)
        days = int(90 + (loss_frac ** 1.35) * (2100 / (econ * 0.75 + logi * 0.55)) * (1.0 + 1.8 * sanc))
        return int(_clip(days, 30, 3650))

    rec_blue = recovery_days(blue, state["remaining"]["blue"])
    rec_red = recovery_days(red, state["remaining"]["red"])

    summary = dict(
        winner=winner,
        end_reason=end_reason,
        duration_days=series[-1]["day"] if series else 0,
        battleground=sc.battleground_iso3,
        year=sc.year,
        intensity=sc.intensity,
        mode=sc.mode,
        theatre=theatre,
        local_side=local_side,
        penetration=dict(blue=float(pen_blue), red=float(pen_red)),
        commit=dict(blue=float(commit_blue), red=float(commit_red)),
        distance=dict(blue=float(dist_blue), red=float(dist_red)),
        totals=state["totals"],
        init=init_snapshot,
        recovery_days=dict(blue=rec_blue, red=rec_red),
        doctrines=dict(blue=blue["doctrines"], red=red["doctrines"]),
        nukes_allowed=bool(nukes_allowed),
    )
    return summary, series


def monte_carlo_fast(bundle: DataBundle, sc: Scenario, iters: int, chunk: int = 200_000, seed: int = 123, progress_callback=None) -> dict:
    df = bundle.countries
    rng = np.random.default_rng(seed)

    bg = df[df["ISO3"] == sc.battleground_iso3]
    if len(bg) == 0:
        raise ValueError("Battleground ISO3 not found in dataset.")
    bg = bg.iloc[0].to_dict()
    theatre = _theatre_modifiers(bg)

    local_side = "none"
    if sc.battleground_iso3 in sc.blue:
        local_side = "blue"
    elif sc.battleground_iso3 in sc.red:
        local_side = "red"

    blue = _side_aggregate(df, sc.blue, sc.sanctions_blue, sc.neutral_support_blue)
    red = _side_aggregate(df, sc.red, sc.sanctions_red, sc.neutral_support_red)

    def avg_distance(iso3_list):
        sub = df[df["ISO3"].isin(iso3_list)]
        if len(sub) == 0:
            return 1.6
        return float(np.mean([_continent_distance(c, bg["continent"]) for c in sub["continent"].tolist()]))


    dist_mult_blue = 0.75 if local_side == "blue" else (1.15 if local_side == "red" else 1.0)
    dist_mult_red = 0.75 if local_side == "red" else (1.15 if local_side == "blue" else 1.0)

    dist_blue = avg_distance(sc.blue) * theatre["friction"] * dist_mult_blue
    dist_red = avg_distance(sc.red) * theatre["friction"] * dist_mult_red

    blue_def = dict(blue)
    red_def = dict(red)
    if local_side == "blue":
        blue_def["a2ad"] = min(100.0, blue_def["a2ad"] * theatre["defender_edge"])
    elif local_side == "red":
        red_def["a2ad"] = min(100.0, red_def["a2ad"] * theatre["defender_edge"])

    pen_blue = _penetration_chance(blue, red_def, sc.year, dist_blue, sc.mode)
    pen_red = _penetration_chance(red, blue_def, sc.year, dist_red, sc.mode)

    params = dict(_intensity_params(sc.intensity))
    for k in ("missile_rate", "air_rate", "sea_rate", "land_rate"):
        params[k] = params[k] / theatre["friction"]
    params["contact"] = _clip(params["contact"] / theatre["friction"], 0.10, 0.85)
    params["civ_factor"] = _clip(params["civ_factor"] * (0.85 + 0.30 * (theatre["terrain"] / 100.0)), 0.05, 1.25)

    def base_power(side: dict) -> float:
        inv = (
            0.36 * _log1p(side["aircraft"]) +
            0.20 * _log1p(side["naval"]) +
            0.24 * _log1p(side["tanks"]) +
            0.20 * _log1p(side["artillery"])
        )
        scores = (
            0.28 * (side["milcap_score"] / 100.0) +
            0.18 * (side["networks_score"] / 100.0) +
            0.16 * (side["logistics_score"] / 100.0) +
            0.16 * (side["resilience_score"] / 100.0) +
            0.22 * (side["future_resources_score"] / 100.0)
        )
        econ = 0.10 * _log1p(side["gdp_usd"] / 1e11) + 0.08 * _log1p(side["mil_spend_usd"] / 1e10)
        return 100.0 * (0.65 * inv + 0.35 * scores + econ)

    pb = base_power(blue) * (0.75 + 0.25 * pen_blue)
    pr = base_power(red) * (0.75 + 0.25 * pen_red)

    bg_a2 = float(bg.get("a2ad", 50.0) or 50.0)
    bg_a2_eff = min(100.0, bg_a2 * theatre["defender_edge"])

    if local_side == "blue":
        target_blue_a2 = 0.55 * bg_a2 + 0.45 * red["a2ad"]
        target_red_a2 = 0.80 * bg_a2_eff + 0.20 * blue_def["a2ad"]
    elif local_side == "red":
        target_red_a2 = 0.55 * bg_a2 + 0.45 * blue["a2ad"]
        target_blue_a2 = 0.80 * bg_a2_eff + 0.20 * red_def["a2ad"]
    else:
        target_blue_a2 = 0.70 * bg_a2 + 0.30 * red["a2ad"]
        target_red_a2 = 0.70 * bg_a2 + 0.30 * blue["a2ad"]

    commit_blue = _commit_fraction(blue, target_blue_a2, dist_blue, sc.mode)
    commit_red = _commit_fraction(red, target_red_a2, dist_red, sc.mode)

    if local_side == "blue":
        commit_blue = _clip(commit_blue * 1.10, 0.08, 0.90)
        commit_red = _clip(commit_red * 0.92, 0.08, 0.85)
    elif local_side == "red":
        commit_red = _clip(commit_red * 1.10, 0.08, 0.90)
        commit_blue = _clip(commit_blue * 0.92, 0.08, 0.85)

    pb *= (0.70 + 0.30 * commit_blue)
    pr *= (0.70 + 0.30 * commit_red)

    techB = _tech_leap_bonus(sc.year, 100.0 * blue["tech_index"])
    techR = _tech_leap_bonus(sc.year, 100.0 * red["tech_index"])
    pb *= (1.0 + 0.10 * techB)
    pr *= (1.0 + 0.10 * techR)

    nukes_on = bool(sc.nukes_allowed and sc.mode.strip().lower().startswith("total"))
    nucB = 1.0 if (nukes_on and blue["nukes"]) else 0.0
    nucR = 1.0 if (nukes_on and red["nukes"]) else 0.0

    ratio = pb / max(1e-6, pr)

    # Closed-loop control multiplier (used only when progress_callback returns a dict)
    ratio_ctrl = 1.0

    base_days_raw = 45 + 180 * (1.0 / max(0.55, abs(math.log(ratio)) + 0.35)) + 120 * (1.0 - max(pen_blue, pen_red))
    base_days = int(_clip(base_days_raw * theatre["friction"], 20, sc.max_days))
    parity = 1.0 / (1.0 + abs(math.log(ratio)))

    base_kia = 5000 * params["shock"] * (base_days / 120.0) * (0.70 + 0.70 * parity)
    base_civ = 9000 * params["shock"] * (base_days / 120.0) * (0.65 + 0.85 * parity)

    base_air_loss = (0.012 * params["shock"] * base_days) * (0.65 + 0.60 * parity)
    base_ship_loss = (0.006 * params["shock"] * base_days) * (0.55 + 0.70 * parity)
    base_tank_loss = (0.010 * params["shock"] * base_days) * (0.60 + 0.70 * parity)

    base_miss_spent = (params["missile_rate"] * base_days) * 0.85
    base_int_spent = (params["missile_rate"] * base_days) * 0.70



    # Dense model adjustment (optional): scale loss/spend rates per side using the same latent-driven logic.
    base_air_loss_blue = base_air_loss
    base_air_loss_red = base_air_loss
    base_ship_loss_blue = base_ship_loss
    base_ship_loss_red = base_ship_loss
    base_tank_loss_blue = base_tank_loss
    base_tank_loss_red = base_tank_loss
    base_miss_spent_blue = base_miss_spent
    base_miss_spent_red = base_miss_spent
    base_int_spent_blue = base_int_spent
    base_int_spent_red = base_int_spent

    if str(getattr(sc, "core_model", "Cinematic")).lower().startswith("dense"):
        dense = _dense_context(bundle, sc, bg, theatre, blue_def, red_def, dist_blue, dist_red, pen_blue, pen_red)

        commit_blue = _clip(commit_blue * dense["commit_blue_mult"], 0.05, 0.95)
        commit_red = _clip(commit_red * dense["commit_red_mult"], 0.05, 0.95)

        # Map per-side param multipliers into MC's coarse base rates.
        bmul = dense["blue_params_mult"]
        rmul = dense["red_params_mult"]

        base_air_loss_blue *= float(bmul.get("air_rate", 1.0))
        base_air_loss_red *= float(rmul.get("air_rate", 1.0))
        base_ship_loss_blue *= float(bmul.get("sea_rate", 1.0))
        base_ship_loss_red *= float(rmul.get("sea_rate", 1.0))
        base_tank_loss_blue *= float(bmul.get("land_rate", 1.0))
        base_tank_loss_red *= float(rmul.get("land_rate", 1.0))

        base_miss_spent_blue *= float(bmul.get("missile_rate", 1.0))
        base_miss_spent_red *= float(rmul.get("missile_rate", 1.0))
        base_int_spent_blue *= float(bmul.get("missile_rate", 1.0))
        base_int_spent_red *= float(rmul.get("missile_rate", 1.0))

        # Civilian + KIA baseline is driven by contact + land intensity (coarse).
        mc_kia_mult_blue = float(np.sqrt(max(0.01, bmul.get("contact", 1.0) * bmul.get("land_rate", 1.0))))
        mc_kia_mult_red = float(np.sqrt(max(0.01, rmul.get("contact", 1.0) * rmul.get("land_rate", 1.0))))
        mc_civ_mult_blue = float(np.sqrt(max(0.01, bmul.get("civ_factor", 1.0) * bmul.get("contact", 1.0))))
        mc_civ_mult_red = float(np.sqrt(max(0.01, rmul.get("civ_factor", 1.0) * rmul.get("contact", 1.0))))
    else:
        mc_kia_mult_blue = 1.0
        mc_kia_mult_red = 1.0
        mc_civ_mult_blue = 1.0
        mc_civ_mult_red = 1.0

    iters = int(max(1, min(int(iters), 10_000_000)))
    wins = dict(Blue=0, Red=0, Draw=0)

    cap_samples = 200_000
    store_every = max(1, iters // cap_samples)

    samples = {k: [] for k in [
        "winner", "duration",
        "blue_kia", "red_kia", "blue_wia", "red_wia", "blue_civ", "red_civ",
        "blue_air", "red_air", "blue_ship", "red_ship", "blue_tank", "red_tank",
        "blue_miss", "red_miss", "blue_int", "red_int"
    ]}

    def outcome_prob(r: np.ndarray) -> np.ndarray:
        x = 2.0 * (np.log(r + 1e-9))
        p = _sigmoid(x)
        stalemate = 0.22 * (1.0 - max(pen_blue, pen_red))
        p = (1.0 - stalemate) * p
        return np.clip(p, 0.02, 0.98)

    done = 0
    while done < iters:
        m = min(chunk, iters - done)
        done2 = done + m

        fog = np.exp(rng.normal(0.0, 0.22, size=m))
        r_eff = (ratio * ratio_ctrl) * fog

        pB = outcome_prob(r_eff)
        u = rng.random(m)

        draw_prob = 0.10 + 0.25 * (1.0 - max(pen_blue, pen_red)) + 0.10 * (1.0 / (1.0 + np.abs(np.log(r_eff + 1e-9))))
        draw_prob = np.clip(draw_prob, 0.05, 0.45)

        is_draw = u < draw_prob
        u2 = (u - draw_prob) / np.clip(1.0 - draw_prob, 1e-9, 1.0)

        is_blue = (~is_draw) & (u2 < pB)
        is_red = (~is_draw) & (~is_blue)

        wins["Blue"] += int(np.sum(is_blue))
        wins["Red"] += int(np.sum(is_red))
        wins["Draw"] += int(np.sum(is_draw))

        dur = base_days * (0.85 + 0.35 * rng.random(m))
        dur = np.where(is_draw, dur * (1.15 + 0.35 * rng.random(m)), dur)
        dur = np.where(is_blue, dur * (0.88 + 0.22 * (1.0 / np.clip(r_eff, 0.35, 3.0))), dur)
        dur = np.where(is_red, dur * (0.88 + 0.22 * (np.clip(r_eff, 0.35, 3.0))), dur)
        dur = np.clip(dur, 10, sc.max_days)

        if nukes_on and (nucB or nucR):
            loseB = is_red.astype(float)
            loseR = is_blue.astype(float)
            pN_B = (0.0006 + 0.006 * loseB) * nucB
            pN_R = (0.0006 + 0.006 * loseR) * nucR
            nukeB = rng.random(m) < pN_B
            nukeR = rng.random(m) < pN_R
            nuke_any = nukeB | nukeR
        else:
            nuke_any = np.zeros(m, dtype=bool)

        scale = (dur / base_days) * (0.85 + 0.55 * rng.random(m)) * (0.90 + 0.30 * (1.0 / (1.0 + np.abs(np.log(r_eff + 1e-9)))))
        nuke_mult = np.where(nuke_any, (8.0 + 6.0 * rng.random(m)), 1.0)

        kia_total = base_kia * scale * nuke_mult * np.sqrt(mc_kia_mult_blue * mc_kia_mult_red)
        civ_total = base_civ * scale * nuke_mult * np.sqrt(mc_civ_mult_blue * mc_civ_mult_red)
        wia_total = kia_total * (2.0 + 0.8 * rng.random(m))

        lose_weight_blue = np.where(is_red, 0.62, np.where(is_blue, 0.42, 0.50))
        lose_weight_red = 1.0 - lose_weight_blue

        kia_bias = mc_kia_mult_blue / max(1e-9, (mc_kia_mult_blue + mc_kia_mult_red))
        civ_bias = mc_civ_mult_blue / max(1e-9, (mc_civ_mult_blue + mc_civ_mult_red))

        lose_weight_blue_kia = np.clip(0.65 * lose_weight_blue + 0.35 * kia_bias, 0.18, 0.82)
        lose_weight_blue_civ = np.clip(0.65 * lose_weight_blue + 0.35 * civ_bias, 0.18, 0.82)

        blue_kia = kia_total * lose_weight_blue_kia
        red_kia = kia_total * (1.0 - lose_weight_blue_kia)
        blue_civ = civ_total * lose_weight_blue_civ
        red_civ = civ_total * (1.0 - lose_weight_blue_civ)

        blue_wia = wia_total * lose_weight_blue
        red_wia = wia_total * lose_weight_red

        blue_air = np.clip(base_air_loss_blue * scale * lose_weight_blue, 0, max(1.0, blue["aircraft"] * commit_blue * 0.40))
        red_air = np.clip(base_air_loss_red * scale * lose_weight_red, 0, max(1.0, red["aircraft"] * commit_red * 0.40))
        blue_ship = np.clip(base_ship_loss_blue * scale * lose_weight_blue, 0, max(1.0, blue["naval"] * commit_blue * 0.25))
        red_ship = np.clip(base_ship_loss_red * scale * lose_weight_red, 0, max(1.0, red["naval"] * commit_red * 0.25))
        blue_tank = np.clip(base_tank_loss_blue * scale * lose_weight_blue, 0, max(1.0, blue["tanks"] * commit_blue * 0.35))
        red_tank = np.clip(base_tank_loss_red * scale * lose_weight_red, 0, max(1.0, red["tanks"] * commit_red * 0.35))

        blue_miss = np.clip(blue["missiles"] * base_miss_spent_blue * (scale / base_days), 0, blue["missiles"])
        red_miss = np.clip(red["missiles"] * base_miss_spent_red * (scale / base_days), 0, red["missiles"])
        blue_int = np.clip(blue["interceptors"] * base_int_spent_blue * (scale / base_days), 0, blue["interceptors"])
        red_int = np.clip(red["interceptors"] * base_int_spent_red * (scale / base_days), 0, red["interceptors"])

        idx = np.arange(done, done2)
        keep = (idx % store_every) == 0
        if np.any(keep):
            wlab = np.where(is_draw[keep], "Draw", np.where(is_blue[keep], "Blue", "Red"))
            samples["winner"].extend(wlab.tolist())
            samples["duration"].extend(dur[keep].tolist())
            samples["blue_kia"].extend(blue_kia[keep].tolist())
            samples["red_kia"].extend(red_kia[keep].tolist())
            samples["blue_wia"].extend(blue_wia[keep].tolist())
            samples["red_wia"].extend(red_wia[keep].tolist())
            samples["blue_civ"].extend(blue_civ[keep].tolist())
            samples["red_civ"].extend(red_civ[keep].tolist())
            samples["blue_air"].extend(blue_air[keep].tolist())
            samples["red_air"].extend(red_air[keep].tolist())
            samples["blue_ship"].extend(blue_ship[keep].tolist())
            samples["red_ship"].extend(red_ship[keep].tolist())
            samples["blue_tank"].extend(blue_tank[keep].tolist())
            samples["red_tank"].extend(red_tank[keep].tolist())
            samples["blue_miss"].extend(blue_miss[keep].tolist())
            samples["red_miss"].extend(red_miss[keep].tolist())
            samples["blue_int"].extend(blue_int[keep].tolist())
            samples["red_int"].extend(red_int[keep].tolist())

        done = done2
        if progress_callback is not None:
            payload = None
            try:
                total_so_far = max(1, sum(wins.values()))
                payload = dict(
                    win_prob=dict(
                        Blue=wins["Blue"] / total_so_far,
                        Red=wins["Red"] / total_so_far,
                        Draw=wins["Draw"] / total_so_far,
                    ),
                    done=int(done),
                    total=int(iters),
                    samples_n=int(len(samples.get("duration", []))),
                )
                for k in ("blue_air", "red_air", "blue_ship", "red_ship", "blue_tank", "red_tank", "blue_kia", "red_kia", "blue_miss", "red_miss", "blue_int", "red_int"):
                    arr = samples.get(k, [])
                    if arr:
                        payload[k] = float(np.mean(np.asarray(arr, dtype=float)))
            except Exception:
                payload = None

            ret = None
            try:
                ret = progress_callback(int(done), int(iters), payload)
            except TypeError:
                try:
                    ret = progress_callback(int(done), int(iters))
                except Exception:
                    ret = None
            except Exception:
                ret = None

            # Allow chess (or other UI control loops) to bias the remaining chunks.
            if isinstance(ret, dict):
                try:
                    ratio_ctrl = float(np.clip(ratio_ctrl * float(ret.get('ratio_ctrl', 1.0)), 0.75, 1.35))
                except Exception:
                    pass

    total = sum(wins.values())
    probs = {k: (wins[k] / total if total else 0.0) for k in wins}

    def pct(arr, p):
        if len(arr) == 0:
            return 0.0
        return float(np.percentile(np.asarray(arr, dtype=float), p))

    def pack(prefix):
        return dict(p10=pct(samples[prefix], 10), p50=pct(samples[prefix], 50), p90=pct(samples[prefix], 90),
                    mean=float(np.mean(np.asarray(samples[prefix], dtype=float))) if len(samples[prefix]) else 0.0)

    return dict(
        iters=int(iters),
        win_prob=probs,
        duration=pack("duration"),
        blue=dict(kia=pack("blue_kia"), wia=pack("blue_wia"), civ=pack("blue_civ"),
                  aircraft=pack("blue_air"), ships=pack("blue_ship"), tanks=pack("blue_tank"),
                  missiles=pack("blue_miss"), interceptors=pack("blue_int")),
        red=dict(kia=pack("red_kia"), wia=pack("red_wia"), civ=pack("red_civ"),
                 aircraft=pack("red_air"), ships=pack("red_ship"), tanks=pack("red_tank"),
                 missiles=pack("red_miss"), interceptors=pack("red_int")),
        meta=dict(ratio=float(ratio),
                  penetration=dict(blue=float(pen_blue), red=float(pen_red)),
                  commit=dict(blue=float(commit_blue), red=float(commit_red)),
                  distance=dict(blue=float(dist_blue), red=float(dist_red)),
                  theatre=theatre,
                  local_side=local_side,
                  nukes_allowed=bool(nukes_on))
    )


def monte_carlo_integrated_fast(
    bundle: DataBundle,
    sc: Scenario,
    iters: int,
    chunk: int = 200_000,
    seed: int = 123,
    progress_callback=None,
) -> dict:
    """Fast Monte Carlo surrogate for the Integrated (4-engine) model.

    - Keeps the same data plumbing as the rest of the app.
    - Uses a vectorized, Hughes-inspired + logistics-gated surrogate.
    - Output schema matches monte_carlo_fast().

    This is still an educational toy model.
    """
    df = bundle.countries
    rng = np.random.default_rng(seed)

    bg = df[df["ISO3"] == sc.battleground_iso3]
    if len(bg) == 0:
        raise ValueError("Battleground ISO3 not found in dataset.")
    bg_row = bg.iloc[0].to_dict()
    theatre = _theatre_modifiers(bg_row)

    local_side = "none"
    if sc.battleground_iso3 in sc.blue:
        local_side = "blue"
    elif sc.battleground_iso3 in sc.red:
        local_side = "red"

    blue = _side_aggregate(df, sc.blue, sc.sanctions_blue, sc.neutral_support_blue)
    red = _side_aggregate(df, sc.red, sc.sanctions_red, sc.neutral_support_red)

    icfg = (getattr(sc, "integrated_cfg", None) or {})
    days = int(max(7, min(int(icfg.get("days", 24)), int(sc.max_days))))

    def _mix_auto(row: dict) -> dict:
        terr = float(row.get("terrain_score_0_100", 50.0) or 50.0) / 100.0
        trn = float(row.get("transport_score_0_100", 50.0) or 50.0) / 100.0
        a2 = float(row.get("a2ad", 50.0) or 50.0) / 100.0
        area = float(row.get("area_km2", row.get("area", 0.0)) or 0.0)
        per = float(row.get("perimeter_km", row.get("perimeter", 0.0)) or 0.0)
        shape = 0.0
        if area > 0 and per > 0:
            shape = per / max(1e-6, math.sqrt(area))
        coastal = float(np.clip((shape - 0.9) / 1.2, 0.0, 1.0))
        sea = 0.22 + 0.52 * coastal + 0.10 * trn + 0.06 * (1.0 - terr)
        land = 0.26 + 0.46 * terr + 0.06 * (1.0 - trn)
        air = max(0.10, 1.0 - sea - land)
        v = np.array([air, sea, land], dtype=float)
        v = np.clip(v, 0.05, None)
        v = v / v.sum()
        return dict(air=float(v[0]), sea=float(v[1]), land=float(v[2]), standoff=float(np.clip(0.20 + 0.55 * a2, 0.05, 0.95)))

    mix = _mix_auto(bg_row)
    override = icfg.get("theatre_mix_override")
    blend = float(np.clip(icfg.get("theatre_mix_blend", 0.0), 0.0, 1.0))
    if isinstance(override, dict) and blend > 0:
        o = np.array([
            float(override.get("air", mix["air"])),
            float(override.get("sea", mix["sea"])),
            float(override.get("land", mix["land"])),
        ], dtype=float)
        o = np.clip(o, 0.0, None)
        if o.sum() > 0:
            o = o / o.sum()
            v = (1.0 - blend) * np.array([mix["air"], mix["sea"], mix["land"]]) + blend * o
            v = v / max(1e-9, v.sum())
            mix["air"], mix["sea"], mix["land"] = float(v[0]), float(v[1]), float(v[2])

    commit_mult = float(icfg.get("commit_mult", 1.0))
    commit_blue = float(np.clip(icfg.get("commit_blue", 0.35 * commit_mult), 0.05, 0.95))
    default_red = 0.45 if str(icfg.get("invader", "red")).lower().startswith("red") else 0.35
    commit_red = float(np.clip(icfg.get("commit_red", default_red * commit_mult), 0.05, 0.95))

    params = _intensity_params(sc.intensity)
    miss_rate = float(params.get("missile_rate", 0.022)) / float(theatre.get("friction", 1.0))
    base_scale = float(np.clip(0.35 + 20.0 * miss_rate, 0.45, 1.25))

    def _miss_scale(side: dict) -> float:
        sat = float(side.get("missile_saturation", 50.0) or 50.0)
        den = float(side.get("missile_density", 50.0) or 50.0)
        tier = float(side.get("missile_tier", 3.0) or 3.0)
        return base_scale * (0.70 + 0.006 * sat) * (0.85 + 0.003 * (den - 50.0)) * (0.92 + 0.03 * (4.0 - min(4.0, tier)))

    b_scale = _miss_scale(blue)
    r_scale = _miss_scale(red)

    b_miss = float(blue.get("missiles", 0.0) or 0.0) * commit_blue
    r_miss = float(red.get("missiles", 0.0) or 0.0) * commit_red
    b_int = float(blue.get("interceptors", 0.0) or 0.0) * commit_blue
    r_int = float(red.get("interceptors", 0.0) or 0.0) * commit_red

    b_mpd = min(b_miss, max(0.0, (b_miss / max(1.0, days)) * b_scale))
    r_mpd = min(r_miss, max(0.0, (r_miss / max(1.0, days)) * r_scale))
    b_total = float(min(b_miss, b_mpd * days))
    r_total = float(min(r_miss, r_mpd * days))

    # Base totals (tempo control applies as a multiplier per chunk, capped by stock)
    b_total0 = float(b_total)
    r_total0 = float(r_total)

    def _pk(side: dict) -> float:
        net = float(side.get("networks_score", 50.0) or 50.0) / 100.0
        fut = float(side.get("future_resources_score", 50.0) or 50.0) / 100.0
        br = float(side.get("missile_breakthrough", 50.0) or 50.0) / 100.0
        return float(np.clip(0.12 + 0.14 * net + 0.10 * fut + 0.14 * br, 0.10, 0.55))

    b_pk = _pk(blue)
    r_pk = _pk(red)

    def _pd(side: dict) -> float:
        iq = float(side.get("missile_intercept", 50.0) or 50.0) / 100.0
        rank = float(side.get("missile_shield_rank", 25.0) or 25.0)
        shq = _ic_shield_quality(rank)
        return float(np.clip(0.18 + 0.18 * (iq - 0.5) + 0.16 * (shq - 0.5), 0.05, 0.65))

    b_pd = _pd(blue)
    r_pd = _pd(red)

    ship_frac_default = float(np.clip(icfg.get("missile_fraction_to_ships", 0.60), 0.20, 0.90))
    # Separate per-side target fractions (allows asymmetric doctrines + chess coupling)
    ship_frac_blue_ctrl = float(np.clip(icfg.get("missile_fraction_to_ships_blue", ship_frac_default), 0.05, 0.95))
    ship_frac_red_ctrl  = float(np.clip(icfg.get("missile_fraction_to_ships_red",  ship_frac_default), 0.05, 0.95))

    # Closed-loop control knobs (can be updated chunk-by-chunk via progress_callback return dict)
    det_prob_ctrl = float(np.clip(icfg.get("blue_detection_prob", 0.60), 0.15, 0.95))
    pk_mult_blue = 1.0
    pk_mult_red  = 1.0
    pd_mult_blue = 1.0
    pd_mult_red  = 1.0
    tempo_mult_blue = 1.0
    tempo_mult_red  = 1.0
    ratio_ctrl = 1.0
    shock_ctrl = 1.0
    land_gate_mult = 1.0

    b_ships = max(1.0, float(blue.get("naval", 0.0) or 0.0) * commit_blue)
    r_ships = max(1.0, float(red.get("naval", 0.0) or 0.0) * commit_red)
    b_car = float(np.clip(float(blue.get("carriers", 0.0) or 0.0), 0.0, b_ships))
    r_car = float(np.clip(float(red.get("carriers", 0.0) or 0.0), 0.0, r_ships))

    b_air = max(1.0, float(blue.get("combat_aircraft", blue.get("aircraft", 0.0)) or 0.0) * commit_blue)
    r_air = max(1.0, float(red.get("combat_aircraft", red.get("aircraft", 0.0)) or 0.0) * commit_red)
    b_tanks = max(1.0, float(blue.get("tanks", 0.0) or 0.0) * commit_blue)
    r_tanks = max(1.0, float(red.get("tanks", 0.0) or 0.0) * commit_red)

    port_capacity = float(25_000.0 + 700.0 * float(bg_row.get("transport_score_0_100", 50.0) or 50.0))
    consumption = float(np.clip(260.0 + 60.0 * float(params.get("contact", 0.5)), 200.0, 450.0))
    lots = float(max(0.0, (float(red.get("logistics_score", 50.0) or 50.0) / 100.0) * 0.25 * 60_000.0))
    integrity = float(np.clip(icfg.get("port_integrity", 1.0), 0.05, 1.0))
    supportable = _ic_supportable_battalions(port_capacity, integrity, consumption, lots)

    iters = int(max(1, min(int(iters), 10_000_000)))
    wins = dict(Blue=0, Red=0, Draw=0)

    cap_samples = 200_000
    store_every = max(1, iters // cap_samples)
    samples = {k: [] for k in [
        "winner", "duration",
        "blue_kia", "red_kia", "blue_wia", "red_wia", "blue_civ", "red_civ",
        "blue_air", "red_air", "blue_ship", "red_ship", "blue_tank", "red_tank",
        "blue_miss", "red_miss", "blue_int", "red_int"
    ]}

    def base_power(side: dict) -> float:
        inv = 0.38 * _log1p(side["combat_aircraft"]) + 0.20 * _log1p(side["naval"]) + 0.22 * _log1p(side["tanks"]) + 0.20 * _log1p(side["artillery"])
        qual = 0.30 * (side["milcap_score"] / 100.0) + 0.22 * (side["networks_score"] / 100.0) + 0.18 * (side["logistics_score"] / 100.0) + 0.15 * (side["resilience_score"] / 100.0) + 0.15 * (side["future_resources_score"] / 100.0)
        miss = 0.10 * (side.get("missile_breakthrough", 50.0) / 100.0) + 0.10 * (side.get("missile_saturation", 50.0) / 100.0)
        return 100.0 * (0.70 * inv + 0.30 * qual + 0.18 * miss)

    pb = base_power(blue) * (0.88 + 0.25 * mix["air"] + 0.15 * mix["sea"]) * (0.80 + 0.40 * commit_blue)
    pr = base_power(red)  * (0.88 + 0.25 * mix["air"] + 0.15 * mix["sea"]) * (0.80 + 0.40 * commit_red)
    ratio = pb / max(1e-6, pr)

    done = 0
    while done < iters:
        m = min(chunk, iters - done)
        done2 = done + m

        fog = np.exp(rng.normal(0.0, 0.20, size=m))
        r_eff = (ratio * ratio_ctrl) * fog

        det_prob = det_prob_ctrl
        det_hits = rng.binomial(days, det_prob, size=m)
        det_frac = det_hits / max(1.0, days)

        x = 2.2 * np.log(r_eff + 1e-9)
        p_blue = _sigmoid(x)
        stalemate = 0.12 + 0.18 * (1.0 - max(mix["land"], 0.25))
        stalemate = np.clip(stalemate, 0.05, 0.35)
        p_blue = (1.0 - stalemate) * p_blue
        u = rng.random(m)
        is_draw = u < stalemate
        u2 = (u - stalemate) / np.clip(1.0 - stalemate, 1e-9, 1.0)
        is_blue = (~is_draw) & (u2 < p_blue)
        is_red = (~is_draw) & (~is_blue)

        wins["Blue"] += int(np.sum(is_blue))
        wins["Red"] += int(np.sum(is_red))
        wins["Draw"] += int(np.sum(is_draw))

        dur = float(days) * (0.85 + 0.35 * rng.random(m))
        dur = np.where(is_draw, dur * (1.10 + 0.30 * rng.random(m)), dur)
        dur = np.clip(dur, 7, sc.max_days)

        b_total_eff = float(np.clip(b_total0 * tempo_mult_blue, 0.0, b_miss))
        r_total_eff = float(np.clip(r_total0 * tempo_mult_red,  0.0, r_miss))

        M_b = b_total_eff * ship_frac_blue_ctrl * (0.90 + 0.25 * rng.random(m))
        M_r = r_total_eff * ship_frac_red_ctrl  * (0.90 + 0.25 * rng.random(m))
        D_b = b_int * (0.55 + 0.50 * (np.clip(b_pd * pd_mult_blue, 0.05, 0.85) + 0.15 * det_frac)) * (dur / max(1.0, days))
        D_r = r_int * (0.55 + 0.50 * (np.clip(r_pd * pd_mult_red, 0.05, 0.85) + 0.15 * (1.0 - det_frac))) * (dur / max(1.0, days))

        inter_b = rng.binomial(np.minimum(M_r, D_b).astype(int), np.clip((b_pd * pd_mult_blue) + 0.10 * det_frac, 0.05, 0.85))
        inter_r = rng.binomial(np.minimum(M_b, D_r).astype(int), np.clip((r_pd * pd_mult_red) + 0.10 * (1.0 - det_frac), 0.05, 0.85))
        through_r = np.maximum(0, M_r.astype(int) - inter_b)
        through_b = np.maximum(0, M_b.astype(int) - inter_r)

        hits_r = rng.binomial(through_r, np.clip((r_pk * pk_mult_red) * (0.85 + 0.35 * (1.0 - det_frac)), 0.05, 0.95))
        hits_b = rng.binomial(through_b, np.clip((b_pk * pk_mult_blue) * (0.85 + 0.35 * det_frac), 0.05, 0.95))

        car_share_b = np.clip((b_car / max(1.0, b_ships)) * (0.55 + 0.55 * mix["sea"]), 0.02, 0.18)
        car_share_r = np.clip((r_car / max(1.0, r_ships)) * (0.55 + 0.55 * mix["sea"]), 0.02, 0.18)
        car_hits_b = rng.binomial(hits_r, car_share_b)
        car_hits_r = rng.binomial(hits_b, car_share_r)
        car_lost_b = np.minimum(b_car, (car_hits_b // 4).astype(float))
        car_lost_r = np.minimum(r_car, (car_hits_r // 4).astype(float))
        other_lost_b = np.minimum(b_ships - b_car, (hits_r - car_hits_b).astype(float))
        other_lost_r = np.minimum(r_ships - r_car, (hits_b - car_hits_r).astype(float))
        blue_ship = np.clip(car_lost_b + other_lost_b, 0, b_ships)
        red_ship  = np.clip(car_lost_r + other_lost_r, 0, r_ships)

        parity = 1.0 / (1.0 + np.abs(np.log(r_eff + 1e-9)))
        air_loss_rate = float(params.get("air_rate", 0.014)) / float(theatre.get("friction", 1.0))
        land_loss_rate = float(params.get("land_rate", 0.012)) / float(theatre.get("friction", 1.0))
        blue_air = np.clip((air_loss_rate * dur) * (0.55 + 0.55 * parity) * (0.55 + 0.55 * (1.0 - p_blue)) * b_air * 0.04, 0, b_air * 0.55)
        red_air  = np.clip((air_loss_rate * dur) * (0.55 + 0.55 * parity) * (0.55 + 0.55 * p_blue) * r_air * 0.04, 0, r_air * 0.55)

        land_gate = land_gate_mult * mix["land"] * (0.55 + 0.55 * params.get("contact", 0.5))
        supply_ratio = np.clip(supportable / max(1.0, 0.85 * (r_tanks / 900.0 + 5.0)), 0.05, 1.40)
        inv_pen = np.where(is_red, (1.0 / np.clip(supply_ratio, 0.30, 1.40)), 1.0)
        blue_tank = np.clip((land_loss_rate * dur) * land_gate * (0.55 + 0.55 * parity) * (0.55 + 0.55 * (1.0 - p_blue)) * b_tanks * 0.04, 0, b_tanks * 0.55)
        red_tank  = np.clip((land_loss_rate * dur) * land_gate * (0.55 + 0.55 * parity) * (0.55 + 0.55 * p_blue) * r_tanks * 0.04 * inv_pen, 0, r_tanks * 0.60)

        shock = float(params.get("shock", 1.1)) * float(shock_ctrl)
        kia_total = 4800 * shock * (dur / max(1.0, days)) * (0.60 + 0.90 * parity) * (0.35 + 1.10 * land_gate)
        wia_total = kia_total * (2.1 + 0.9 * rng.random(m))
        civ_total = 8200 * float(params.get("civ_factor", 0.45)) * (dur / max(1.0, days)) * (0.55 + 0.90 * parity) * (0.25 + 1.00 * land_gate)
        lose_weight_blue = np.where(is_red, 0.62, np.where(is_blue, 0.42, 0.50))
        blue_kia = kia_total * lose_weight_blue
        red_kia = kia_total * (1.0 - lose_weight_blue)
        blue_wia = wia_total * lose_weight_blue
        red_wia = wia_total * (1.0 - lose_weight_blue)
        blue_civ = civ_total * (0.50 + 0.10 * rng.random(m))
        red_civ = civ_total - blue_civ

        blue_miss = np.clip(b_total * (0.85 + 0.20 * rng.random(m)), 0, b_miss)
        red_miss  = np.clip(r_total * (0.85 + 0.20 * rng.random(m)), 0, r_miss)
        blue_int  = np.clip(D_b, 0, b_int)
        red_int   = np.clip(D_r, 0, r_int)

        idx = np.arange(done, done2)
        keep = (idx % store_every) == 0
        if np.any(keep):
            wlab = np.where(is_draw[keep], "Draw", np.where(is_blue[keep], "Blue", "Red"))
            samples["winner"].extend(wlab.tolist())
            samples["duration"].extend(dur[keep].tolist())
            samples["blue_kia"].extend(blue_kia[keep].tolist())
            samples["red_kia"].extend(red_kia[keep].tolist())
            samples["blue_wia"].extend(blue_wia[keep].tolist())
            samples["red_wia"].extend(red_wia[keep].tolist())
            samples["blue_civ"].extend(blue_civ[keep].tolist())
            samples["red_civ"].extend(red_civ[keep].tolist())
            samples["blue_air"].extend(blue_air[keep].tolist())
            samples["red_air"].extend(red_air[keep].tolist())
            samples["blue_ship"].extend(blue_ship[keep].tolist())
            samples["red_ship"].extend(red_ship[keep].tolist())
            samples["blue_tank"].extend(blue_tank[keep].tolist())
            samples["red_tank"].extend(red_tank[keep].tolist())
            samples["blue_miss"].extend(blue_miss[keep].tolist())
            samples["red_miss"].extend(red_miss[keep].tolist())
            samples["blue_int"].extend(blue_int[keep].tolist())
            samples["red_int"].extend(red_int[keep].tolist())

        done = done2
        if progress_callback is not None:
            payload = None
            try:
                total_so_far = max(1, sum(wins.values()))
                payload = dict(
                    win_prob=dict(Blue=wins["Blue"] / total_so_far, Red=wins["Red"] / total_so_far, Draw=wins["Draw"] / total_so_far),
                    done=int(done),
                    total=int(iters),
                    samples_n=int(len(samples.get("duration", []))),
                    mix=mix,
                )
                for k in ("blue_air", "red_air", "blue_ship", "red_ship", "blue_tank", "red_tank", "blue_kia", "red_kia", "blue_miss", "red_miss", "blue_int", "red_int"):
                    arr = samples.get(k, [])
                    if arr:
                        payload[k] = float(np.mean(np.asarray(arr, dtype=float)))
            except Exception:
                payload = None

            ret = None
            try:
                ret = progress_callback(int(done), int(iters), payload)
            except TypeError:
                # Backward compatible callbacks
                try:
                    ret = progress_callback(int(done), int(iters))
                except Exception:
                    ret = None
            except Exception:
                ret = None

            # Closed-loop control: callback may return a dict of parameter adjustments.
            if isinstance(ret, dict):
                try:
                    # Detection (Blue)
                    det_prob_ctrl = float(np.clip(det_prob_ctrl + float(ret.get('det_prob_delta', 0.0)), 0.10, 0.98))

                    # Target mix: missiles to ships
                    ship_frac_blue_ctrl = float(np.clip(ship_frac_blue_ctrl + float(ret.get('ship_frac_blue_delta', 0.0)), 0.05, 0.95))
                    ship_frac_red_ctrl  = float(np.clip(ship_frac_red_ctrl  + float(ret.get('ship_frac_red_delta',  0.0)), 0.05, 0.95))

                    # Accuracy / interception multipliers
                    pk_mult_blue = float(np.clip(pk_mult_blue * float(ret.get('pk_mult_blue', 1.0)), 0.75, 1.45))
                    pk_mult_red  = float(np.clip(pk_mult_red  * float(ret.get('pk_mult_red',  1.0)), 0.75, 1.45))
                    pd_mult_blue = float(np.clip(pd_mult_blue * float(ret.get('pd_mult_blue', 1.0)), 0.70, 1.60))
                    pd_mult_red  = float(np.clip(pd_mult_red  * float(ret.get('pd_mult_red',  1.0)), 0.70, 1.60))

                    # Tempo multipliers (affects missiles committed to the fight)
                    tempo_mult_blue = float(np.clip(tempo_mult_blue * float(ret.get('tempo_mult_blue', 1.0)), 0.70, 1.60))
                    tempo_mult_red  = float(np.clip(tempo_mult_red  * float(ret.get('tempo_mult_red',  1.0)), 0.70, 1.60))

                    # Strategic edge (affects win probability)
                    ratio_ctrl = float(np.clip(ratio_ctrl * float(ret.get('ratio_ctrl', 1.0)), 0.75, 1.35))

                    # Global escalation / ground contact
                    shock_ctrl = float(np.clip(shock_ctrl * float(ret.get('shock_mult', 1.0)), 0.60, 2.00))
                    land_gate_mult = float(np.clip(land_gate_mult * float(ret.get('land_gate_mult', 1.0)), 0.60, 1.70))
                except Exception:
                    pass

    total = sum(wins.values())
    probs = {k: (wins[k] / total if total else 0.0) for k in wins}

    def pct(arr, p):
        if len(arr) == 0:
            return 0.0
        return float(np.percentile(np.asarray(arr, dtype=float), p))

    def pack(prefix):
        arr = samples[prefix]
        return dict(p10=pct(arr, 10), p50=pct(arr, 50), p90=pct(arr, 90), mean=float(np.mean(np.asarray(arr, dtype=float))) if len(arr) else 0.0)

    return dict(
        iters=int(iters),
        win_prob=probs,
        duration=pack("duration"),
        blue=dict(kia=pack("blue_kia"), wia=pack("blue_wia"), civ=pack("blue_civ"),
                  aircraft=pack("blue_air"), ships=pack("blue_ship"), tanks=pack("blue_tank"),
                  missiles=pack("blue_miss"), interceptors=pack("blue_int")),
        red=dict(kia=pack("red_kia"), wia=pack("red_wia"), civ=pack("red_civ"),
                 aircraft=pack("red_air"), ships=pack("red_ship"), tanks=pack("red_tank"),
                 missiles=pack("red_miss"), interceptors=pack("red_int")),
        meta=dict(
            ratio=float(ratio),
            commit=dict(blue=float(commit_blue), red=float(commit_red)),
            theatre=theatre,
            mix=mix,
            local_side=local_side,
        )
    )



# ============================================================
# Integrated 4-engine campaign model (Attrition + Salvo + Logistics + Fog of War)
# ============================================================
# This is an ABSTRACT educational simulator, not a predictive model.
# Do not use for real-world planning.


def _ic_clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))

def _ic_lanchester_square_step(blue_units: float, red_units: float, k_blue: float, k_red: float, dt_days: float = 1.0) -> Tuple[float, float]:
    """
    Higher-fidelity integration of Lanchester Square Law using RK4 on the 2D system:

        dB/dt = -k_red * R
        dR/dt = -k_blue * B

    RK4 reduces timestep artifacts (overshoot/clamp) and behaves better under high tempo.
    """
    B0 = max(0.0, float(blue_units))
    R0 = max(0.0, float(red_units))
    kb = max(0.0, float(k_blue))
    kr = max(0.0, float(k_red))
    dt = float(max(0.0, dt_days))

    def fB(B, R): return -kr * max(0.0, R)
    def fR(B, R): return -kb * max(0.0, B)

    k1B, k1R = fB(B0, R0), fR(B0, R0)
    k2B, k2R = fB(B0 + 0.5*dt*k1B, R0 + 0.5*dt*k1R), fR(B0 + 0.5*dt*k1B, R0 + 0.5*dt*k1R)
    k3B, k3R = fB(B0 + 0.5*dt*k2B, R0 + 0.5*dt*k2R), fR(B0 + 0.5*dt*k2B, R0 + 0.5*dt*k2R)
    k4B, k4R = fB(B0 + dt*k3B, R0 + dt*k3R), fR(B0 + dt*k3B, R0 + dt*k3R)

    B1 = B0 + (dt/6.0) * (k1B + 2*k2B + 2*k3B + k4B)
    R1 = R0 + (dt/6.0) * (k1R + 2*k2R + 2*k3R + k4R)

    return max(0.0, B1), max(0.0, R1)

def _ic_hughes_stochastic_hits(rng: np.random.Generator, M: int, Pk: float, D: int, Pd: float) -> Tuple[int, int]:
    """
    Saturation variant:
      - up to min(M, D) intercept attempts; each attempt consumes 1 interceptor
      - successful intercept removes a missile
      - remaining missiles hit with probability Pk
    Returns: (hits, interceptors_spent)
    """
    M = int(max(0, M)); D = int(max(0, D))
    Pk = _ic_clip01(Pk); Pd = _ic_clip01(Pd)
    if M == 0:
        return 0, 0
    intercept_trials = min(M, D)
    intercept_success = int(rng.binomial(intercept_trials, Pd))
    missiles_through = max(0, M - intercept_success)
    hits = int(rng.binomial(missiles_through, Pk))
    return hits, intercept_trials

def _ic_adjudicate_roll_hits(rng: np.random.Generator, missiles: int, defense_rating_0_100: float) -> int:
    missiles = int(max(0, missiles))
    if missiles == 0:
        return 0
    d = float(np.clip(defense_rating_0_100, 0.0, 100.0))
    rolls = rng.integers(1, 101, size=missiles)
    return int(np.sum(rolls > d))

def _ic_supportable_battalions(port_capacity_tons_per_day: float, integrity_coeff_0_1: float, consumption_tons_per_battalion_day: float, lots_capacity_tons_per_day: float = 0.0) -> float:
    port_capacity_tons_per_day = max(0.0, float(port_capacity_tons_per_day))
    integrity_coeff_0_1 = _ic_clip01(integrity_coeff_0_1)
    consumption_tons_per_battalion_day = max(1e-9, float(consumption_tons_per_battalion_day))
    lots_capacity_tons_per_day = max(0.0, float(lots_capacity_tons_per_day))
    effective = (port_capacity_tons_per_day * integrity_coeff_0_1) + lots_capacity_tons_per_day
    return effective / consumption_tons_per_battalion_day

@dataclass
class IC_ShipProfile:
    name: str
    staying_power_W: int = 1
    defense_rating_0_100: float = 60.0
    interceptors_D_per_ship: int = 10
    intercept_success_Pd: float = 0.25

@dataclass
class IC_FleetState:
    ships: Dict[str, int]
    damage_bank_hits: Dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return int(sum(max(0, v) for v in self.ships.values()))

    def clamp(self) -> None:
        for k, v in list(self.ships.items()):
            self.ships[k] = int(max(0, v))
        for k, v in list(self.damage_bank_hits.items()):
            self.damage_bank_hits[k] = int(max(0, v))

@dataclass
class IC_SideState:
    name: str
    aircraft: float
    tanks: float
    artillery: float
    ground_battalions_committed: float

    fleet: IC_FleetState

    missiles_stock: float
    interceptors_stock: float

    missiles_per_day: float
    missile_hit_prob_Pk: float

    # Missile add-on metrics (0..100). These tilt launch tempo / Pd / Pk.
    missile_density_0_100: float = 50.0
    missile_saturation_0_100: float = 50.0
    missile_intercept_0_100: float = 50.0
    missile_breakthrough_0_100: float = 50.0
    missile_shield_rank: float = 25.0
    missile_tier: float = 3.0

    transport_ships_key: str = "transport"
    battalions_per_transport_ship: float = 1.0
    lots_capacity_tons_per_day: float = 0.0

    ground_effectiveness_0_1: float = 1.0

    def clamp(self) -> None:
        self.aircraft = max(0.0, float(self.aircraft))
        self.tanks = max(0.0, float(self.tanks))
        self.artillery = max(0.0, float(self.artillery))
        self.ground_battalions_committed = max(0.0, float(self.ground_battalions_committed))
        self.missiles_stock = max(0.0, float(self.missiles_stock))
        self.interceptors_stock = max(0.0, float(self.interceptors_stock))
        self.missiles_per_day = max(0.0, float(self.missiles_per_day))
        self.missile_hit_prob_Pk = _ic_clip01(self.missile_hit_prob_Pk)
        self.missile_density_0_100 = float(np.clip(self.missile_density_0_100, 0.0, 100.0))
        self.missile_saturation_0_100 = float(np.clip(self.missile_saturation_0_100, 0.0, 100.0))
        self.missile_intercept_0_100 = float(np.clip(self.missile_intercept_0_100, 0.0, 100.0))
        self.missile_breakthrough_0_100 = float(np.clip(self.missile_breakthrough_0_100, 0.0, 100.0))
        self.missile_tier = float(np.clip(self.missile_tier, 1.0, 10.0))
        self.lots_capacity_tons_per_day = max(0.0, float(self.lots_capacity_tons_per_day))
        self.ground_effectiveness_0_1 = _ic_clip01(self.ground_effectiveness_0_1)
        self.fleet.clamp()

@dataclass
class IC_CampaignConfig:
    days: int = 24
    seed: int = 7

    # Detection
    blue_detection_prob: float = 0.65
    pk_bonus_if_detected: float = 0.05
    pd_bonus_if_detected: float = 0.10

    # Air (Lanchester)
    k_blue_air: float = 0.002
    k_red_air: float = 0.0015

    # Land (Lanchester)
    k_blue_land: float = 0.0010
    k_red_land: float = 0.0010

    # Logistics / Scuttle
    port_capacity_tons_per_day: float = 50_000.0
    consumption_tons_per_battalion_day: float = 300.0
    port_integrity_intact: float = 1.0
    port_integrity_scuttled: float = 0.1
    scuttle_day: int = 2
    daily_effectiveness_loss_if_unsupplied: float = 0.10

    # Ground power / victory
    combat_power_per_battalion: float = 1.0
    blue_defense_multiplier: float = 1.0

    # Missile target mix
    missile_fraction_to_ships: float = 0.65  # rest goes to "airbase/ground" strikes
    use_roll_based_hits: bool = False

@dataclass
class IC_DayLog:
    day: int
    detected: bool

    blue_aircraft: float
    red_aircraft: float

    blue_ships_total: int
    red_ships_total: int

    blue_tanks: float
    red_tanks: float
    blue_artillery: float
    red_artillery: float

    red_transports: int
    landed_battalions: float
    supportable_battalions: float
    red_ground_effectiveness: float

    missiles_blue_left: float
    missiles_red_left: float
    interceptors_blue_left: float
    interceptors_red_left: float

    red_ground_power: float
    blue_ground_power: float

    winner: Optional[str] = None
    end_reason: str = ""
    notes: List[str] = field(default_factory=list)

# -----------------------------
# Integrated "Orders" layer (stateful intent)
# -----------------------------
@dataclass
class IC_Order:
    name: str
    k_air_mult: float = 1.0
    k_land_mult: float = 1.0
    missile_day_mult: float = 1.0
    missile_ship_frac_delta: float = 0.0
    detection_mult: float = 1.0
    defense_pd_bonus: float = 0.0
    port_strike_bias: float = 0.0  # pushes missiles to land/ports

def _ic_shield_quality(rank: float) -> float:
    """Convert shield rank (1 best .. 60 worst) -> 0..1 quality."""
    try:
        r = float(rank)
    except Exception:
        r = 25.0
    r = float(np.clip(r, 1.0, 60.0))
    return float(np.clip(1.0 - (r - 1.0) / 59.0, 0.0, 1.0))

def _ic_choose_order(
    rng: np.random.Generator,
    *,
    side: "IC_SideState",
    enemy: "IC_SideState",
    day: int,
    scenario_type: str,
    aggression_0_1: float,
) -> IC_Order:
    """
    Lightweight intent model to make the campaign feel like a duel of decisions.
    Not optimization, just a reactive policy.
    """
    stype = (scenario_type or "General").strip().lower()
    ag = float(np.clip(aggression_0_1, 0.0, 1.0))

    ship_gap = (enemy.fleet.total() + 1) / max(1.0, side.fleet.total() + 1)
    air_gap = (enemy.aircraft + 1.0) / max(1.0, side.aircraft + 1.0)
    low_missiles = bool(side.missiles_stock < 0.20 * max(1.0, side.missiles_stock + enemy.missiles_stock))
    early = bool(day <= 5)

    orders: List[IC_Order] = [
        IC_Order("Hold & Shield", k_air_mult=0.95, k_land_mult=0.95, missile_day_mult=0.80, detection_mult=1.10, defense_pd_bonus=0.05),
        IC_Order("Fleet Hunt", k_air_mult=1.00, k_land_mult=0.95, missile_day_mult=1.10, missile_ship_frac_delta=0.18, detection_mult=1.05),
        IC_Order("Air Supremacy", k_air_mult=1.20, k_land_mult=0.95, missile_day_mult=1.00, missile_ship_frac_delta=-0.12, detection_mult=1.10),
        IC_Order("Port Denial", k_air_mult=1.00, k_land_mult=1.00, missile_day_mult=1.05, missile_ship_frac_delta=-0.10, port_strike_bias=0.18),
        IC_Order("Shock & Awe", k_air_mult=1.10, k_land_mult=1.10, missile_day_mult=1.25, missile_ship_frac_delta=0.08, detection_mult=0.95, port_strike_bias=0.10),
    ]

    scores: List[float] = []
    for o in orders:
        s = 0.0
        if ("blockade" in stype) or ("port" in stype):
            s += 0.9 * o.port_strike_bias + 0.35 * max(0.0, -o.missile_ship_frac_delta)
        if ("amph" in stype) or ("invasion" in stype) or ("island" in stype):
            s += 0.6 * o.port_strike_bias + 0.35 * (o.missile_ship_frac_delta + 0.10)
        if ("air" in stype) and (o.name == "Air Supremacy"):
            s += 0.6
        if ("land" in stype) and (o.name == "Shock & Awe"):
            s += 0.5

        if ship_gap > 1.10 and o.name == "Fleet Hunt":
            s += 0.6 * (ship_gap - 1.0)
        if air_gap > 1.10 and o.name == "Air Supremacy":
            s += 0.6 * (air_gap - 1.0)

        if o.name in ("Shock & Awe", "Fleet Hunt", "Air Supremacy"):
            s += 0.55 * ag
        if o.name == "Hold & Shield":
            s += 0.35 * (1.0 - ag)

        if low_missiles and o.missile_day_mult > 1.05:
            s -= 0.55
        if early and o.name == "Shock & Awe":
            s += 0.35

        s += float(rng.normal(0.0, 0.10))
        scores.append(s)

    return orders[int(np.argmax(scores))]

def _ic_apply_salvo(
    *,
    rng: np.random.Generator,
    attacker: IC_SideState,
    defender: IC_SideState,
    ship_db: Dict[str, IC_ShipProfile],
    missiles_to_ships: int,
    attacker_Pk: float,
    defender_Pd_bonus: float,
    use_roll_based_hits: bool,
    notes: List[str],
) -> Tuple[int, int, int]:
    """
    Returns: (ships_sunk_total, missiles_spent, interceptors_spent)
    """
    missiles_to_ships = int(max(0, missiles_to_ships))
    if missiles_to_ships == 0 or defender.fleet.total() == 0:
        return 0, 0, 0

    ship_types = [t for t, c in defender.fleet.ships.items() if c > 0 and t in ship_db]
    if not ship_types:
        return 0, 0, 0

    counts = np.array([defender.fleet.ships[t] for t in ship_types], dtype=float)
    weights = counts / max(1e-9, counts.sum())
    allocation = rng.multinomial(missiles_to_ships, weights)

    ships_sunk = 0
    missiles_spent = 0
    interceptors_spent = 0

    for t, M_t in zip(ship_types, allocation):
        if M_t <= 0:
            continue
        prof = ship_db[t]
        ship_count = int(defender.fleet.ships.get(t, 0))
        if ship_count <= 0:
            continue

        # Interceptors available this salvo are limited by remaining interceptor stock
        D_theory = int(max(0, prof.interceptors_D_per_ship) * ship_count)
        D_avail = int(min(D_theory, defender.interceptors_stock))
        Pd = _ic_clip01(prof.intercept_success_Pd + defender_Pd_bonus)

        if use_roll_based_hits:
            hits = _ic_adjudicate_roll_hits(rng, int(M_t), prof.defense_rating_0_100)
            ints_used = 0
        else:
            hits, ints_used = _ic_hughes_stochastic_hits(rng, int(M_t), attacker_Pk, D_avail, Pd)

        # Spend stocks
        missiles_spent += int(M_t)
        attacker.missiles_stock = max(0.0, attacker.missiles_stock - float(M_t))
        interceptors_spent += int(ints_used)
        defender.interceptors_stock = max(0.0, defender.interceptors_stock - float(ints_used))

        # Hits -> sinks via staying power + damage bank
        bank = int(defender.fleet.damage_bank_hits.get(t, 0)) + int(hits)
        W = int(max(1, prof.staying_power_W))
        newly_sunk = min(ship_count, bank // W)
        bank -= newly_sunk * W

        if newly_sunk > 0:
            defender.fleet.ships[t] = ship_count - newly_sunk
            ships_sunk += newly_sunk

        defender.fleet.damage_bank_hits[t] = bank

    if ships_sunk > 0:
        notes.append(f"{attacker.name} salvo sank ~{ships_sunk} {defender.name} ship(s).")

    return ships_sunk, missiles_spent, interceptors_spent

def _ic_airbase_strikes(
    rng: np.random.Generator,
    *,
    attacker: IC_SideState,
    defender: IC_SideState,
    missiles_to_land: int,
    attacker_Pk: float,
    defender_defense_rating_0_100: float,
    notes: List[str],
) -> int:
    """
    Convert missiles aimed at airbases/ground into aircraft losses (toy logic).
    """
    missiles_to_land = int(max(0, missiles_to_land))
    if missiles_to_land <= 0 or defender.aircraft <= 0:
        return 0
    # Use a roll-based hit gate that depends on defender rating, then scale to aircraft killed.
    hits = _ic_adjudicate_roll_hits(rng, missiles_to_land, defender_defense_rating_0_100)
    # Each "hit" destroys a fraction of parked aircraft (clipped).
    kill_frac = float(np.clip(0.0025 * hits, 0.0, 0.20))
    lost = int(round(defender.aircraft * kill_frac))
    if lost > 0:
        defender.aircraft = max(0.0, defender.aircraft - lost)
        attacker.missiles_stock = max(0.0, attacker.missiles_stock - missiles_to_land)
        notes.append(f"{attacker.name} airbase strike: ~{lost} aircraft destroyed (kill_frac={kill_frac:.2f}).")
    return lost

def _ic_default_ship_db() -> Dict[str, IC_ShipProfile]:
    # Default profiles: conservative "staying power" and defense.
    return {
        "carrier":   IC_ShipProfile("carrier",   staying_power_W=4, defense_rating_0_100=85, interceptors_D_per_ship=30, intercept_success_Pd=0.35),
        "surface":   IC_ShipProfile("surface",   staying_power_W=1, defense_rating_0_100=70, interceptors_D_per_ship=18, intercept_success_Pd=0.30),
        "submarine": IC_ShipProfile("submarine", staying_power_W=1, defense_rating_0_100=65, interceptors_D_per_ship=6,  intercept_success_Pd=0.25),
        "transport": IC_ShipProfile("transport", staying_power_W=1, defense_rating_0_100=40, interceptors_D_per_ship=2,  intercept_success_Pd=0.10),
    }

def _ic_campaign_from_bundle(bundle: DataBundle, sc: Scenario) -> Tuple[IC_SideState, IC_SideState, Dict[str, IC_ShipProfile], IC_CampaignConfig, str]:
    """
    Build a default integrated campaign from the existing DataBundle + Scenario.
    Keeps *all data calling intact*: we only read from bundle.countries and existing score columns.
    """
    df = bundle.countries
    bg = df[df["ISO3"] == sc.battleground_iso3]
    bg_row = bg.iloc[0].to_dict() if len(bg) else {}

    blue_iso = sc.blue if sc.blue else []
    red_iso  = sc.red if sc.red else []
    blue_ag = _side_aggregate(df, blue_iso, sc.sanctions_blue, sc.neutral_support_blue) if blue_iso else _side_aggregate(df, ["USA"], sc.sanctions_blue, sc.neutral_support_blue)
    red_ag  = _side_aggregate(df, red_iso,  sc.sanctions_red,  sc.neutral_support_red)  if red_iso  else _side_aggregate(df, ["CHN"], sc.sanctions_red,  sc.neutral_support_red)

    theatre = _theatre_modifiers(bg_row)
    # Integrated model uses the same intensity control as the rest of the app.
    intensity = _intensity_params(sc.intensity if sc.intensity is not None else "Campaign")
    days = int(sc.max_days or 28)

    # Determine local side for invasion logic
    if sc.battleground_iso3 in blue_iso:
        local_side = "blue"
    elif sc.battleground_iso3 in red_iso:
        local_side = "red"
    else:
        local_side = "none"

    # Heuristic: "invader" is the opposite of local side (if known); default red.
    invader_is_red = True
    if local_side == "blue":
        invader_is_red = True
    elif local_side == "red":
        invader_is_red = False

    def pick_air(side: dict) -> float:
        # prefer combat_aircraft if present, else 35% of total aircraft
        ca = float(side.get("combat_aircraft", 0.0) or 0.0)
        tot = float(side.get("aircraft", 0.0) or 0.0)
        base = ca if ca > 0 else 0.35 * tot
        # readiness proxy: military_capability_score if present
        mc = float(side.get("military_capability_score", side.get("mc_score", 50.0)) or 50.0)
        return base * (0.55 + 0.45 * (mc / 100.0))

    def pick_land(side: dict) -> Tuple[float, float, float]:
        tanks = float(side.get("tanks", 0.0) or 0.0)
        arty  = float(side.get("artillery", 0.0) or 0.0)
        active = float(side.get("active", 0.0) or 0.0)
        # battalions from active personnel (rough)
        battalions = max(0.0, active / 750.0)
        return tanks, arty, battalions

    def pick_fleet(side: dict, is_invader: bool) -> Dict[str, int]:
        carriers = int(round(float(side.get("carriers", 0.0) or 0.0)))
        # "major ships" prefer Units_Tracked, else a conservative fraction of naval_assets_total
        major = float(side.get("Units_Tracked", 0.0) or 0.0)
        if major <= 0:
            major = float(side.get("naval", 0.0) or 0.0)  # already a conservative major-unit proxy in _side_aggregate
        major = int(round(max(0.0, major)))
        # subs (if unknown, assume 15% of major)
        subs = int(round(max(0.0, float(side.get("submarines", 0.0) or 0.0))))
        if subs <= 0:
            subs = int(round(0.15 * major))
        # transports: invader needs them; otherwise small number
        if is_invader:
            transports = int(round(max(8.0, 0.25 * major)))
        else:
            transports = int(round(max(2.0, 0.05 * major)))
        surface = max(0, major - carriers - subs - transports)
        return {"carrier": carriers, "submarine": subs, "surface": surface, "transport": transports}

    # Sides
    blue_is_invader = (not invader_is_red)
    red_is_invader  = invader_is_red

    b_tanks, b_arty, b_bats = pick_land(blue_ag)
    r_tanks, r_arty, r_bats = pick_land(red_ag)

    # committed fractions (global knob in integrated_cfg; safe defaults)
    icfg = (getattr(sc, "integrated_cfg", None) or {}) if sc is not None else {}
    commit_mult = float(icfg.get("commit_mult", 1.0))
    default_blue = 0.35
    default_red = 0.45 if red_is_invader else 0.35
    commit_blue = float(icfg.get("commit_blue", default_blue * commit_mult))
    commit_red  = float(icfg.get("commit_red",  default_red  * commit_mult))
    commit_blue = float(np.clip(commit_blue, 0.0, 1.0))
    commit_red  = float(np.clip(commit_red,  0.0, 1.0))

    
    # missiles / interceptors (stocks are already sanctions-adjusted in _side_aggregate)
    b_miss = float(blue_ag.get("missiles", 0.0) or 0.0)
    r_miss = float(red_ag.get("missiles", 0.0) or 0.0)
    b_int  = float(blue_ag.get("interceptors", 0.0) or 0.0)
    r_int  = float(red_ag.get("interceptors", 0.0) or 0.0)

    # Add-on missile metrics (0..100). Defaults are mid-values if the add-on file isn't present.
    b_den = float(blue_ag.get("missile_density", 50.0) or 50.0)
    b_sat = float(blue_ag.get("missile_saturation", 50.0) or 50.0)
    b_intq = float(blue_ag.get("missile_intercept", 50.0) or 50.0)
    b_br = float(blue_ag.get("missile_breakthrough", 50.0) or 50.0)
    b_tier = float(blue_ag.get("missile_tier", 3.0) or 3.0)

    r_den = float(red_ag.get("missile_density", 50.0) or 50.0)
    r_sat = float(red_ag.get("missile_saturation", 50.0) or 50.0)
    r_intq = float(red_ag.get("missile_intercept", 50.0) or 50.0)
    r_br = float(red_ag.get("missile_breakthrough", 50.0) or 50.0)
    r_tier = float(red_ag.get("missile_tier", 3.0) or 3.0)

    # Missiles/day scales with (a) intensity missile_rate and (b) saturation/density/tier,
    # then capped by remaining stock.
    miss_rate = float(intensity.get("missile_rate", 0.022))
    base_scale = float(np.clip(0.35 + 20.0 * miss_rate, 0.45, 1.25))

    b_scale = base_scale * (0.70 + 0.006 * b_sat) * (0.85 + 0.003 * (b_den - 50.0)) * (0.92 + 0.03 * (4.0 - min(4.0, b_tier)))
    r_scale = base_scale * (0.70 + 0.006 * r_sat) * (0.85 + 0.003 * (r_den - 50.0)) * (0.92 + 0.03 * (4.0 - min(4.0, r_tier)))

    b_mpd = min(b_miss, max(0.0, (b_miss / max(1.0, days)) * b_scale))
    r_mpd = min(r_miss, max(0.0, (r_miss / max(1.0, days)) * r_scale))

    # Hit probability from networks + future resources + breakthrough (bounded)
    b_net = float(blue_ag.get("networks_score", 50.0) or 50.0) / 100.0
    b_future = float(blue_ag.get("future_resources_score", 50.0) or 50.0) / 100.0
    r_net = float(red_ag.get("networks_score", 50.0) or 50.0) / 100.0
    r_future = float(red_ag.get("future_resources_score", 50.0) or 50.0) / 100.0

    b_pk = float(np.clip(0.12 + 0.14 * b_net + 0.10 * b_future + 0.14 * (b_br / 100.0), 0.10, 0.48))
    r_pk = float(np.clip(0.12 + 0.14 * r_net + 0.10 * r_future + 0.14 * (r_br / 100.0), 0.10, 0.48))

# LOTS proxy from logistics+transport score
    bg_transport = float(bg_row.get("transport_score_0_100", bg_row.get("transport_score", 50.0)) or 50.0)
    r_lots = max(0.0, (float(red_ag.get("logistics_score", 50.0) or 50.0) / 100.0) * 0.25 * 60_000.0)

    blue = IC_SideState(
        name="Blue",
        aircraft=pick_air(blue_ag) * commit_blue * float(theatre.get("air", 1.0)),
        tanks=b_tanks * commit_blue * float(theatre.get("land", 1.0)),
        artillery=b_arty * commit_blue * float(theatre.get("land", 1.0)),
        ground_battalions_committed=b_bats * float(theatre.get("land", 1.0)),
        fleet=IC_FleetState(ships=pick_fleet(blue_ag, blue_is_invader)),
        missiles_stock=b_miss * commit_blue,
        interceptors_stock=b_int * commit_blue,
        missiles_per_day=b_mpd * commit_blue,
        missile_hit_prob_Pk=b_pk,
        missile_density_0_100=b_den,
        missile_saturation_0_100=b_sat,
        missile_intercept_0_100=b_intq,
        missile_breakthrough_0_100=b_br,
        missile_shield_rank=float(blue_ag.get("missile_shield_rank", 25.0) or 25.0),
        missile_tier=b_tier,
        battalions_per_transport_ship=1.0,
        lots_capacity_tons_per_day=0.0,
    )

    red = IC_SideState(
        name="Red",
        aircraft=pick_air(red_ag) * commit_red * float(theatre.get("air", 1.0)),
        tanks=r_tanks * commit_red * float(theatre.get("land", 1.0)),
        artillery=r_arty * commit_red * float(theatre.get("land", 1.0)),
        ground_battalions_committed=r_bats * float(theatre.get("land", 1.0)) * (0.65 if red_is_invader else 0.35),
        fleet=IC_FleetState(ships=pick_fleet(red_ag, red_is_invader)),
        missiles_stock=r_miss * commit_red,
        interceptors_stock=r_int * commit_red,
        missiles_per_day=r_mpd * commit_red,
        missile_hit_prob_Pk=r_pk,
        missile_density_0_100=r_den,
        missile_saturation_0_100=r_sat,
        missile_intercept_0_100=r_intq,
        missile_breakthrough_0_100=r_br,
        missile_shield_rank=float(red_ag.get("missile_shield_rank", 25.0) or 25.0),
        missile_tier=r_tier,
        battalions_per_transport_ship=float(np.clip(0.9 + 0.006 * (float(red_ag.get("logistics_score", 50.0) or 50.0) - 50.0), 0.7, 1.4)),
        lots_capacity_tons_per_day=r_lots,
    )

    # Campaign knobs (defaults, then override from sc.integrated_cfg if supplied)
    det_base = 0.55 + 0.003 * (float(blue_ag.get("dn_score", 50.0) or 50.0) - float(red_ag.get("dn_score", 50.0) or 50.0))
    det_base = float(np.clip(det_base, 0.35, 0.80))

    # Port capacity from battleground transport score
    port_capacity = 25_000.0 + 700.0 * float(bg_transport)

    cfg = IC_CampaignConfig(
        days=days,
        seed=int(getattr(sc, "seed", 7) or 7),
        blue_detection_prob=det_base,
        k_blue_air=float(np.clip(0.0015 + 0.00002 * float(blue_ag.get("mc_score", 50.0) or 50.0), 0.001, 0.004)),
        k_red_air=float(np.clip(0.0015 + 0.00002 * float(red_ag.get("mc_score", 50.0) or 50.0), 0.001, 0.004)),
        k_blue_land=float(np.clip(0.0008 + 0.000015 * float(blue_ag.get("mc_score", 50.0) or 50.0), 0.0005, 0.003)),
        k_red_land=float(np.clip(0.0008 + 0.000015 * float(red_ag.get("mc_score", 50.0) or 50.0), 0.0005, 0.003)),
        port_capacity_tons_per_day=port_capacity,
        consumption_tons_per_battalion_day=float(np.clip(260.0 + 1.0 * float(intensity.get("land", 1.0)) * 60.0, 200.0, 450.0)),
        scuttle_day=2 if invader_is_red else 9999,
        blue_defense_multiplier=float(np.clip(0.85 + 0.005 * float(bg_row.get("terrain_score_0_100", bg_row.get("terrain_score", 50.0)) or 50.0) / 2.0, 0.9, 1.5)),
        missile_fraction_to_ships=float(np.clip(0.55 + 0.10 * float(intensity.get("sea", 1.0)) - 0.05 * float(theatre.get("littoral", 0.0)), 0.40, 0.80)),
        use_roll_based_hits=False,
    )

    # Overrides from scenario (kept as dict so app serialization stays simple)
    if getattr(sc, "integrated_cfg", None):
        ic = dict(sc.integrated_cfg)
        for k, v in ic.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception:
                    setattr(cfg, k, v)

    ship_db = _ic_default_ship_db()
    return blue, red, ship_db, cfg, local_side

def simulate_integrated(bundle: DataBundle, sc: Scenario) -> Tuple[dict, List[dict]]:
    """
    Wrapper that returns the same (summary, series) shape as simulate_cinematic(),
    so the Streamlit app doesn't need to change its data plumbing.
    """
    blue, red, ship_db, cfg, local_side = _ic_campaign_from_bundle(bundle, sc)
    rng = np.random.default_rng(cfg.seed)

    # Snapshot initial
    init = {
        "blue": {"aircraft": blue.aircraft, "ships": blue.fleet.total(), "tanks": blue.tanks, "artillery": blue.artillery,
                 "missiles": blue.missiles_stock, "interceptors": blue.interceptors_stock},
        "red":  {"aircraft": red.aircraft,  "ships": red.fleet.total(),  "tanks": red.tanks,  "artillery": red.artillery,
                 "missiles": red.missiles_stock, "interceptors": red.interceptors_stock},
    }

    series: List[dict] = []
    winner = None
    end_reason = ""

    # Cumulative personnel casualty trackers (used in series for Battle Losses 'Today' consistency)
    cum_kia_blue = 0.0
    cum_wia_blue = 0.0
    cum_kia_red  = 0.0
    cum_wia_red  = 0.0

    # Simple will model: starts at 100, decays with proportional daily losses
    will_blue = 100.0
    will_red  = 100.0

    # Orders config
    icfg = (getattr(sc, "integrated_cfg", None) or {}) if sc is not None else {}
    df = bundle.countries
    scenario_type = str(getattr(sc, "scenario_type", None) or icfg.get("scenario_type") or "General")
    # Job-Done add-on state for Integrated model
    jd = {
        "base_id": _jd_base_intensity(sc) if sc is not None else 2.5,
        "scenario_type": scenario_type,
        "blue": _jd_init_side(df, getattr(sc, "blue", []) or [], getattr(sc, "neutral_support_blue", []) or []),
        "red": _jd_init_side(df, getattr(sc, "red", []) or [], getattr(sc, "neutral_support_red", []) or []),
    }
    enable_orders = bool(icfg.get("enable_orders", True))
    ag_blue = float(np.clip(icfg.get("aggression_blue", 0.55), 0.0, 1.0))
    ag_red  = float(np.clip(icfg.get("aggression_red",  0.60), 0.0, 1.0))
    port_shock = 0.0


    for day in range(1, cfg.days + 1):
        blue.clamp(); red.clamp()
        notes: List[str] = []

        # 1) Orders (optional) + Detection (Blue detects Red)
        order_blue = None
        order_red = None
        det_prob = float(np.clip(cfg.blue_detection_prob, 0.0, 1.0))
        if enable_orders:
            order_blue = _ic_choose_order(rng, side=blue, enemy=red, day=day, scenario_type=scenario_type, aggression_0_1=ag_blue)
            order_red  = _ic_choose_order(rng, side=red,  enemy=blue, day=day, scenario_type=scenario_type, aggression_0_1=ag_red)
            notes.append(f"Blue order: {order_blue.name}")
            notes.append(f"Red order: {order_red.name}")
            det_prob = float(np.clip(det_prob * order_blue.detection_mult * (1.0 - 0.08 * (order_red.name == "Hold & Shield")), 0.15, 0.95))
        detected = bool(rng.random() < det_prob)

        
        # cueing bonuses (detection) + missile defense quality (add-on dataset)
        blue_intq = float(np.clip(blue.missile_intercept_0_100 / 100.0, 0.0, 1.0))
        red_intq  = float(np.clip(red.missile_intercept_0_100 / 100.0, 0.0, 1.0))
        blue_brq  = float(np.clip(blue.missile_breakthrough_0_100 / 100.0, 0.0, 1.0))
        red_brq   = float(np.clip(red.missile_breakthrough_0_100 / 100.0, 0.0, 1.0))

        blue_shq = _ic_shield_quality(getattr(blue, "missile_shield_rank", 25.0))
        red_shq  = _ic_shield_quality(getattr(red,  "missile_shield_rank", 25.0))

        base_pd_blue = 0.07 * (blue_intq - 0.5) + 0.06 * (blue_shq - 0.5)
        base_pd_red  = 0.07 * (red_intq  - 0.5) + 0.06 * (red_shq  - 0.5)

        if enable_orders and order_blue is not None:
            base_pd_blue += float(order_blue.defense_pd_bonus)
        if enable_orders and order_red is not None:
            base_pd_red  += float(order_red.defense_pd_bonus)

        if detected:
            blue_pk = float(np.clip(blue.missile_hit_prob_Pk + cfg.pk_bonus_if_detected * (0.75 + 0.50 * blue_brq), 0.0, 1.0))
            blue_pd_bonus = float(cfg.pd_bonus_if_detected + base_pd_blue)
            red_pk = float(np.clip(red.missile_hit_prob_Pk, 0.0, 1.0))
            red_pd_bonus = float(base_pd_red)
        else:
            blue_pk = float(np.clip(blue.missile_hit_prob_Pk, 0.0, 1.0))
            blue_pd_bonus = float(base_pd_blue)
            red_pk = float(np.clip(red.missile_hit_prob_Pk + cfg.pk_bonus_if_detected * (0.75 + 0.50 * red_brq), 0.0, 1.0))
            red_pd_bonus = float(cfg.pd_bonus_if_detected + base_pd_red)

        # 2) Missile Salvos (pulsed)
        mdm_b = float(order_blue.missile_day_mult) if (enable_orders and order_blue is not None) else 1.0
        mdm_r = float(order_red.missile_day_mult)  if (enable_orders and order_red  is not None) else 1.0
        m_blue = int(min(blue.missiles_stock, max(0.0, blue.missiles_per_day * mdm_b)))
        m_red  = int(min(red.missiles_stock,  max(0.0, red.missiles_per_day  * mdm_r)))

        ship_frac_b = float(np.clip(cfg.missile_fraction_to_ships + (float(order_blue.missile_ship_frac_delta) if (enable_orders and order_blue is not None) else 0.0), 0.22, 0.90))
        ship_frac_r = float(np.clip(cfg.missile_fraction_to_ships + (float(order_red.missile_ship_frac_delta)  if (enable_orders and order_red  is not None) else 0.0), 0.22, 0.90))

        b_to_ships = int(round(ship_frac_b * m_blue))
        r_to_ships = int(round(ship_frac_r * m_red))
        b_to_land = max(0, m_blue - b_to_ships)
        r_to_land = max(0, m_red - r_to_ships)

        # ship salvos
        sunk_b, spent_m_red, spent_i_blue = _ic_apply_salvo(
            rng=rng, attacker=red, defender=blue, ship_db=ship_db, missiles_to_ships=r_to_ships,
            attacker_Pk=red_pk, defender_Pd_bonus=blue_pd_bonus, use_roll_based_hits=cfg.use_roll_based_hits, notes=notes
        )
        sunk_r, spent_m_blue, spent_i_red = _ic_apply_salvo(
            rng=rng, attacker=blue, defender=red, ship_db=ship_db, missiles_to_ships=b_to_ships,
            attacker_Pk=blue_pk, defender_Pd_bonus=red_pd_bonus, use_roll_based_hits=cfg.use_roll_based_hits, notes=notes
        )

        # land/airbase strikes (only meaningful early, but allowed)
        # Defense rating proxy: defender networks + resilience
        def_rating_blue = float(np.clip(55.0 + 0.25 * (init["blue"]["interceptors"] / max(1.0, init["blue"]["ships"]+1)) , 35.0, 90.0))
        def_rating_red  = float(np.clip(55.0 + 0.25 * (init["red"]["interceptors"] / max(1.0, init["red"]["ships"]+1)) , 35.0, 90.0))
        air_kill_b = _ic_airbase_strikes(rng, attacker=red, defender=blue, missiles_to_land=r_to_land, attacker_Pk=red_pk, defender_defense_rating_0_100=def_rating_blue, notes=notes)
        air_kill_r = _ic_airbase_strikes(rng, attacker=blue, defender=red, missiles_to_land=b_to_land, attacker_Pk=blue_pk, defender_defense_rating_0_100=def_rating_red, notes=notes)

        # 3) Air Attrition (continuous daily, order-aware)
        kB_air = float(cfg.k_blue_air) * (float(order_blue.k_air_mult) if (enable_orders and order_blue is not None) else 1.0)
        kR_air = float(cfg.k_red_air)  * (float(order_red.k_air_mult)  if (enable_orders and order_red  is not None) else 1.0)
        blue.aircraft, red.aircraft = _ic_lanchester_square_step(
            blue_units=blue.aircraft, red_units=red.aircraft, k_blue=kB_air, k_red=kR_air, dt_days=1.0
        )

        # 4) Amphibious Check: transports surviving -> battalions landed (assume Red is invader when battleground local_side==blue/none)
        red_transports = int(red.fleet.ships.get(red.transport_ships_key, 0))
        landed_battalions = min(red.ground_battalions_committed, red_transports * red.battalions_per_transport_ship)

        # 5) Throughput / Scuttle (+ port strike shock)
        port_integrity = float(cfg.port_integrity_intact)
        if 1 <= int(cfg.scuttle_day) <= cfg.days and day >= int(cfg.scuttle_day):
            port_integrity = float(cfg.port_integrity_scuttled)
            notes.append(f"Port scuttled: integrity={port_integrity:.2f}")

        # Operational port strikes (toy): Port Denial pushes sustained throughput shock.
        port_shock = max(0.0, port_shock * 0.88)  # decay each day
        if enable_orders and (order_blue is not None) and (order_blue.name == "Port Denial") and (m_blue > 0):
            port_shock = min(0.75, port_shock + 0.06 * (b_to_land / max(1.0, m_blue)) * (0.45 + 0.55 * blue_pk))
        if enable_orders and (order_red is not None) and (order_red.name == "Port Denial") and (m_red > 0):
            port_shock = min(0.75, port_shock + 0.05 * (r_to_land / max(1.0, m_red)) * (0.45 + 0.55 * red_pk))

        if port_shock > 0.02:
            notes.append(f"Port integrity shock (strikes): -{port_shock*100:.0f}%")
        port_integrity = float(np.clip(port_integrity * (1.0 - port_shock), 0.05, 1.0))

        supportable = _ic_supportable_battalions(
            port_capacity_tons_per_day=cfg.port_capacity_tons_per_day,
            integrity_coeff_0_1=port_integrity,
            consumption_tons_per_battalion_day=cfg.consumption_tons_per_battalion_day,
            lots_capacity_tons_per_day=red.lots_capacity_tons_per_day,
        )

        if landed_battalions > supportable + 1e-9:
            red.ground_effectiveness_0_1 = float(np.clip(red.ground_effectiveness_0_1 * (1.0 - cfg.daily_effectiveness_loss_if_unsupplied), 0.0, 1.0))
            notes.append(f"Supply shortfall: landed={landed_battalions:.1f} > supportable={supportable:.1f}; Red eff -> {red.ground_effectiveness_0_1:.2f}")

        # Land attrition (tanks+artillery) once landing is meaningful
        if landed_battalions > 1.0:
            # committed land force scales with landed fraction
            frac = float(np.clip(landed_battalions / max(1e-9, red.ground_battalions_committed), 0.0, 1.0))
            red_land = (red.tanks + 0.35 * red.artillery) * frac * red.ground_effectiveness_0_1
            blue_land = (blue.tanks + 0.35 * blue.artillery) * cfg.blue_defense_multiplier
            kB_land = float(cfg.k_blue_land) * (float(order_blue.k_land_mult) if (enable_orders and order_blue is not None) else 1.0)
            kR_land = float(cfg.k_red_land)  * (float(order_red.k_land_mult)  if (enable_orders and order_red  is not None) else 1.0)
            blue_land_next, red_land_next = _ic_lanchester_square_step(blue_land, red_land, kB_land, kR_land, 1.0)
            # map back to tank+artillery proportional drawdown
            if blue_land > 0:
                blue_loss_frac = float(np.clip((blue_land - blue_land_next) / blue_land, 0.0, 0.25))
                blue.tanks *= (1.0 - blue_loss_frac)
                blue.artillery *= (1.0 - 0.8 * blue_loss_frac)
            if red_land > 0:
                red_loss_frac = float(np.clip((red_land - red_land_next) / red_land, 0.0, 0.25))
                red.tanks *= (1.0 - red_loss_frac)
                red.artillery *= (1.0 - 0.8 * red_loss_frac)

        # 6) Victory condition
        red_ground_power = landed_battalions * cfg.combat_power_per_battalion * red.ground_effectiveness_0_1
        blue_ground_power = cfg.blue_defense_multiplier * cfg.combat_power_per_battalion * blue.ground_battalions_committed

        if red_ground_power > blue_ground_power:
            winner = "Red"
            end_reason = "Red ground power exceeded Blue defense."
        elif day == cfg.days:
            winner = "Blue"
            end_reason = "Time horizon reached: Blue held."

        # Update will (toy)
        def _loss_frac(init_val: float, cur_val: float) -> float:
            return 0.0 if init_val <= 0 else float(np.clip((init_val - cur_val) / init_val, 0.0, 1.0))
        will_blue = max(0.0, will_blue - 8.0 * _loss_frac(init["blue"]["aircraft"], blue.aircraft) - 15.0 * _loss_frac(init["blue"]["ships"], blue.fleet.total()))
        will_red  = max(0.0, will_red  - 8.0 * _loss_frac(init["red"]["aircraft"],  red.aircraft)  - 15.0 * _loss_frac(init["red"]["ships"],  red.fleet.total()))

        # Job-Done sustainment: repairs + production + endurance drain (Integrated)
        pre_b = _jd_pre_mult(jd["blue"], day, float(jd["base_id"]), {"aircraft": blue.aircraft, "tanks": blue.tanks, "ships": blue.fleet.total(), "artillery": blue.artillery}, init["blue"], jd["scenario_type"])
        pre_r = _jd_pre_mult(jd["red"],  day, float(jd["base_id"]), {"aircraft": red.aircraft,  "tanks": red.tanks,  "ships": red.fleet.total(),  "artillery": red.artillery},  init["red"],  jd["scenario_type"])
        # approximate losses by delta from previous-step snapshot stored earlier in loop
        # (we keep last values on state objects)
        if "prev_aircraft" not in locals():
            prev_aircraft = {"blue": blue.aircraft, "red": red.aircraft}
            prev_tanks = {"blue": blue.tanks, "red": red.tanks}
            prev_art = {"blue": blue.artillery, "red": red.artillery}
            prev_miss = {"blue": blue.missiles_stock, "red": red.missiles_stock}
            prev_ships = {"blue": float(blue.fleet.total()), "red": float(red.fleet.total())}

        blue_losses_today = {
            "aircraft": max(0.0, float(prev_aircraft["blue"]) - float(blue.aircraft)),
            "tanks": max(0.0, float(prev_tanks["blue"]) - float(blue.tanks)),
            "ships": max(0.0, float(prev_ships["blue"]) - float(blue.fleet.total())),
            "artillery": max(0.0, float(prev_art["blue"]) - float(blue.artillery)),
        }
        red_losses_today = {
            "aircraft": max(0.0, float(prev_aircraft["red"]) - float(red.aircraft)),
            "tanks": max(0.0, float(prev_tanks["red"]) - float(red.tanks)),
            "ships": max(0.0, float(prev_ships["red"]) - float(red.fleet.total())),
            "artillery": max(0.0, float(prev_art["red"]) - float(red.artillery)),
        }

        # Approx daily personnel casualties from today's equipment losses (keeps Battle Losses 'Today' consistent)
        raw_b = 12.0 * blue_losses_today.get('aircraft', 0.0) + 280.0 * blue_losses_today.get('ships', 0.0) + 2.0 * blue_losses_today.get('tanks', 0.0) + 1.2 * blue_losses_today.get('artillery', 0.0)
        raw_r = 12.0 * red_losses_today.get('aircraft', 0.0) + 280.0 * red_losses_today.get('ships', 0.0) + 2.0 * red_losses_today.get('tanks', 0.0) + 1.2 * red_losses_today.get('artillery', 0.0)
        kia_today_blue = 0.22 * raw_b
        wia_today_blue = 0.78 * raw_b
        kia_today_red  = 0.22 * raw_r
        wia_today_red  = 0.78 * raw_r
        cum_kia_blue += kia_today_blue
        cum_wia_blue += wia_today_blue
        cum_kia_red  += kia_today_red
        cum_wia_red  += wia_today_red

        # apply production to the integrated state variables
        tmp_rem_blue = {"aircraft": blue.aircraft, "tanks": blue.tanks, "ships": blue.fleet.total(), "artillery": blue.artillery, "missiles": blue.missiles_stock}
        tmp_rem_red  = {"aircraft": red.aircraft,  "tanks": red.tanks,  "ships": red.fleet.total(),  "artillery": red.artillery,  "missiles": red.missiles_stock}
        _jd_apply_post(jd["blue"], day, float(pre_b["Id"]), tmp_rem_blue, init["blue"], blue_losses_today, jd["scenario_type"])
        _jd_apply_post(jd["red"],  day, float(pre_r["Id"]), tmp_rem_red,  init["red"],  red_losses_today,  jd["scenario_type"])
        blue.aircraft = float(tmp_rem_blue["aircraft"]); blue.tanks = float(tmp_rem_blue["tanks"]); blue.artillery = float(tmp_rem_blue["artillery"]); blue.missiles_stock = float(tmp_rem_blue["missiles"])
        red.aircraft  = float(tmp_rem_red["aircraft"]);  red.tanks  = float(tmp_rem_red["tanks"]);  red.artillery  = float(tmp_rem_red["artillery"]);  red.missiles_stock  = float(tmp_rem_red["missiles"])

        prev_aircraft = {"blue": blue.aircraft, "red": red.aircraft}
        prev_tanks = {"blue": blue.tanks, "red": red.tanks}
        prev_art = {"blue": blue.artillery, "red": red.artillery}
        prev_miss = {"blue": blue.missiles_stock, "red": red.missiles_stock}
        prev_ships = {"blue": float(blue.fleet.total()), "red": float(red.fleet.total())}

        # Aggregate power (for UI continuity)
        blue_power = (blue.aircraft ** 0.5) + 1.4 * (blue.fleet.total() ** 0.5) + 0.8 * ((blue.tanks + 0.35 * blue.artillery) ** 0.5)
        red_power  = (red.aircraft ** 0.5)  + 1.4 * (red.fleet.total() ** 0.5)  + 0.8 * ((red.tanks + 0.35 * red.artillery) ** 0.5)

        series.append({
            "day": day,
            "blue_power": float(blue_power),
            "red_power": float(red_power),
            # UI expects blue_will/red_will (used by dot arena + charts)
            "blue_will": float(will_blue),
            "red_will": float(will_red),
            # keep legacy keys too (harmless, helps backwards compatibility)
            "will_blue": float(will_blue),
            "will_red": float(will_red),
            "blue_aircraft": float(blue.aircraft),
            "red_aircraft": float(red.aircraft),
            "blue_ships": int(blue.fleet.total()),
            "red_ships": int(red.fleet.total()),
            "blue_tanks": float(blue.tanks),
            "red_tanks": float(red.tanks),
            "blue_artillery": float(blue.artillery),
            "red_artillery": float(red.artillery),
            "missiles_blue_left": float(blue.missiles_stock),
            "missiles_red_left": float(red.missiles_stock),
            "interceptors_blue_left": float(blue.interceptors_stock),
            "interceptors_red_left": float(red.interceptors_stock),
            "landed_battalions": float(landed_battalions),
            "supportable_battalions": float(supportable),
            "port_integrity": float(port_integrity),
            "red_ground_effectiveness": float(red.ground_effectiveness_0_1),
            "winner": winner,
            "end_reason": end_reason,
            "order_blue": (order_blue.name if order_blue is not None else ""),
            "order_red": (order_red.name if order_red is not None else ""),
            "blue_total_kia": float(cum_kia_blue),
            "red_total_kia": float(cum_kia_red),
            "blue_total_wia": float(cum_wia_blue),
            "red_total_wia": float(cum_wia_red),
            "notes": notes,
        })

        if winner is not None:
            break

    # Final remaining
    rem = {
        "blue": {"aircraft": blue.aircraft, "ships": blue.fleet.total(), "tanks": blue.tanks, "artillery": blue.artillery},
        "red":  {"aircraft": red.aircraft,  "ships": red.fleet.total(),  "tanks": red.tanks,  "artillery": red.artillery},
    }
    losses = {
        "blue": {k: max(0.0, init["blue"][k] - rem["blue"][k]) for k in ("aircraft","ships","tanks","artillery")},
        "red":  {k: max(0.0, init["red"][k]  - rem["red"][k])  for k in ("aircraft","ships","tanks","artillery")},
    }

    # Approx personnel casualties proxy (very rough)
    def casualties(side_key: str) -> Tuple[float, float]:
        la = losses[side_key]["aircraft"]
        ls = losses[side_key]["ships"]
        lt = losses[side_key]["tanks"]
        lart = losses[side_key]["artillery"]
        # scale: ship losses are "heavy", air moderate, land moderate
        raw = 12.0 * la + 280.0 * ls + 2.0 * lt + 1.2 * lart
        kia = 0.22 * raw
        wia = 0.78 * raw
        return kia, wia

    kia_b, wia_b = casualties("blue")
    kia_r, wia_r = casualties("red")

    # Recovery days proxy (same style as cinematic model)
    def recovery_days(side_ag: dict, rem_side: dict, init_side: dict) -> int:
        econ = max(0.25, float(side_ag.get("economy_score", 50.0) or 50.0) / 100.0)
        logi = max(0.20, float(side_ag.get("logistics_score", 50.0) or 50.0) / 100.0)
        baseline = float(init_side["aircraft"] + init_side["ships"] + init_side["tanks"] + init_side["artillery"])
        remaining = float(rem_side["aircraft"] + rem_side["ships"] + rem_side["tanks"] + rem_side["artillery"])
        loss_frac = 0.0 if baseline <= 0 else float(np.clip((baseline - remaining) / baseline, 0.0, 1.0))
        # higher econ/logistics -> faster; higher loss -> slower
        days = int(round(60 + 520 * loss_frac / max(0.15, econ * logi)))
        return int(np.clip(days, 30, 2500))

    # Need aggs again for recovery
    df = bundle.countries
    bg = df[df["ISO3"] == sc.battleground_iso3]
    bg_row = bg.iloc[0].to_dict() if len(bg) else {}
    theatre = _theatre_modifiers(bg_row)

    blue_ag = _side_aggregate(df, sc.blue, sc.sanctions_blue, sc.neutral_support_blue) if sc.blue else _side_aggregate(df, ["USA"], sc.sanctions_blue, sc.neutral_support_blue)
    red_ag  = _side_aggregate(df, sc.red,  sc.sanctions_red,  sc.neutral_support_red)  if sc.red  else _side_aggregate(df, ["CHN"], sc.sanctions_red,  sc.neutral_support_red)

    summary = dict(
        core_model="Integrated (4-engine)",
        winner=winner,
        end_reason=end_reason,
        duration_days=int(series[-1]["day"]) if series else 0,
        battleground=sc.battleground_iso3,
        year=int(sc.year),
        intensity=str(sc.intensity),
        mode=str(sc.mode),
        theatre=theatre,
        local_side=local_side,
        init=init,
        totals=dict(
            blue=dict(
                kia=float(kia_b),
                wia=float(wia_b),
                civ=float(0.0),
                aircraft=float(losses["blue"]["aircraft"]),
                ships=float(losses["blue"]["ships"]),
                tanks=float(losses["blue"]["tanks"]),
                artillery=float(losses["blue"]["artillery"]),
                missiles=float(max(0.0, init["blue"]["missiles"] - blue.missiles_stock)),
                interceptors=float(max(0.0, init["blue"]["interceptors"] - blue.interceptors_stock)),
            ),
            red=dict(
                kia=float(kia_r),
                wia=float(wia_r),
                civ=float(0.0),
                aircraft=float(losses["red"]["aircraft"]),
                ships=float(losses["red"]["ships"]),
                tanks=float(losses["red"]["tanks"]),
                artillery=float(losses["red"]["artillery"]),
                missiles=float(max(0.0, init["red"]["missiles"] - red.missiles_stock)),
                interceptors=float(max(0.0, init["red"]["interceptors"] - red.interceptors_stock)),
            ),
        ),
        recovery_days=dict(
            blue=int(recovery_days(blue_ag, rem["blue"], init["blue"])),
            red=int(recovery_days(red_ag, rem["red"], init["red"])),
        ),
        doctrines=dict(
            blue=list(blue_ag.get("doctrines", []) or []),
            red=list(red_ag.get("doctrines", []) or []),
        ),
        nukes_allowed=bool(sc.nukes_allowed),
        integrated_cfg=cfg.__dict__,
    )

    return summary, series

def fit_integrated_cfg(
    bundle: DataBundle,
    sc: Scenario,
    targets: Dict[str, Dict[str, float]],
    *,
    iters: int = 1200,
    mc: int = 3,
    seed: int = 0,
) -> Tuple[dict, float]:
    """
    Random-search calibration for Integrated model.
    targets example:
      {"blue":{"aircraft_lost":300,"ships_lost":15}, "red":{"aircraft_lost":160,"ships_lost":140}}
    Returns: (best_integrated_cfg_dict, best_loss)
    """
    rng = np.random.default_rng(seed)
    best_cfg = None
    best_loss = 1e18

    base_cfg = dict(getattr(sc, "integrated_cfg", {}) or {})

    def loss_of(cfg_dict: dict) -> float:
        # Average over a few MC seeds to reduce stochastic noise
        e = 0.0
        for k in range(mc):
            sc2 = Scenario(**{**sc.__dict__, "core_model": "Integrated (4-engine)", "integrated_cfg": cfg_dict, "seed": int(seed + 17*k)})
            summ, _ = simulate_integrated(bundle, sc2)
            for side in ("blue","red"):
                for key in ("aircraft_lost","ships_lost"):
                    tgt = float(targets.get(side, {}).get(key, 0.0))
                    if tgt <= 0:
                        continue
                    pred = float(summ["totals"][side][key])
                    e += ((pred - tgt) / (tgt + 1e-9))**2
        return e / max(1, mc)

    # Search space
    for _ in range(int(iters)):
        cand = dict(base_cfg)
        cand["blue_detection_prob"] = float(rng.uniform(0.35, 0.80))
        cand["k_blue_air"] = float(rng.uniform(0.0010, 0.0042))
        cand["k_red_air"]  = float(rng.uniform(0.0010, 0.0042))
        cand["missile_fraction_to_ships"] = float(rng.uniform(0.40, 0.82))
        cand["pk_bonus_if_detected"] = float(rng.uniform(0.03, 0.09))
        cand["pd_bonus_if_detected"] = float(rng.uniform(0.05, 0.14))
        cand["daily_effectiveness_loss_if_unsupplied"] = float(rng.uniform(0.06, 0.14))
        cand["scuttle_day"] = int(rng.integers(1, min(7, int(sc.max_days or 28)) + 1))

        e = loss_of(cand)
        if e < best_loss:
            best_loss = e
            best_cfg = cand

    return (best_cfg or base_cfg), float(best_loss)


def allocate_losses_to_countries(bundle: DataBundle, iso3s: List[str], totals: dict, committed_weight: str = "mil_spend_usd") -> pd.DataFrame:
    df = bundle.countries
    sub = df[df["ISO3"].isin(iso3s)].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["ISO3", "Country_Display"])
    w = pd.to_numeric(sub.get(committed_weight, 1.0), errors="coerce").fillna(1.0).clip(lower=0.0)
    if float(w.sum()) <= 0:
        w = pd.Series(np.ones(len(sub)), index=sub.index)
    share = w / w.sum()

    out = sub[["ISO3", "Country_Display"]].copy()
    for k in ["kia", "wia", "civ", "aircraft", "tanks", "ships", "artillery", "missiles", "interceptors"]:
        out[k] = (share * float(totals.get(k, 0.0))).round(0).astype(int)
    out["share"] = share.values
    out = out.sort_values(["kia", "aircraft", "tanks", "ships"], ascending=False)
    return out.reset_index(drop=True)
