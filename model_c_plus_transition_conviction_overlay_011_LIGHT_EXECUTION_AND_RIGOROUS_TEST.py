"""
model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST.py

Purpose
-------
Light execution version based on the heavy 011 transition-conviction overlay.
It does NOT run grid search every day. It:

1) Loads your current-best rebalance log:
   model_c_plus_current_best_with_divergence_alerts_rebalance_log.csv

2) Loads best 011 parameters if available from:
   model_c_plus_transition_conviction_overlay_011_grid_results_sorted.csv
   Otherwise uses the frozen default 011 config below.

3) Rebuilds 011 transition/conviction features without lookahead.

4) Applies the 011 dynamic execution-weight logic to the latest row.

5) Saves production latest recommendation CSV:
   model_c_plus_vol_target_leverage_budget_010_latest_recommendation.csv

6) Runs rigorous validation using the same execution logic:
   - executed-weight turnover cost
   - transaction cost sensitivity
   - stress windows
   - rebalance interval diagnostics from your existing rebalance log

Important
---------
This script is only as good as the input rebalance log. The heavy 011 model is a second-layer overlay
on top of current-best signals. It does NOT recreate the original RandomForest signal engine.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).resolve().parent

PREFIX_OUT = "model_c_plus_transition_conviction_overlay_011_LIGHT"
CURRENT_PREFIX = "model_c_plus_current_best_with_divergence_alerts"
GENERIC_LATEST = "model_c_plus_vol_target_leverage_budget_010_latest_recommendation.csv"

BASE_ASSETS = ["QQQM", "XLE", "XSOE", "XLI", "XLB", "BIL"]
EXEC_ASSETS = ["TQQQ", "UXI", "ERX", "QQQM", "XLE", "XSOE", "XLI", "XLB", "BIL"]
PRICE_ASSETS = sorted(set(EXEC_ASSETS + BASE_ASSETS))

DEFAULT_TRANSACTION_COST = 0.001

# =========================
# Performance helpers
# =========================

def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    cumulative = (1.0 + r).prod()
    years = len(r) / 252.0
    if years <= 0:
        return np.nan
    return cumulative ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return r.std() * np.sqrt(252.0)


def sharpe_ratio(returns: pd.Series) -> float:
    vol = annualized_volatility(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return annualized_return(returns) / vol


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def summary(name: str, returns: pd.Series, avg_turnover=np.nan, trade_count=np.nan, tx_cost=np.nan) -> dict:
    r = returns.dropna()
    return {
        "model": name,
        "annual_return": annualized_return(r),
        "volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "avg_turnover": avg_turnover,
        "trade_count": trade_count,
        "transaction_cost": tx_cost,
        "start": r.index.min(),
        "end": r.index.max(),
        "days": len(r),
        "final_equity": (1.0 + r).prod() if len(r) else np.nan,
    }


def compute_turnover(old_w: dict, new_w: dict, universe: list) -> float:
    return float(sum(abs(new_w.get(a, 0.0) - old_w.get(a, 0.0)) for a in universe))


def safe_float(row, col, default=0.0):
    try:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    except Exception:
        pass
    return default


def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def positive(x):
    return max(0.0, float(x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-clip(x, -50.0, 50.0)))

# =========================
# Frozen fallback config from heavy 011 base config.
# If grid_results_sorted exists, the script overrides these with the top row.
# =========================

FALLBACK_CFG = dict(
    budget_base=0.20,
    budget_tech_floor=-0.10,
    budget_ind_floor=-0.10,
    budget_energy_floor=0.10,
    budget_risk_floor=0.40,
    budget_crash_floor=1.20,
    budget_move_floor=1.20,
    budget_tech_mult=0.25,
    budget_ind_mult=0.10,
    budget_energy_mult=0.05,
    budget_risk_mult=0.20,
    budget_crash_mult=0.20,
    budget_move_mult=0.08,
    budget_min=0.00,
    budget_max=0.85,
    vol_target=0.18,
    vol_floor=0.08,
    vol_scale_min=0.55,
    vol_scale_max=1.15,
    hard_risk_off_max=1.60,
    hard_crash_max=2.50,
    tqqq_growth_min=-0.10,
    tqqq_soxx_min=-0.10,
    tqqq_risk_off_max=1.25,
    uxi_industrial_min=-0.20,
    uxi_credit_min=-0.70,
    uxi_copper_min=-0.10,
    uxi_materials_min=-0.25,
    uxi_risk_off_max=1.25,
    erx_war_min=0.00,
    erx_credit_min=-0.60,
    erx_risk_off_max=0.80,
    score_base=0.50,
    tqqq_priority=2.50,
    uxi_priority=0.80,
    erx_priority=0.35,
    krw_bonus=0.05,
    tech_dominance_min=0.50,
    tech_dominance_boost=1.25,
    tqqq_abs_max=0.75,
    uxi_abs_max=0.35,
    erx_abs_max=0.25,
    instability_mid=0.25,
    instability_slope=1.25,
    instability_cut=0.25,
    opportunity_mid=0.25,
    opportunity_slope=1.00,
    opportunity_boost=0.10,
    leader_flip_cut=0.10,
    entropy_floor=0.55,
    entropy_cut=0.10,
    danger_transition_cut=0.10,
    danger_transition_cap=2.0,
    wide_gap_min=0.012,
    wide_gap_boost=0.05,
    conviction_scale_min=0.50,
    conviction_scale_max=1.15,
)


def load_best_cfg() -> tuple[dict, str]:
    cfg = dict(FALLBACK_CFG)
    candidates = [
        SCRIPT_DIR / "model_c_plus_transition_conviction_overlay_011_grid_results_sorted.csv",
        SCRIPT_DIR / "model_c_plus_transition_conviction_overlay_011_best_summary.csv",
    ]
    sorted_path = candidates[0]
    if sorted_path.exists():
        df = pd.read_csv(sorted_path)
        if len(df) > 0:
            row = df.iloc[0]
            for k in cfg.keys():
                if k in row.index and pd.notna(row[k]):
                    cfg[k] = float(row[k])
            model_name = str(row["model"]) if "model" in row.index else "BEST_FROM_SORTED_GRID"
            return cfg, f"best config loaded from {sorted_path.name}: {model_name}"
    return cfg, "WARNING: grid_results_sorted not found; using fallback 011 base config"

# =========================
# Data loaders
# =========================

def require_file(filename: str) -> Path:
    p = SCRIPT_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return p


def load_rebalance_log() -> pd.DataFrame:
    p = require_file(f"{CURRENT_PREFIX}_rebalance_log.csv")
    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError(f"{p.name} must contain a date column")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_current_returns_optional() -> pd.Series | None:
    p = SCRIPT_DIR / f"{CURRENT_PREFIX}_portfolio_daily_returns.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        first = df.columns[0]
        df[first] = pd.to_datetime(df[first])
        df = df.set_index(first)
    ret_col = "portfolio_return" if "portfolio_return" in df.columns else df.columns[0]
    return df[ret_col].astype(float).sort_index()


def download_prices(start_date: str) -> pd.DataFrame:
    prices = yf.download(PRICE_ASSETS, start=start_date, auto_adjust=True, progress=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.sort_index().dropna(how="all")
    missing = [a for a in PRICE_ASSETS if a not in prices.columns]
    if missing:
        raise ValueError(f"Missing downloaded prices for: {missing}")
    return prices.dropna(subset=PRICE_ASSETS)


def align_rebalance_dates(rebalance: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    def align_date(dt: pd.Timestamp) -> pd.Timestamp | None:
        idx = prices.index.searchsorted(dt)
        if idx >= len(prices.index):
            return None
        return prices.index[idx]

    d = rebalance.copy()
    d["aligned_date"] = d["date"].apply(align_date)
    d = d.dropna(subset=["aligned_date"]).copy()
    d["aligned_date"] = pd.to_datetime(d["aligned_date"])
    d = d.drop_duplicates("aligned_date").sort_values("aligned_date").reset_index(drop=True)
    return d

# =========================
# Signal and transition features
# =========================

def get_signal_weights(row) -> dict:
    w = {a: 0.0 for a in BASE_ASSETS}
    has_signal = any(f"signal_w_{a}" in row.index for a in BASE_ASSETS)
    if has_signal:
        for a in BASE_ASSETS:
            w[a] = max(0.0, safe_float(row, f"signal_w_{a}", 0.0))
    else:
        w["QQQM"] = max(0.0, safe_float(row, "exec_w_QQQM", 0.0) + safe_float(row, "exec_w_TQQQ", 0.0))
        w["XLI"] = max(0.0, safe_float(row, "exec_w_XLI", 0.0) + safe_float(row, "exec_w_UXI", 0.0))
        w["XLE"] = max(0.0, safe_float(row, "exec_w_XLE", 0.0) + safe_float(row, "exec_w_ERX", 0.0))
        for a in ["XSOE", "XLB", "BIL"]:
            w[a] = max(0.0, safe_float(row, f"exec_w_{a}", 0.0))
    s = sum(w.values())
    if s <= 0:
        w = {a: 0.0 for a in BASE_ASSETS}
        w["BIL"] = 1.0
        s = 1.0
    return {k: v / s for k, v in w.items()}


def infer_top_asset_from_row(row) -> str:
    for col in ["top_asset", "top", "winner", "best_asset"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    sig = get_signal_weights(row)
    return max(sig.keys(), key=lambda a: sig[a])


def get_score_gap(row) -> float:
    for col in ["gap", "score_gap", "top_second_gap", "top_gap"]:
        if col in row.index and pd.notna(row[col]):
            return safe_float(row, col, 0.0)
    vals = []
    for a in ["QQQM", "XLE", "XSOE", "XLI", "XLB"]:
        for prefix in ["pred_", "adjusted_pred_", "score_", "adj_score_", "adj_pred_", "raw_pred_"]:
            c = f"{prefix}{a}"
            if c in row.index:
                vals.append(safe_float(row, c, np.nan))
    vals = [v for v in vals if pd.notna(v)]
    if len(vals) >= 2:
        vals = sorted(vals, reverse=True)
        return float(vals[0] - vals[1])
    return 0.0


def row_signal_entropy(row) -> float:
    sig = get_signal_weights(row)
    vals = np.array([max(0.0, sig[a]) for a in BASE_ASSETS], dtype=float)
    s = vals.sum()
    if s <= 0:
        return 1.0
    p = vals / s
    p = p[p > 1e-12]
    ent = -np.sum(p * np.log(p))
    max_ent = np.log(len(BASE_ASSETS))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("aligned_date").reset_index(drop=True)
    d["top_asset_inferred"] = d.apply(infer_top_asset_from_row, axis=1)
    d["score_gap_inferred"] = d.apply(get_score_gap, axis=1)
    d["signal_entropy"] = d.apply(row_signal_entropy, axis=1)

    for col in [
        "growth_strength", "soxx_strength", "risk_off_strength", "industrial_strength",
        "materials_strength", "copper_strength", "credit_strength", "hyg_strength",
        "war_strength", "crash_pressure", "move_stress", "krw_strength",
    ]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    if d["credit_strength"].abs().sum() == 0 and "hyg_strength" in d.columns:
        d["credit_strength"] = d["hyg_strength"]

    d["tech_state_calc"] = 0.55 * d["growth_strength"] + 0.45 * d["soxx_strength"]
    d["industrial_state_calc"] = (
        0.45 * d["industrial_strength"]
        + 0.25 * d["copper_strength"]
        + 0.20 * d["materials_strength"]
        + 0.10 * d["credit_strength"]
    )
    d["top_asset_changed"] = (d["top_asset_inferred"] != d["top_asset_inferred"].shift(1)).astype(float)
    d["tech_state_chg"] = d["tech_state_calc"].diff().abs().fillna(0.0)
    d["risk_off_chg"] = d["risk_off_strength"].diff().abs().fillna(0.0)
    d["industrial_state_chg"] = d["industrial_state_calc"].diff().abs().fillna(0.0)
    d["leader_flip_4"] = d["top_asset_changed"].rolling(4, min_periods=1).mean()
    d["leader_flip_8"] = d["top_asset_changed"].rolling(8, min_periods=1).mean()
    d["tech_down"] = (-d["tech_state_calc"].diff()).clip(lower=0.0).fillna(0.0)
    d["risk_off_up"] = d["risk_off_strength"].diff().clip(lower=0.0).fillna(0.0)
    d["danger_transition"] = d["tech_down"] * d["risk_off_up"]

    d["opportunity_stability_raw"] = (
        0.65 * d["tech_state_calc"]
        + 0.35 * d["score_gap_inferred"] * 50.0
        - 0.60 * d["risk_off_strength"]
        - 0.30 * d["signal_entropy"]
        - 0.25 * d["leader_flip_4"]
    )
    d["transition_instability_raw"] = (
        1.20 * d["leader_flip_4"]
        + 0.75 * d["signal_entropy"]
        + 0.55 * d["tech_state_chg"]
        + 0.55 * d["risk_off_chg"]
        + 0.35 * d["industrial_state_chg"]
        + 0.40 * d["danger_transition"]
        - 0.35 * d["score_gap_inferred"] * 50.0
    )

    for col in ["transition_instability_raw", "opportunity_stability_raw"]:
        mean = d[col].expanding(min_periods=20).mean().shift(1)
        std = d[col].expanding(min_periods=20).std().shift(1)
        z = (d[col] - mean) / std.replace(0, np.nan)
        d[col.replace("_raw", "_z")] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d


def build_realized_vol(prices: pd.DataFrame) -> pd.Series:
    base_vol_returns = pd.DataFrame(index=prices.index)
    for a in ["QQQM", "XLI", "XLE", "XLB", "XSOE"]:
        base_vol_returns[a] = prices[a].pct_change()
    vol_weights = {"QQQM": 0.45, "XLI": 0.20, "XLE": 0.15, "XLB": 0.10, "XSOE": 0.10}
    vol_proxy_daily = pd.Series(0.0, index=prices.index)
    for a, w in vol_weights.items():
        vol_proxy_daily = vol_proxy_daily.add(w * base_vol_returns[a].fillna(0.0), fill_value=0.0)
    return (0.65 * vol_proxy_daily.rolling(21).std() * np.sqrt(252.0) +
            0.35 * vol_proxy_daily.rolling(63).std() * np.sqrt(252.0)).ffill()

# =========================
# 011 light execution logic
# =========================

def build_dynamic_exec_weights(row, cfg: dict, realized_vol_blend: pd.Series) -> tuple[dict, dict]:
    sig = get_signal_weights(row)
    growth = safe_float(row, "growth_strength", 0.0)
    soxx = safe_float(row, "soxx_strength", 0.0)
    risk_off = safe_float(row, "risk_off_strength", 0.0)
    industrial = safe_float(row, "industrial_strength", 0.0)
    materials = safe_float(row, "materials_strength", 0.0)
    copper = safe_float(row, "copper_strength", 0.0)
    credit = safe_float(row, "credit_strength", safe_float(row, "hyg_strength", 0.0))
    war = safe_float(row, "war_strength", 0.0)
    crash = safe_float(row, "crash_pressure", 0.0)
    move_stress = safe_float(row, "move_stress", 0.0)
    krw = safe_float(row, "krw_strength", 0.0)
    transition_z = safe_float(row, "transition_instability_z", 0.0)
    opportunity_z = safe_float(row, "opportunity_stability_z", 0.0)
    leader_flip_4 = safe_float(row, "leader_flip_4", 0.0)
    entropy = safe_float(row, "signal_entropy", 0.0)
    danger_transition = safe_float(row, "danger_transition", 0.0)
    score_gap = safe_float(row, "score_gap_inferred", 0.0)

    if risk_off >= cfg["hard_risk_off_max"] or crash >= cfg["hard_crash_max"]:
        exec_w = {a: 0.0 for a in EXEC_ASSETS}
        for a in BASE_ASSETS:
            exec_w[a] = sig.get(a, 0.0)
        details = {"total_budget": 0.0, "budget_reason": "hard risk off", "conviction_scale": 0.0,
                   "transition_instability_z": transition_z, "opportunity_stability_z": opportunity_z,
                   "allowed_TQQQ": False, "allowed_UXI": False, "allowed_ERX": False,
                   "score_TQQQ": 0.0, "score_UXI": 0.0, "score_ERX": 0.0}
        return exec_w, details

    tech_state = 0.55 * growth + 0.45 * soxx
    industrial_state = 0.45 * industrial + 0.25 * copper + 0.20 * materials + 0.10 * credit
    energy_state = 0.65 * war

    raw_budget = (
        cfg["budget_base"]
        + cfg["budget_tech_mult"] * positive(tech_state - cfg["budget_tech_floor"])
        + cfg["budget_ind_mult"] * positive(industrial_state - cfg["budget_ind_floor"])
        + cfg["budget_energy_mult"] * positive(energy_state - cfg["budget_energy_floor"])
        - cfg["budget_risk_mult"] * positive(risk_off - cfg["budget_risk_floor"])
        - cfg["budget_crash_mult"] * positive(crash - cfg["budget_crash_floor"])
        - cfg["budget_move_mult"] * positive(move_stress - cfg["budget_move_floor"])
    )
    total_budget_before_vol = clip(raw_budget, cfg["budget_min"], cfg["budget_max"])
    dt = pd.to_datetime(row.get("aligned_date", row.get("date", pd.NaT)))
    realized_vol_now = float(realized_vol_blend.loc[dt]) if pd.notna(dt) and dt in realized_vol_blend.index and pd.notna(realized_vol_blend.loc[dt]) else np.nan
    if np.isnan(realized_vol_now) or realized_vol_now <= 0:
        vol_scale = 1.0
    else:
        vol_scale = clip(cfg["vol_target"] / max(realized_vol_now, cfg["vol_floor"]), cfg["vol_scale_min"], cfg["vol_scale_max"])
    total_budget_before_conviction = clip(total_budget_before_vol * vol_scale, cfg["budget_min"], cfg["budget_max"])

    instability_penalty = sigmoid(cfg["instability_slope"] * (transition_z - cfg["instability_mid"]))
    opportunity_bonus = sigmoid(cfg["opportunity_slope"] * (opportunity_z - cfg["opportunity_mid"]))
    conviction_scale = (
        1.0
        - cfg["instability_cut"] * instability_penalty
        + cfg["opportunity_boost"] * opportunity_bonus
        - cfg["leader_flip_cut"] * leader_flip_4
        - cfg["entropy_cut"] * max(0.0, entropy - cfg["entropy_floor"])
        - cfg["danger_transition_cut"] * min(danger_transition, cfg["danger_transition_cap"])
    )
    if score_gap >= cfg["wide_gap_min"]:
        conviction_scale += cfg["wide_gap_boost"]
    conviction_scale = clip(conviction_scale, cfg["conviction_scale_min"], cfg["conviction_scale_max"])
    total_budget = clip(total_budget_before_conviction * conviction_scale, cfg["budget_min"], cfg["budget_max"])

    tqqq_allowed = sig.get("QQQM", 0.0) > 0 and growth >= cfg["tqqq_growth_min"] and soxx >= cfg["tqqq_soxx_min"] and risk_off <= cfg["tqqq_risk_off_max"]
    uxi_allowed = sig.get("XLI", 0.0) > 0 and industrial >= cfg["uxi_industrial_min"] and credit >= cfg["uxi_credit_min"] and copper >= cfg["uxi_copper_min"] and materials >= cfg["uxi_materials_min"] and risk_off <= cfg["uxi_risk_off_max"]
    erx_allowed = sig.get("XLE", 0.0) > 0 and war >= cfg["erx_war_min"] and risk_off <= cfg["erx_risk_off_max"] and credit >= cfg["erx_credit_min"]

    scores = {"TQQQ": 0.0, "UXI": 0.0, "ERX": 0.0}
    if tqqq_allowed:
        scores["TQQQ"] = cfg["tqqq_priority"] * sig["QQQM"] * (cfg["score_base"] + positive(tech_state - cfg["tqqq_growth_min"]))
        if krw > 0:
            scores["TQQQ"] *= (1.0 + cfg["krw_bonus"] * min(krw, 2.0))
    if uxi_allowed:
        scores["UXI"] = cfg["uxi_priority"] * sig["XLI"] * (cfg["score_base"] + positive(industrial_state - cfg["uxi_industrial_min"]))
    if erx_allowed:
        scores["ERX"] = cfg["erx_priority"] * sig["XLE"] * (cfg["score_base"] + positive(energy_state - cfg["erx_war_min"]))
    if scores["TQQQ"] > 0 and tech_state > cfg["tech_dominance_min"]:
        scores["TQQQ"] *= cfg["tech_dominance_boost"]

    score_sum = sum(scores.values())
    target_lev = {"TQQQ": 0.0, "UXI": 0.0, "ERX": 0.0}
    if score_sum > 0 and total_budget > 0:
        for lev in target_lev:
            target_lev[lev] = total_budget * scores[lev] / score_sum
    target_lev["TQQQ"] = min(target_lev["TQQQ"], cfg["tqqq_abs_max"])
    target_lev["UXI"] = min(target_lev["UXI"], cfg["uxi_abs_max"])
    target_lev["ERX"] = min(target_lev["ERX"], cfg["erx_abs_max"])

    exec_w = {a: 0.0 for a in EXEC_ASSETS}
    exec_w["TQQQ"] = min(target_lev["TQQQ"], sig.get("QQQM", 0.0))
    exec_w["UXI"] = min(target_lev["UXI"], sig.get("XLI", 0.0))
    exec_w["ERX"] = min(target_lev["ERX"], sig.get("XLE", 0.0))
    exec_w["QQQM"] = sig.get("QQQM", 0.0) - exec_w["TQQQ"]
    exec_w["XLI"] = sig.get("XLI", 0.0) - exec_w["UXI"]
    exec_w["XLE"] = sig.get("XLE", 0.0) - exec_w["ERX"]
    exec_w["XSOE"] = sig.get("XSOE", 0.0)
    exec_w["XLB"] = sig.get("XLB", 0.0)
    exec_w["BIL"] = sig.get("BIL", 0.0)
    total = sum(exec_w.values())
    if total > 0 and abs(total - 1.0) > 1e-8:
        exec_w = {a: w / total for a, w in exec_w.items()}

    details = dict(
        raw_budget=raw_budget,
        total_budget_before_vol=total_budget_before_vol,
        realized_vol_now=realized_vol_now,
        vol_scale=vol_scale,
        total_budget_before_conviction=total_budget_before_conviction,
        total_budget=total_budget,
        tech_state=tech_state,
        industrial_state=industrial_state,
        energy_state=energy_state,
        transition_instability_z=transition_z,
        opportunity_stability_z=opportunity_z,
        leader_flip_4=leader_flip_4,
        signal_entropy=entropy,
        danger_transition=danger_transition,
        score_gap_inferred=score_gap,
        instability_penalty=instability_penalty,
        opportunity_bonus=opportunity_bonus,
        conviction_scale=conviction_scale,
        score_TQQQ=scores["TQQQ"],
        score_UXI=scores["UXI"],
        score_ERX=scores["ERX"],
        target_TQQQ=target_lev["TQQQ"],
        target_UXI=target_lev["UXI"],
        target_ERX=target_lev["ERX"],
        allowed_TQQQ=tqqq_allowed,
        allowed_UXI=uxi_allowed,
        allowed_ERX=erx_allowed,
    )
    return exec_w, details

# =========================
# Backtest and diagnostics
# =========================

def run_backtest(rebalance: pd.DataFrame, prices: pd.DataFrame, cfg: dict, tx_cost: float):
    asset_returns = prices[PRICE_ASSETS].pct_change().fillna(0.0)
    realized_vol_blend = build_realized_vol(prices)
    dates = prices.index
    rebal_dates = [d for d in list(rebalance["aligned_date"]) if d in dates]
    if len(rebal_dates) < 2:
        raise ValueError("Not enough aligned rebalance dates.")
    port = pd.Series(index=dates, dtype=float)
    old_w = {a: 0.0 for a in EXEC_ASSETS}
    old_w["BIL"] = 1.0
    turnovers = []
    logs = []
    date_to_loc = {d: i for i, d in enumerate(dates)}
    max_loc = len(dates) - 1
    for i, d in enumerate(rebal_dates):
        loc = date_to_loc[d]
        row = rebalance.loc[rebalance["aligned_date"] == d].iloc[0]
        exec_w, details = build_dynamic_exec_weights(row, cfg, realized_vol_blend)
        turnover = compute_turnover(old_w, exec_w, EXEC_ASSETS)
        turnovers.append(turnover)
        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[max_loc]
        next_loc = date_to_loc.get(next_d, max_loc)
        hold_dates = dates[loc + 1: next_loc + 1]
        if len(hold_dates) == 0:
            old_w = exec_w.copy()
            continue
        r = pd.Series(0.0, index=hold_dates)
        for a, w in exec_w.items():
            if abs(w) > 1e-12:
                r = r.add(w * asset_returns[a].reindex(hold_dates).fillna(0.0), fill_value=0.0)
        r.iloc[0] -= turnover * tx_cost
        port.loc[hold_dates] = r.values
        log = {"date": d, "turnover": turnover, **details}
        for a in EXEC_ASSETS:
            log[f"exec_w_{a}"] = exec_w.get(a, 0.0)
        logs.append(log)
        old_w = exec_w.copy()
    return port.dropna(), pd.DataFrame(logs), float(np.mean(turnovers)) if turnovers else np.nan


def stress_summary(returns: pd.Series) -> pd.DataFrame:
    windows = {
        "2018_Q4": ("2018-10-01", "2018-12-31"),
        "COVID_2020": ("2020-02-19", "2020-04-30"),
        "BEAR_2022": ("2022-01-01", "2022-12-31"),
        "AI_RALLY_2023_2024": ("2023-01-01", "2024-12-31"),
        "RECENT_2025_2026": ("2025-01-01", "2026-12-31"),
    }
    rows = []
    for name, (s, e) in windows.items():
        r = returns.loc[(returns.index >= pd.Timestamp(s)) & (returns.index <= pd.Timestamp(e))]
        if len(r) == 0:
            continue
        rows.append(summary(name, r))
    return pd.DataFrame(rows)


def activation_summary(log_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in EXEC_ASSETS:
        col = f"exec_w_{a}"
        if col not in log_df.columns:
            continue
        s = log_df[col].fillna(0.0)
        active = s > 1e-8
        rows.append({
            "asset": a,
            "active_rebalances": int(active.sum()),
            "active_pct": float(active.mean()) if len(s) else np.nan,
            "avg_weight_all": float(s.mean()) if len(s) else np.nan,
            "avg_weight_when_active": float(s[active].mean()) if active.any() else 0.0,
            "max_weight": float(s.max()) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def rebalance_interval_summary(log_df: pd.DataFrame) -> dict:
    if len(log_df) < 2:
        return {}
    d = pd.to_datetime(log_df["date"]).sort_values()
    gaps = d.diff().dt.days.dropna()
    return {
        "rebalances": len(d),
        "median_calendar_gap_days": float(gaps.median()),
        "mean_calendar_gap_days": float(gaps.mean()),
        "min_calendar_gap_days": float(gaps.min()),
        "max_calendar_gap_days": float(gaps.max()),
    }


def main():
    print("\n=== 011 LIGHT EXECUTION + RIGOROUS TEST ===")
    print("Working directory:", os.getcwd())
    print("Script folder:", SCRIPT_DIR)

    cfg, cfg_msg = load_best_cfg()
    print(cfg_msg)

    rebalance_raw = load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    prices = download_prices(start)
    print("Downloaded prices:", prices.index.min().date(), "to", prices.index.max().date())

    rebalance = align_rebalance_dates(rebalance_raw, prices)
    rebalance = add_transition_features(rebalance)
    realized_vol_blend = build_realized_vol(prices)

    # ============================================================
    # TRUE LIVE LATEST 011 LIGHT RECOMMENDATION
    # Do NOT use rebalance.iloc[-1] here.
    # That is historical/backtest rebalance row and can be stale.
    # ============================================================

    current_best_latest_path = SCRIPT_DIR / f"{CURRENT_PREFIX}_latest_recommendation.csv"
    if not current_best_latest_path.exists():
        raise FileNotFoundError(f"Missing live latest input: {current_best_latest_path}")

    live_df = pd.read_csv(current_best_latest_path)
    if live_df.empty:
        raise ValueError(f"Empty live latest input: {current_best_latest_path}")

    live_input_raw = live_df.iloc[-1].copy()

    if "signal_date" in live_input_raw.index:
        live_date = pd.to_datetime(live_input_raw["signal_date"])
    elif "latest_data_date" in live_input_raw.index:
        live_date = pd.to_datetime(live_input_raw["latest_data_date"])
    elif "date" in live_input_raw.index:
        live_date = pd.to_datetime(live_input_raw["date"])
    else:
        raise ValueError("Live latest file has no signal_date/latest_data_date/date column.")

    live_input_raw["date"] = live_date

    combined = pd.concat(
        [rebalance_raw.copy(), pd.DataFrame([live_input_raw])],
        ignore_index=True,
        sort=False,
    )

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = align_rebalance_dates(combined, prices)
    combined = add_transition_features(combined)

    live_candidates = combined[combined["aligned_date"] == live_date]
    if live_candidates.empty:
        live_candidates = combined[combined["aligned_date"] <= live_date]

    if live_candidates.empty:
        raise ValueError(f"Could not create live 011 row for {live_date.date()}")

    live_row = live_candidates.iloc[-1]

    exec_w, details = build_dynamic_exec_weights(live_row, cfg, realized_vol_blend)

    latest_out = {
        "model": "TRANSITION_CONVICTION_011_LIGHT_EXECUTION",
        "signal_date": live_date,
        "latest_data_date": live_input_raw.get("latest_data_date", live_date),
        "source_input": f"{CURRENT_PREFIX}_latest_recommendation.csv",
        **details,
    }

    for col in [
        "top_asset", "second_asset", "top_score", "second_score", "score_gap",
        "overlay_fraction",
        "growth_strength", "soxx_strength", "risk_off_strength",
        "war_strength", "industrial_strength", "materials_strength",
        "copper_strength", "copper_3m_strength", "usd_3m_strength",
        "credit_strength", "tech_real_economy_divergence",
        "crash_pressure", "breakdown_score",
        "adj_pred_QQQM", "adj_pred_XLE", "adj_pred_XSOE", "adj_pred_XLI", "adj_pred_XLB",
        "raw_pred_QQQM", "raw_pred_XLE", "raw_pred_XSOE", "raw_pred_XLI", "raw_pred_XLB",
    ]:
        if col in live_row.index and pd.notna(live_row[col]):
            latest_out[col] = live_row[col]

    for a in EXEC_ASSETS:
        latest_out[f"exec_w_{a}"] = exec_w.get(a, 0.0)

    for a in BASE_ASSETS:
        latest_out[f"signal_w_{a}"] = get_signal_weights(live_row).get(a, 0.0)

    latest_df = pd.DataFrame([latest_out])
    latest_df.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_latest_recommendation.csv", index=False)

    print("\nDEBUG LATEST OUT")
    print("TOP ASSET:", latest_out.get("top_asset"))
    print("SECOND ASSET:", latest_out.get("second_asset"))
    print("TOP SCORE:", latest_out.get("top_score"))
    print("SECOND SCORE:", latest_out.get("second_score"))
    print("SCORE GAP:", latest_out.get("score_gap"))
    print("OVERLAY FRACTION:", latest_out.get("overlay_fraction"))
    print("\nLATEST 011 LIGHT LIVE EXECUTION PORTFOLIO")
    print("Signal date:", pd.to_datetime(latest_out["signal_date"]).date())
    print("Latest data:", pd.to_datetime(latest_out["latest_data_date"]).date())
    for a in EXEC_ASSETS:
        w = exec_w.get(a, 0.0)
        if abs(w) > 1e-6:
            print(f"{a}: {w*100:.1f}%")
    print("Total leveraged:", sum(exec_w.get(a, 0.0) for a in ["TQQQ", "UXI", "ERX"]) * 100)

    # Main backtest with production cost
    main_ret, main_log, avg_turn = run_backtest(rebalance, prices, cfg, DEFAULT_TRANSACTION_COST)
    stats = [summary("011_LIGHT_EXECUTION_COST_0.001", main_ret, avg_turn, len(main_log), DEFAULT_TRANSACTION_COST)]

    # Cost sensitivity
    for c in [0.0, 0.0005, 0.001, 0.002, 0.003]:
        ret_c, log_c, avg_c = run_backtest(rebalance, prices, cfg, c)
        stats.append(summary(f"cost_sensitivity_{c:.4f}", ret_c, avg_c, len(log_c), c))

    stats_df = pd.DataFrame(stats)
    stress_df = stress_summary(main_ret)
    act_df = activation_summary(main_log)
    interval_df = pd.DataFrame([rebalance_interval_summary(main_log)])

    # Compare current-best returns if available
    current_returns = load_current_returns_optional()
    compare_df = pd.DataFrame()
    if current_returns is not None:
        common = current_returns.index.intersection(main_ret.index)
        if len(common) > 20:
            cur = current_returns.loc[common]
            new = main_ret.loc[common]
            compare_df = pd.DataFrame([
                summary("CURRENT_BEST_CSV_COMMON", cur, trade_count=len(rebalance)),
                summary("011_LIGHT_EXECUTION_COMMON", new, avg_turn, len(main_log), DEFAULT_TRANSACTION_COST),
            ])
            delta = {k: compare_df.iloc[1][k] for k in compare_df.columns}
            delta["model"] = "DELTA_011_LIGHT_MINUS_CURRENT"
            for k in ["annual_return", "volatility", "sharpe", "max_drawdown", "final_equity"]:
                delta[k] = compare_df.iloc[1][k] - compare_df.iloc[0][k]
            compare_df = pd.concat([compare_df, pd.DataFrame([delta])], ignore_index=True)

    # Save outputs
    stats_df.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_rigorous_stats.csv", index=False)
    main_ret.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_daily_returns.csv", header=["portfolio_return"])
    main_log.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_rebalance_log.csv", index=False)
    stress_df.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_stress_summary.csv", index=False)
    act_df.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_activation_summary.csv", index=False)
    interval_df.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_rebalance_interval_summary.csv", index=False)
    if not compare_df.empty:
        compare_df.to_csv(SCRIPT_DIR / f"{PREFIX_OUT}_compare_current_best.csv", index=False)

    print("\n=== MAIN RIGOROUS RESULT ===")
    print(stats_df.head(1).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== COST SENSITIVITY ===")
    print(stats_df.iloc[1:].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not compare_df.empty:
        print("\n=== COMPARE WITH CURRENT BEST CSV ===")
        print(compare_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== STRESS SUMMARY ===")
    print(stress_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== ACTIVATION SUMMARY ===")
    print(act_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== REBALANCE INTERVAL SUMMARY ===")
    print(interval_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nSaved files:")
    for f in [
        f"{PREFIX_OUT}_latest_recommendation.csv",
        GENERIC_LATEST,
        f"{PREFIX_OUT}_rigorous_stats.csv",
        f"{PREFIX_OUT}_daily_returns.csv",
        f"{PREFIX_OUT}_rebalance_log.csv",
        f"{PREFIX_OUT}_stress_summary.csv",
        f"{PREFIX_OUT}_activation_summary.csv",
        f"{PREFIX_OUT}_rebalance_interval_summary.csv",
        f"{PREFIX_OUT}_compare_current_best.csv",
    ]:
        print("-", f)


if __name__ == "__main__":
    main()
