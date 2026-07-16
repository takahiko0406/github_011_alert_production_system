#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

PREFIX = "model_c_plus_034_execution_grade_expected_return_signal"
LATEST_FILE = f"{PREFIX}_latest_recommendation.csv"
SCOREBOARD_FILE = f"{PREFIX}_scoreboard.csv"
PERFORMANCE_FILE = f"{PREFIX}_performance_summary.csv"
VALIDATION_FILE = "model_c_plus_034_live_dashboard_validation.json"
MARKET_FRESHNESS_FILE = "model_c_plus_market_data_freshness.csv"
FEATURE_FRESHNESS_FILE = "model_c_plus_feature_freshness.csv"
CURRENT_BEST_FILE = "model_c_plus_current_best_with_divergence_alerts_latest_recommendation.csv"
OUT_FILES = [f"model_c_plus_034_live_dashboard_telegram_page_{i}.txt" for i in range(1, 5)]
OUT_ALL = "model_c_plus_034_live_dashboard_telegram_preview.txt"


def read_last_csv(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=object)
    df = pd.read_csv(path)
    return pd.Series(dtype=object) if df.empty else df.iloc[-1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def freshness_status(path: Path, expected_date: str) -> str:
    row = read_last_csv(path)
    if row.empty:
        return "MISSING"
    actual = date_text(first([row], ["latest_data_date", "data_date", "signal_date"]))
    status = str(first([row], ["status", "freshness", "data_status"]) or "UNKNOWN").upper()
    if actual == expected_date and status in {"PASS", "CURRENT", "OK"}:
        return f"CURRENT ({actual})"
    if actual == expected_date:
        return f"{status} ({actual})"
    return f"STALE ({actual}; expected {expected_date})"


def number(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def first(sources: list[Any], names: list[str]) -> Any:
    for source in sources:
        if source is None:
            continue
        for name in names:
            try:
                if isinstance(source, dict) and name in source:
                    value = source[name]
                elif hasattr(source, "index") and name in source.index:
                    value = source[name]
                else:
                    continue
                if pd.isna(value):
                    continue
                return value
            except Exception:
                continue
    return None


def pct(value: Any, decimals: int = 2) -> str:
    x = number(value)
    return "N/A" if x is None else f"{x * 100:.{decimals}f}%"


def num(value: Any, decimals: int = 3) -> str:
    x = number(value)
    return "N/A" if x is None else f"{x:.{decimals}f}"


def date_text(value: Any) -> str:
    return "N/A" if value is None else str(value)[:10]


def yes_no(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "active", "on", "pass", "current"}:
            return "YES"
        if text in {"false", "no", "0", "inactive", "off", "fail", "stale"}:
            return "NO"
        return value
    return "YES" if bool(value) else "NO"


def section(title: str) -> list[str]:
    return ["", f"━━ {title} ━━"]


def state(value: Any, positive: str, neutral: str, negative: str) -> str:
    x = number(value)
    if x is None:
        return "Unavailable"
    if x >= 0.50:
        return positive
    if x <= -0.50:
        return negative
    return neutral


def choose(columns: pd.Index, names: list[str]) -> str | None:
    return next((name for name in names if name in columns), None)


def extract_weights(row: pd.Series) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in row.index:
        name = str(column)
        if name.startswith("exec_w_"):
            value = number(row[column])
            if value is not None and value > 1e-10:
                result[name[7:]] = value
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


def normalize_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    asset_col = choose(df.columns, ["asset", "ETF", "ticker"])
    if asset_col is None:
        return pd.DataFrame()
    expected_col = choose(df.columns, ["adjusted_expected_10d_return", "expected_10d_return", "raw_expected_10d_return", "predicted_10d_return"])
    tradable_col = choose(df.columns, ["tradable_score_0_100", "execution_score", "xscore"])
    model_col = choose(df.columns, ["selected_model_score", "execution_model_score", "model_score"])
    weight_col = choose(df.columns, ["selected_exec_weight", "exec_weight", "target_weight"])
    authority_col = choose(df.columns, ["authority", "execution_authority"])
    return pd.DataFrame({
        "asset": df[asset_col].astype(str),
        "expected_return": pd.to_numeric(df[expected_col], errors="coerce") if expected_col else float("nan"),
        "tradable_score": pd.to_numeric(df[tradable_col], errors="coerce") if tradable_col else float("nan"),
        "model_score": pd.to_numeric(df[model_col], errors="coerce") if model_col else float("nan"),
        "weight": pd.to_numeric(df[weight_col], errors="coerce") if weight_col else 0.0,
        "authority": df[authority_col].astype(str) if authority_col else "N/A",
    }).drop_duplicates("asset", keep="first")


def authority_code(value: Any) -> str:
    text = str(value)
    if "LIVE_EXECUTION" in text:
        return "LIVE"
    if "OPPORTUNISTIC" in text:
        return "SHADOW"
    if "ROBUST" in text:
        return "GATED"
    if "RESEARCH" in text:
        return "RESEARCH"
    return "N/A"


def row_for(scoreboard: pd.DataFrame, asset: str) -> pd.Series | None:
    match = scoreboard[scoreboard["asset"] == asset]
    return None if match.empty else match.iloc[0]


def strength_text(scoreboard: pd.DataFrame, asset: str) -> str:
    row = row_for(scoreboard, asset)
    if row is None:
        return "Unavailable"
    score = number(row["tradable_score"])
    expected = number(row["expected_return"])
    if score is None:
        return f"Exp10D {pct(expected)}"
    label = "Strong" if score >= 75 else "Constructive" if score >= 60 else "Mixed" if score >= 40 else "Weak"
    return f"{label} | score {score:.1f} | Exp10D {pct(expected)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", dest="input_dir", type=Path)
    args = parser.parse_args()

    env_root = os.getenv("DASHBOARD_INPUT_DIR", "").strip()
    root = (args.input_dir or (Path(env_root) if env_root else args.root)).resolve()

    latest = read_last_csv(root / LATEST_FILE)
    if latest.empty:
        raise FileNotFoundError(LATEST_FILE)
    scoreboard = normalize_scoreboard(pd.read_csv(root / SCOREBOARD_FILE))
    if scoreboard.empty:
        raise ValueError("Scoreboard columns could not be recognized")

    performance = pd.read_csv(root / PERFORMANCE_FILE) if (root / PERFORMANCE_FILE).exists() else pd.DataFrame()
    performance_row = pd.Series(dtype=object)
    if not performance.empty:
        matched = performance[performance["model"].astype(str).str.contains("034", case=False, na=False)]
        performance_row = matched.iloc[0] if not matched.empty else performance.iloc[0]

    validation = read_json(root / VALIDATION_FILE)
    feature_source = read_last_csv(root / CURRENT_BEST_FILE)
    data_date = date_text(first([latest, validation, feature_source], ["latest_data_date", "data_date"]))
    allocation_date = date_text(first([latest, validation], ["allocation_date", "base_weight_date"]))
    market_freshness = freshness_status(root / MARKET_FRESHNESS_FILE, data_date)
    feature_freshness = freshness_status(root / FEATURE_FRESHNESS_FILE, data_date)
    allocation_freshness = "CURRENT" if allocation_date == data_date else f"STALE ({allocation_date}; expected {data_date})"
    freshness = "CURRENT" if market_freshness.startswith("CURRENT") and feature_freshness.startswith("CURRENT") and allocation_freshness == "CURRENT" else "CHECK"

    weights = extract_weights(latest)
    # The verified allocation is authoritative. Override any stale/intermediate scoreboard weights.
    scoreboard["weight"] = scoreboard["asset"].map(weights).fillna(0.0)

    live_ranked = scoreboard[scoreboard["weight"] > 1e-10].sort_values(["weight", "model_score"], ascending=[False, False])
    live_top_asset = live_ranked.iloc[0]["asset"] if not live_ranked.empty else "N/A"
    live_second_asset = live_ranked.iloc[1]["asset"] if len(live_ranked) > 1 else "N/A"

    research_ranked = scoreboard.sort_values(["tradable_score", "expected_return"], ascending=[False, False], na_position="last")
    research_top_asset = research_ranked.iloc[0]["asset"] if not research_ranked.empty else "N/A"
    research_second_asset = research_ranked.iloc[1]["asset"] if len(research_ranked) > 1 else "N/A"

    top_asset = first([latest], ["top_asset", "expanded_top_asset"]) or research_top_asset
    second_asset = first([latest], ["second_asset", "expanded_second_asset"]) or research_second_asset
    score_gap = first([latest], ["score_gap", "expanded_score_gap"])
    expected_portfolio_return = float((scoreboard["expected_return"].fillna(0) * scoreboard["weight"].fillna(0)).sum())

    page1 = ["034 DAILY EXECUTION", f"Data date: {data_date}", f"Allocation date: {allocation_date}", f"Freshness: {freshness}"]
    page1 += section("VERIFIED ALLOCATION")
    for asset, weight in weights.items():
        page1.append(f"{asset:<6} {weight * 100:>6.1f}%")
    page1.append(f"{'TOTAL':<6} {sum(weights.values()) * 100:>6.1f}%")
    page1 += section("EXECUTION STATE")
    page1.extend([
        f"Execution safe: {yes_no(first([validation], ['execution_safe']))}",
        f"Emergency state: {first([latest], ['emergency_status', 'emergency_alert', 'emergency']) or 'N/A'}",
        f"Normal rebalance due: {yes_no(first([latest], ['normal_rebalance_due', 'rebalance_due']))}",
        f"Live allocation leader: {live_top_asset}",
        f"Second live holding: {live_second_asset}",
        f"Research strength leader: {research_top_asset}",
        f"Top-two score gap: {num(score_gap, 4)}",
        f"Required gap: {num(first([latest], ['required_gap', 'score_gap_required']), 4)}",
        f"Effective technology exposure: {num(first([latest], ['effective_technology_exposure']))}x",
    ])
    page1 += section("PORTFOLIO EVIDENCE")
    page1.extend([
        f"Expected 10-day portfolio return: {pct(expected_portfolio_return)}",
        f"Historical annual return: {pct(first([performance_row], ['annual_return']))}",
        f"Historical volatility: {pct(first([performance_row], ['volatility']))}",
        f"Historical Sharpe: {num(first([performance_row], ['sharpe']))}",
        f"Historical max drawdown: {pct(first([performance_row], ['max_drawdown']))}",
    ])

    ranked = scoreboard.sort_values(["weight", "tradable_score", "expected_return", "asset"], ascending=[False, False, False, True], na_position="last").reset_index(drop=True)
    page2 = ["034 COMPLETE ETF STRENGTH RANKING", f"Data date: {data_date} | Freshness: {freshness}", "", "#  ETF    Wt   Exp10D  Strength  Model    Authority", "---------------------------------------------------"]
    for index, row in ranked.iterrows():
        strength = "N/A" if pd.isna(row["tradable_score"]) else f"{row['tradable_score']:.1f}"
        model_score = "N/A" if pd.isna(row["model_score"]) else f"{row['model_score']:.4f}"
        page2.append(f"{index + 1:>2} {row['asset']:<5} {pct(row['weight'], 0):>4} {pct(row['expected_return']):>8} {strength:>8} {model_score:>8} {authority_code(row['authority'])}")
    page2 += section("HOW TO READ")
    page2.extend([
        "Weight is the verified execution allocation.",
        "Exp10D is the expected return forecast for the next 10 trading days.",
        "Strength is the 0-100 tradable score.",
        "Model is the execution-model score and uses a different scale.",
        "LIVE may execute. GATED, SHADOW and RESEARCH do not execute unless approved.",
    ])

    growth = first([latest, feature_source], ["growth_strength", "expanded_growth_strength"])
    semis = first([latest, feature_source], ["soxx_strength"])
    credit = first([latest, feature_source], ["credit_strength"])
    industrials = first([latest, feature_source], ["industrial_strength", "expanded_industrial_strength"])
    materials = first([latest, feature_source], ["materials_strength", "expanded_materials_strength"])
    copper = first([latest, feature_source], ["copper_strength"])
    oil = first([latest, feature_source], ["oil_strength", "oil_3m"])
    usd = first([latest, feature_source], ["usd_3m_strength"])
    risk_off = first([latest, feature_source], ["risk_off_strength"])
    crash = first([latest, feature_source], ["crash_pressure", "expanded_crash_pressure"])
    vix = first([latest, feature_source], ["vix_level", "vix"])
    yield_curve = first([latest, feature_source], ["yield_curve"])

    page3 = ["034 ECONOMIC AND MARKET REGIME", f"Data date: {data_date} | Freshness: {freshness}"]
    page3 += section("CORE ECONOMIC STATE")
    page3.extend([
        f"Growth: {state(growth, 'Strong', 'Mixed', 'Weak')} ({num(growth)})",
        f"Credit: {state(credit, 'Supportive', 'Neutral', 'Deteriorating')} ({num(credit)})",
        f"USD: {state(usd, 'Strong / tightening', 'Neutral', 'Weak / easing')} ({num(usd)})",
        f"Yield curve: {num(yield_curve)}",
        f"Risk-off pressure: {state(risk_off, 'Elevated', 'Moderate', 'Low')} ({num(risk_off)})",
        f"Crash pressure: {state(crash, 'Elevated', 'Moderate', 'Low')} ({num(crash)})",
        f"VIX: {num(vix)}",
    ])
    page3 += section("SECTOR AND CROSS-ASSET REGIME")
    page3.extend([
        f"Semiconductors / technology: {state(semis, 'Leading', 'Transitioning', 'Weak')} ({num(semis)})",
        f"Industrials: {state(industrials, 'Strong', 'Mixed', 'Weak')} ({num(industrials)})",
        f"Materials: {state(materials, 'Strong', 'Mixed', 'Weak')} ({num(materials)})",
        f"Copper: {state(copper, 'Supportive', 'Neutral', 'Weak')} ({num(copper)})",
        f"Oil / energy: {state(oil, 'Strong', 'Neutral', 'Weak')} ({num(oil)})",
        f"Small caps — IWM: {strength_text(scoreboard, 'IWM')}",
        f"Financials — XLF: {strength_text(scoreboard, 'XLF')}",
        f"Real estate — XLRE: {strength_text(scoreboard, 'XLRE')}",
        f"Healthcare — XLV: {strength_text(scoreboard, 'XLV')}",
        f"Utilities — XLU: {strength_text(scoreboard, 'XLU')}",
        f"Staples — XLP: {strength_text(scoreboard, 'XLP')}",
        f"Treasuries — TLT: {strength_text(scoreboard, 'TLT')}",
        f"Gold — GLD: {strength_text(scoreboard, 'GLD')}",
    ])
    page3 += section("GLOBAL AND EM REGIME")
    page3.extend([
        f"Europe — FEZ: {strength_text(scoreboard, 'FEZ')}",
        f"Emerging markets — XSOE: {strength_text(scoreboard, 'XSOE')}",
        "China technology — CQQQ: NOT YET ACTIVE",
        "Latin America — ILF: NOT YET ACTIVE",
        "USD/CNY and USD/CNH regime: NOT YET ACTIVE",
        "Korean won / Asian export-cycle regime: NOT YET ACTIVE",
    ])
    page3 += section("ECONOMIC INTERPRETATION")
    page3.append(f"Verified live allocation is led by {live_top_asset}; {live_second_asset} is the next live holding.")
    if research_top_asset != live_top_asset:
        authority = row_for(scoreboard, research_top_asset)
        authority_text = authority_code(authority["authority"]) if authority is not None else "RESEARCH"
        page3.append(f"{research_top_asset} is the strongest research signal but has {authority_text} authority, so it does not replace the verified live allocation.")
    gap_value = number(score_gap)
    page3.append("The top-two gap is small, so leadership conviction is limited rather than dominant." if gap_value is not None and abs(gap_value) < 0.005 else "The top-two gap indicates comparatively clear leadership.")
    if number(credit) is not None and number(credit) > 0:
        page3.append("Credit remains supportive, reducing immediate stress risk.")
    if number(usd) is not None and number(usd) > 0.5:
        page3.append("The strong-dollar regime remains a headwind for broad EM and international risk assets.")
    if number(semis) is not None and number(semis) < 0:
        page3.append("The slow semiconductor regime remains weak even if short-term price action improves.")
    page3.append("Real estate, small caps, defensives, bonds, gold, Europe and EM remain visible as economic confirmation signals even when their target weight is zero.")

    page4 = ["034 MODEL HEALTH AND DIAGNOSTICS", f"Data date: {data_date} | Freshness: {freshness}"]
    page4 += section("TECHNOLOGY AND LEVERAGE")
    page4.extend([
        f"QQQM base weight: {pct(first([latest], ['qqqm_base_weight']))}",
        f"TQQQ substituted weight: {pct(first([latest], ['tqqq_substituted_weight']))}",
        f"SOXX base weight: {pct(first([latest], ['soxx_base_weight']))}",
        f"SOXL substituted weight: {pct(first([latest], ['soxl_substituted_weight']))}",
        f"Common leverage budget: {pct(first([latest], ['common_leverage_budget']))}",
        f"Effective technology exposure: {num(first([latest], ['effective_technology_exposure']))}x",
        f"Leverage configuration: {first([latest], ['common_leverage_configuration']) or 'N/A'}",
        f"Robust gate: {yes_no(first([latest], ['robust_gate_active', 'robust_gate']))}",
        f"Opportunistic gate: {yes_no(first([latest], ['opportunistic_gate_active', 'opportunistic_gate']))}",
    ])
    page4 += section("FRESHNESS AND VALIDATION")
    assertions = validation.get("assertions", {})
    page4.extend([
        f"Execution safe: {yes_no(first([validation], ['execution_safe']))}",
        f"Validation: {'PASS' if assertions and all(assertions.values()) else 'N/A'}",
        f"Model: {first([latest], ['model']) or 'N/A'}",
        f"Market-data freshness: {market_freshness}",
        f"Feature freshness: {feature_freshness}",
        f"Allocation freshness: {allocation_freshness} ({allocation_date})",
        f"Ranking fingerprint: {first([validation], ['ranking_fingerprint']) or 'N/A'}",
    ])
    page4 += section("EXECUTION AUTHORITY")
    page4.extend([
        "Live execution: QQQM, XLE, XSOE, XLI, XLB",
        "Leveraged substitutes: TQQQ, SOXL, ERX, UXI",
        "Expanded/research: SOXX, IWM, FEZ, XLF, XLV, XLP, XLU, XLRE, TLT, IEF, GLD",
    ])
    page4 += section("UNAVAILABLE / FUTURE EXTENSIONS")
    page4.extend([
        "CQQQ, ILF, USD/CNY, USD/CNH and KRW are explicitly marked NOT YET ACTIVE.",
        "They are not omitted, fabricated or used in today's allocation.",
        "They can be added later without rebuilding this four-page report.",
    ])

    pages = [page1, page2, page3, page4]
    texts = ["\n".join(page).strip() + "\n" for page in pages]
    for filename, text in zip(OUT_FILES, texts):
        (root / filename).write_text(text, encoding="utf-8")
    (root / OUT_ALL).write_text("\n\n".join(texts), encoding="utf-8")
    print("Saved four-page high-information Telegram macro briefing.")
    print("No model logic, weights, scores, gates, leverage or rebalance rules changed.")


if __name__ == "__main__":
    main()
