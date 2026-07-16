#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PREFIX = "model_c_plus_034_execution_grade_expected_return_signal"
LATEST_FILE = f"{PREFIX}_latest_recommendation.csv"
SCOREBOARD_FILE = f"{PREFIX}_scoreboard.csv"
PERFORMANCE_FILE = f"{PREFIX}_performance_summary.csv"
VALIDATION_FILE = "model_c_plus_034_live_dashboard_validation.json"

OUT_FILES = [
    "model_c_plus_034_live_dashboard_telegram_page_1.txt",
    "model_c_plus_034_live_dashboard_telegram_page_2.txt",
    "model_c_plus_034_live_dashboard_telegram_page_3.txt",
    "model_c_plus_034_live_dashboard_telegram_page_4.txt",
]
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


def to_number(value: Any) -> float | None:
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


def fmt_pct(value: Any, decimals: int = 2) -> str:
    x = to_number(value)
    return "N/A" if x is None else f"{x * 100:.{decimals}f}%"


def fmt_num(value: Any, decimals: int = 3) -> str:
    x = to_number(value)
    return "N/A" if x is None else f"{x:.{decimals}f}"


def fmt_date(value: Any) -> str:
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


def extract_weights(row: pd.Series) -> dict[str, float]:
    weights: dict[str, float] = {}
    for col in row.index:
        name = str(col)
        if name.startswith("exec_w_"):
            value = to_number(row[col])
            if value is not None and value > 1e-10:
                weights[name[7:]] = value
    return dict(sorted(weights.items(), key=lambda item: (-item[1], item[0])))


def normalize_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    asset_col = next((c for c in ["asset", "ETF", "ticker"] if c in df.columns), None)
    expected_col = next(
        (
            c
            for c in [
                "adjusted_expected_10d_return",
                "expected_10d_return",
                "raw_expected_10d_return",
                "predicted_10d_return",
            ]
            if c in df.columns
        ),
        None,
    )
    manual_col = next(
        (
            c
            for c in [
                "manual_action_score_0_100",
                "action_score_0_100",
                "manual_score_0_100",
            ]
            if c in df.columns
        ),
        None,
    )
    tradable_col = next(
        (
            c
            for c in [
                "tradable_score_0_100",
                "execution_score",
                "xscore",
            ]
            if c in df.columns
        ),
        None,
    )
    model_col = next(
        (
            c
            for c in [
                "selected_model_score",
                "execution_model_score",
                "model_score",
            ]
            if c in df.columns
        ),
        None,
    )
    weight_col = next(
        (
            c
            for c in [
                "selected_exec_weight",
                "exec_weight",
                "target_weight",
            ]
            if c in df.columns
        ),
        None,
    )
    authority_col = next((c for c in ["authority", "execution_authority"] if c in df.columns), None)
    execution_ok_col = next((c for c in ["execution_ok", "executable"] if c in df.columns), None)
    reason_col = next((c for c in ["reason", "selection_reason"] if c in df.columns), None)

    if asset_col is None:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "asset": df[asset_col].astype(str),
            "expected_return": pd.to_numeric(df[expected_col], errors="coerce")
            if expected_col
            else float("nan"),
            "manual_score": pd.to_numeric(df[manual_col], errors="coerce")
            if manual_col
            else float("nan"),
            "tradable_score": pd.to_numeric(df[tradable_col], errors="coerce")
            if tradable_col
            else float("nan"),
            "model_score": pd.to_numeric(df[model_col], errors="coerce")
            if model_col
            else float("nan"),
            "weight": pd.to_numeric(df[weight_col], errors="coerce")
            if weight_col
            else 0.0,
            "authority": df[authority_col].astype(str) if authority_col else "N/A",
            "execution_ok": df[execution_ok_col] if execution_ok_col else False,
            "reason": df[reason_col].astype(str) if reason_col else "N/A",
        }
    )
    return out.drop_duplicates("asset", keep="first")


def regime_label(value: Any, positive: str, neutral: str, negative: str) -> str:
    x = to_number(value)
    if x is None:
        return "Unavailable"
    if x >= 0.50:
        return positive
    if x <= -0.50:
        return negative
    return neutral


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-dir", dest="input_dir", type=Path)
    args = parser.parse_args()

    root = (args.input_dir or args.root).resolve()

    latest = read_last_csv(root / LATEST_FILE)
    if latest.empty:
        raise FileNotFoundError(LATEST_FILE)

    scoreboard_path = root / SCOREBOARD_FILE
    if not scoreboard_path.exists():
        raise FileNotFoundError(SCOREBOARD_FILE)

    scoreboard = normalize_scoreboard(pd.read_csv(scoreboard_path))
    if scoreboard.empty:
        raise ValueError("Scoreboard columns could not be recognized")

    performance = (
        pd.read_csv(root / PERFORMANCE_FILE)
        if (root / PERFORMANCE_FILE).exists()
        else pd.DataFrame()
    )
    performance_row = pd.Series(dtype=object)
    if not performance.empty:
        matched = performance[
            performance["model"].astype(str).str.contains("034", case=False, na=False)
        ]
        performance_row = matched.iloc[0] if not matched.empty else performance.iloc[0]

    validation = read_json(root / VALIDATION_FILE)

    data_date = fmt_date(first([latest, validation], ["latest_data_date", "data_date"]))
    allocation_date = fmt_date(
        first([latest, validation], ["allocation_date", "base_weight_date"])
    )
    freshness = (
        first(
            [latest, validation],
            ["freshness", "data_status", "freshness_status"],
        )
        or "N/A"
    )

    target_weights = extract_weights(latest)
    expected_portfolio_return = float(
        (
            scoreboard["expected_return"].fillna(0.0)
            * scoreboard["weight"].fillna(0.0)
        ).sum()
    )

    top_asset = first([latest], ["top_asset", "expanded_top_asset"]) or "N/A"
    second_asset = first([latest], ["second_asset", "expanded_second_asset"]) or "N/A"
    score_gap = first([latest], ["score_gap", "expanded_score_gap"])

    # PAGE 1
    page1 = [
        "034 DAILY EXECUTION",
        f"Data date: {data_date}",
        f"Allocation date: {allocation_date}",
        f"Freshness: {freshness}",
    ]

    page1 += section("VERIFIED ALLOCATION")
    if target_weights:
        for asset, weight in target_weights.items():
            page1.append(f"{asset:<6} {weight * 100:>6.1f}%")
        page1.append(f"{'TOTAL':<6} {sum(target_weights.values()) * 100:>6.1f}%")
    else:
        page1.append("N/A")

    page1 += section("EXECUTION STATE")
    page1.extend(
        [
            f"Execution safe: {yes_no(first([validation], ['execution_safe']))}",
            f"Emergency state: {first([latest], ['emergency_status', 'emergency_alert', 'emergency']) or 'N/A'}",
            f"Normal rebalance due: {yes_no(first([latest], ['normal_rebalance_due', 'rebalance_due']))}",
            f"Top signal: {top_asset}",
            f"Second signal: {second_asset}",
            f"Top-two score gap: {fmt_num(score_gap, 4)}",
            f"Required gap: {fmt_num(first([latest], ['required_gap', 'score_gap_required']), 4)}",
        ]
    )

    page1 += section("PORTFOLIO EVIDENCE")
    page1.extend(
        [
            f"Expected 10-day portfolio return: {fmt_pct(expected_portfolio_return)}",
            f"Historical annual return: {fmt_pct(first([performance_row], ['annual_return']))}",
            f"Historical volatility: {fmt_pct(first([performance_row], ['volatility']))}",
            f"Historical Sharpe: {fmt_num(first([performance_row], ['sharpe']))}",
            f"Historical max drawdown: {fmt_pct(first([performance_row], ['max_drawdown']))}",
        ]
    )

    # PAGE 2
    page2 = [
        "034 FULL ETF SCOREBOARD",
        f"Data date: {data_date}",
        f"Freshness: {freshness}",
        "",
        "ETF     Exp10D   Manual   Tradable   Model     Weight",
        "----------------------------------------------------",
    ]

    ranked = scoreboard.sort_values(
        ["weight", "expected_return", "asset"],
        ascending=[False, False, True],
        na_position="last",
    )

    for _, row in ranked.iterrows():
        manual_text = "N/A" if pd.isna(row["manual_score"]) else f"{row['manual_score']:.1f}"
        tradable_text = (
            "N/A" if pd.isna(row["tradable_score"]) else f"{row['tradable_score']:.1f}"
        )
        model_text = "N/A" if pd.isna(row["model_score"]) else f"{row['model_score']:.4f}"
        page2.append(
            f"{row['asset']:<6} "
            f"{fmt_pct(row['expected_return']):>8} "
            f"{manual_text:>8} "
            f"{tradable_text:>10} "
            f"{model_text:>8} "
            f"{fmt_pct(row['weight'], 1):>8}"
        )

    page2 += section("SCORE DEFINITIONS")
    page2.extend(
        [
            "Expected 10D return: forecast for the next 10 trading days.",
            "Manual score: action-oriented 0-100 score.",
            "Tradable score: execution-readiness score on its own scale.",
            "Model score: execution-ranking score on its own scale.",
            "These score columns are not directly comparable.",
        ]
    )

    # PAGE 3
    growth = first([latest], ["growth_strength", "expanded_growth_strength"])
    semis = first([latest], ["soxx_strength"])
    credit = first([latest], ["credit_strength"])
    industrials = first(
        [latest], ["industrial_strength", "expanded_industrial_strength"]
    )
    materials = first(
        [latest], ["materials_strength", "expanded_materials_strength"]
    )
    copper = first([latest], ["copper_strength"])
    oil = first([latest], ["oil_strength", "oil_3m"])
    usd = first([latest], ["usd_3m_strength"])
    risk_off = first([latest], ["risk_off_strength"])
    crash = first([latest], ["crash_pressure", "expanded_crash_pressure"])
    vix = first([latest], ["vix_level", "vix"])
    yield_curve = first([latest], ["yield_curve"])

    page3 = [
        "034 ECONOMIC REGIME",
        f"Data date: {data_date}",
        f"Freshness: {freshness}",
    ]

    page3 += section("REGIME STATES")
    page3.extend(
        [
            f"Growth: {regime_label(growth, 'Strong', 'Mixed', 'Weak')} ({fmt_num(growth)})",
            f"Semiconductors: {regime_label(semis, 'Leading', 'Transitioning', 'Weak')} ({fmt_num(semis)})",
            f"Credit: {regime_label(credit, 'Supportive', 'Neutral', 'Deteriorating')} ({fmt_num(credit)})",
            f"Industrials: {regime_label(industrials, 'Strong', 'Mixed', 'Weak')} ({fmt_num(industrials)})",
            f"Materials: {regime_label(materials, 'Strong', 'Mixed', 'Weak')} ({fmt_num(materials)})",
            f"Copper: {regime_label(copper, 'Supportive', 'Neutral', 'Weak')} ({fmt_num(copper)})",
            f"Oil: {regime_label(oil, 'Strong', 'Neutral', 'Weak')} ({fmt_num(oil)})",
            f"USD: {regime_label(usd, 'Strong / tightening', 'Neutral', 'Weak / easing')} ({fmt_num(usd)})",
            f"Risk-off: {regime_label(risk_off, 'Elevated', 'Moderate', 'Low')} ({fmt_num(risk_off)})",
            f"Crash pressure: {regime_label(crash, 'Elevated', 'Moderate', 'Low')} ({fmt_num(crash)})",
            f"VIX: {fmt_num(vix)}",
            f"Yield curve: {fmt_num(yield_curve)}",
        ]
    )

    page3 += section("ECONOMIC INTERPRETATION")
    page3.append(f"{top_asset} ranks first and {second_asset} ranks second.")
    gap_value = to_number(score_gap)
    if gap_value is not None and abs(gap_value) < 0.005:
        page3.append("Leadership conviction is limited because the top-two gap is small.")
    else:
        page3.append("Leadership is comparatively clear.")

    if to_number(semis) is not None and to_number(semis) < 0:
        page3.append(
            "The slow semiconductor regime remains weak despite any short-term rally."
        )
    elif to_number(semis) is not None:
        page3.append("Semiconductor leadership is improving or positive.")

    if to_number(credit) is not None and to_number(credit) > 0:
        page3.append("Credit remains supportive of risk assets.")

    if to_number(usd) is not None and to_number(usd) > 0.5:
        page3.append(
            "A strong USD remains a headwind for global and emerging-market assets."
        )

    # PAGE 4
    page4 = [
        "034 MODEL DIAGNOSTICS",
        f"Data date: {data_date}",
        f"Freshness: {freshness}",
    ]

    page4 += section("TECHNOLOGY AND LEVERAGE")
    page4.extend(
        [
            f"QQQM base weight: {fmt_pct(first([latest], ['qqqm_base_weight']))}",
            f"TQQQ substituted weight: {fmt_pct(first([latest], ['tqqq_substituted_weight']))}",
            f"SOXX base weight: {fmt_pct(first([latest], ['soxx_base_weight']))}",
            f"SOXL substituted weight: {fmt_pct(first([latest], ['soxl_substituted_weight']))}",
            f"Common leverage budget: {fmt_pct(first([latest], ['common_leverage_budget']))}",
            f"Effective technology exposure: {fmt_num(first([latest], ['effective_technology_exposure']))}",
            f"Leverage configuration: {first([latest], ['common_leverage_configuration']) or 'N/A'}",
            f"Robust gate: {yes_no(first([latest], ['robust_gate_active', 'robust_gate']))}",
            f"Opportunistic gate: {yes_no(first([latest], ['opportunistic_gate_active', 'opportunistic_gate']))}",
        ]
    )

    page4 += section("SYSTEM AND FRESHNESS")
    assertions = validation.get("assertions", {})
    page4.extend(
        [
            f"Execution safe: {yes_no(first([validation], ['execution_safe']))}",
            f"Validation: {'PASS' if assertions and all(assertions.values()) else 'N/A'}",
            f"Model: {first([latest], ['model']) or 'N/A'}",
            f"Market-data freshness: {first([validation], ['market_data_freshness']) or freshness}",
            f"Feature freshness: {first([validation], ['feature_freshness']) or freshness}",
            f"Allocation freshness: {first([validation], ['allocation_freshness']) or freshness}",
        ]
    )

    page4 += section("IMPORTANT UNSELECTED ASSETS")
    important_assets = [
        "SOXX",
        "XSOE",
        "TLT",
        "IEF",
        "GLD",
        "XLV",
        "XLU",
        "XLP",
        "XLRE",
        "IWM",
        "FEZ",
        "XLF",
    ]

    for asset in important_assets:
        match = scoreboard[scoreboard["asset"] == asset]
        if match.empty:
            continue
        row = match.iloc[0]
        manual_text = "N/A" if pd.isna(row["manual_score"]) else f"{row['manual_score']:.1f}"
        tradable_text = (
            "N/A" if pd.isna(row["tradable_score"]) else f"{row['tradable_score']:.1f}"
        )
        model_text = "N/A" if pd.isna(row["model_score"]) else f"{row['model_score']:.4f}"
        page4.append(
            f"{asset}: target {fmt_pct(row['weight'], 1)} | "
            f"Exp10D {fmt_pct(row['expected_return'])} | "
            f"Manual {manual_text} | Tradable {tradable_text} | "
            f"Model {model_text} | Authority {row['authority']}"
        )
        if row["reason"] != "N/A":
            page4.append(f"Reason: {row['reason']}")

    pages = [page1, page2, page3, page4]
    texts = ["\n".join(page).strip() + "\n" for page in pages]

    for filename, text in zip(OUT_FILES, texts):
        (root / filename).write_text(text, encoding="utf-8")

    (root / OUT_ALL).write_text("\n\n".join(texts), encoding="utf-8")

    print("Saved four Telegram pages.")
    print("No model logic, score, weight, gate, leverage, or rebalance rule changed.")


if __name__ == "__main__":
    main()
