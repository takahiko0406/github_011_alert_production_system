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
ALERT_STATE_FILE = "model_c_plus_011_alert_state.json"

OUT1 = "model_c_plus_034_live_dashboard_telegram_page_1.txt"
OUT2 = "model_c_plus_034_live_dashboard_telegram_page_2.txt"
OUT3 = "model_c_plus_034_live_dashboard_telegram_page_3.txt"
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


def number(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def first(source_list, names):
    for source in source_list:
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
                pass
    return None


def fmt_pct(value: Any, decimals: int = 2) -> str:
    x = number(value)
    return "N/A" if x is None else f"{x * 100:.{decimals}f}%"


def fmt_num(value: Any, decimals: int = 3) -> str:
    x = number(value)
    return "N/A" if x is None else f"{x:.{decimals}f}"


def fmt_date(value: Any) -> str:
    return "N/A" if value is None else str(value)[:10]


def yn(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        t = value.strip().lower()
        if t in {"true", "yes", "1", "active", "on"}:
            return "YES"
        if t in {"false", "no", "0", "inactive", "off"}:
            return "NO"
        return value
    return "YES" if bool(value) else "NO"


def section(title: str) -> list[str]:
    return ["", "=" * 30, title, "=" * 30]


def get_target_weights(row: pd.Series) -> dict[str, float]:
    weights = {}
    for col in row.index:
        if str(col).startswith("exec_w_"):
            x = number(row[col])
            if x is not None and x > 1e-10:
                weights[str(col)[7:]] = x
    return dict(sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])))


def get_current_weights(state: dict[str, Any]) -> dict[str, float]:
    for key in ["current_allocation", "current_weights", "weights", "portfolio"]:
        value = state.get(key)
        if isinstance(value, dict):
            parsed = {}
            for asset, weight in value.items():
                x = number(weight)
                if x is not None and x > 1e-10:
                    parsed[str(asset)] = x
            if parsed:
                return dict(sorted(parsed.items(), key=lambda kv: (-kv[1], kv[0])))
    return {}


def add_weights(lines: list[str], title: str, weights: dict[str, float]) -> None:
    lines.extend(section(title))
    if not weights:
        lines.append("N/A")
        return
    for asset, weight in weights.items():
        lines.append(f"{asset:<6} {weight * 100:>7.1f}%")
    lines.append(f"{'TOTAL':<6} {sum(weights.values()) * 100:>7.1f}%")


def normalize_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    asset_col = next((c for c in ["asset", "ETF", "ticker"] if c in df.columns), None)
    er_col = next((c for c in [
        "adjusted_expected_10d_return",
        "expected_10d_return",
        "raw_expected_10d_return",
        "predicted_10d_return",
    ] if c in df.columns), None)
    xscore_col = next((c for c in [
        "manual_action_score_0_100",
        "tradable_score_0_100",
        "execution_score",
        "xscore",
    ] if c in df.columns), None)
    model_score_col = next((c for c in [
        "selected_model_score",
        "execution_model_score",
        "model_score",
    ] if c in df.columns), None)
    weight_col = next((c for c in [
        "selected_exec_weight",
        "exec_weight",
        "target_weight",
    ] if c in df.columns), None)

    if asset_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "asset": df[asset_col].astype(str),
        "expected_return": pd.to_numeric(df[er_col], errors="coerce") if er_col else float("nan"),
        "xscore": pd.to_numeric(df[xscore_col], errors="coerce") if xscore_col else float("nan"),
        "model_score": pd.to_numeric(df[model_score_col], errors="coerce") if model_score_col else float("nan"),
        "weight": pd.to_numeric(df[weight_col], errors="coerce") if weight_col else 0.0,
    })
    return out.drop_duplicates("asset", keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    latest = read_last_csv(root / LATEST_FILE)
    if latest.empty:
        raise FileNotFoundError(LATEST_FILE)

    scoreboard_path = root / SCOREBOARD_FILE
    if not scoreboard_path.exists():
        raise FileNotFoundError(SCOREBOARD_FILE)
    scoreboard = normalize_scoreboard(pd.read_csv(scoreboard_path))
    if scoreboard.empty:
        raise ValueError("Scoreboard columns could not be recognized")

    perf = pd.read_csv(root / PERFORMANCE_FILE) if (root / PERFORMANCE_FILE).exists() else pd.DataFrame()
    perf_row = pd.Series(dtype=object)
    if not perf.empty:
        matched = perf[perf["model"].astype(str).str.contains("CORRECTED", case=False, na=False)]
        perf_row = matched.iloc[0] if not matched.empty else perf.iloc[0]

    validation = read_json(root / VALIDATION_FILE)
    state = read_json(root / ALERT_STATE_FILE)

    target_weights = get_target_weights(latest)
    current_weights = get_current_weights(state)
    expected_portfolio_return = float(
        (scoreboard["expected_return"].fillna(0.0) * scoreboard["weight"].fillna(0.0)).sum()
    )

    data_date = fmt_date(first([latest, validation], ["latest_data_date", "data_date"]))
    allocation_date = fmt_date(first([latest, validation], ["allocation_date", "base_weight_date"]))
    base_date = fmt_date(first([latest, state], [
        "base_weight_date", "current_allocation_base", "base_date", "allocation_base_date"
    ]))
    base_model = first([latest, state], [
        "source_model", "current_best_base", "base_model", "current_base_model"
    ]) or "N/A"

    page1 = [
        "034 DAILY MODEL REPORT",
        f"Data: {data_date}",
        f"Allocation: {allocation_date}",
    ]
    page1 += section("PORTFOLIO METRICS")
    page1.append(f"Expected 10D portfolio return: {fmt_pct(expected_portfolio_return)}")
    page1.append(f"Historical annual return: {fmt_pct(first([perf_row], ['annual_return']))}")
    page1.append(f"Historical volatility: {fmt_pct(first([perf_row], ['volatility']))}")
    page1.append(f"Historical Sharpe: {fmt_num(first([perf_row], ['sharpe']))}")
    page1.append(f"Historical max drawdown: {fmt_pct(first([perf_row], ['max_drawdown']))}")

    page1 += section("EXECUTION STATE")
    page1.append(f"Base model: {base_model}")
    page1.append(f"Base date: {base_date}")
    rebalance_day = first([latest, state], ["rebalance_day", "rebalance_clock_day", "day_in_rebalance_cycle"])
    rebalance_length = first([latest, state], ["rebalance_cycle_length", "rebalance_clock_length", "rebalance_interval"])
    page1.append(f"Rebalance clock: day {rebalance_day if rebalance_day is not None else 'N/A'}/{rebalance_length if rebalance_length is not None else 'N/A'}")
    page1.append(f"Trading days elapsed: {first([latest, state], ['trading_days_elapsed', 'elapsed_trading_days', 'days_elapsed']) or 'N/A'}")
    page1.append(f"Normal rebalance due: {yn(first([latest, state], ['normal_rebalance_due', 'rebalance_due']))}")
    page1.append(f"Emergency: {first([latest, state], ['emergency_status', 'portfolio_wide_emergency', 'emergency_alert', 'emergency']) or 'N/A'}")
    cooldown = first([latest, state], ["cooldown", "cooldown_days", "cooldown_elapsed"])
    cooldown_req = first([latest, state], ["cooldown_required", "required_cooldown", "cooldown_limit"])
    page1.append(f"Cooldown: {cooldown if cooldown is not None else 'N/A'}/{cooldown_req if cooldown_req is not None else 'N/A'}")
    page1.append(f"Top signal: {first([latest], ['top_asset', 'expanded_top_asset']) or 'N/A'}")
    page1.append(f"Second signal: {first([latest], ['second_asset', 'expanded_second_asset']) or 'N/A'}")
    page1.append(f"Score gap: {fmt_num(first([latest], ['score_gap', 'expanded_score_gap']), 4)}")
    page1.append(f"Required gap: {fmt_num(first([latest, state], ['required_gap', 'score_gap_required']), 4)}")

    add_weights(page1, "CURRENT ALLOCATION", current_weights)
    add_weights(page1, "TARGET ALLOCATION", target_weights)

    page2 = ["034 DAILY MODEL REPORT — PAGE 2"]
    page2 += section("EXPECTED 10D RETURNS")
    for _, row in scoreboard.sort_values(["expected_return", "asset"], ascending=[False, True], na_position="last").iterrows():
        page2.append(f"{row['asset']:<6} {fmt_pct(row['expected_return']):>9}")

    page2 += section("X-SCORES")
    for _, row in scoreboard.sort_values(["xscore", "asset"], ascending=[False, True], na_position="last").iterrows():
        value = "N/A" if pd.isna(row["xscore"]) else f"{row['xscore']:.1f}"
        page2.append(f"{row['asset']:<6} {value:>9}")

    page2 += section("EXECUTION MODEL SCORES")
    for _, row in scoreboard.sort_values(["model_score", "asset"], ascending=[False, True], na_position="last").iterrows():
        value = "N/A" if pd.isna(row["model_score"]) else f"{row['model_score']:.4f}"
        page2.append(f"{row['asset']:<6} {value:>9}")

    page3 = ["034 DAILY MODEL REPORT — PAGE 3"]
    page3 += section("REGIME VARIABLES")
    regime = [
        ("Growth", ["growth_strength", "expanded_growth_strength"]),
        ("SOXX", ["soxx_strength"]),
        ("Risk-off", ["risk_off_strength"]),
        ("Crash pressure", ["crash_pressure", "expanded_crash_pressure"]),
        ("Credit", ["credit_strength"]),
        ("USD 3M", ["usd_3m_strength"]),
        ("Copper", ["copper_strength"]),
        ("Industrial", ["industrial_strength", "expanded_industrial_strength"]),
        ("Materials", ["materials_strength", "expanded_materials_strength"]),
        ("Oil", ["oil_strength", "oil_3m"]),
        ("VIX", ["vix_level", "vix"]),
        ("Yield curve", ["yield_curve"]),
    ]
    for label, aliases in regime:
        page3.append(f"{label}: {fmt_num(first([latest], aliases))}")

    page3 += section("TECHNOLOGY LEVERAGE")
    page3.append(f"QQQM base weight: {fmt_pct(first([latest], ['qqqm_base_weight']))}")
    page3.append(f"TQQQ substituted weight: {fmt_pct(first([latest], ['tqqq_substituted_weight']))}")
    page3.append(f"SOXX base weight: {fmt_pct(first([latest], ['soxx_base_weight']))}")
    page3.append(f"SOXL substituted weight: {fmt_pct(first([latest], ['soxl_substituted_weight']))}")
    page3.append(f"Common leverage budget: {fmt_pct(first([latest], ['common_leverage_budget']))}")
    page3.append(f"Effective technology exposure: {fmt_num(first([latest], ['effective_technology_exposure']))}")
    page3.append(f"Leverage configuration: {first([latest], ['common_leverage_configuration']) or 'N/A'}")

    page3 += section("SYSTEM STATUS")
    page3.append(f"Execution safe: {yn(first([validation], ['execution_safe']))}")
    assertions = validation.get("assertions", {})
    page3.append(f"Validation: {'PASS' if assertions and all(assertions.values()) else 'N/A'}")
    page3.append(f"Model: {first([latest], ['model']) or 'N/A'}")
    page3.append(f"Data freshness: {first([latest, validation], ['freshness', 'data_status']) or 'N/A'}")

    texts = [
        "\n".join(page1).strip() + "\n",
        "\n".join(page2).strip() + "\n",
        "\n".join(page3).strip() + "\n",
    ]
    for name, text in zip([OUT1, OUT2, OUT3], texts):
        (root / name).write_text(text, encoding="utf-8")

    (root / OUT_ALL).write_text("\n\n".join(texts), encoding="utf-8")

    print(f"Saved {OUT1}")
    print(f"Saved {OUT2}")
    print(f"Saved {OUT3}")
    print(f"Saved {OUT_ALL}")
    print("No model logic, score, weight, or rebalance rule was changed.")


if __name__ == "__main__":
    main()
