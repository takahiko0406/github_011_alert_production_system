from pathlib import Path

import numpy as np
import pandas as pd


OUT_PREFIX = "model_c_plus_034_execution_grade_expected_return_signal"

BASE_RETURNS = Path("model_c_plus_022F_calibrated_defense_validation_best_daily_returns.csv")
EXPANDED_RETURNS = Path("model_c_plus_expanded_execution_candidate_daily_returns.csv")
BASE_LOG = Path("model_c_plus_022F_calibrated_defense_validation_best_rebalance_log.csv")
EXPANDED_LOG = Path("model_c_plus_expanded_execution_candidate_rebalance_log.csv")
EXPANDED_LATEST = Path("model_c_plus_expanded_execution_candidate_latest_recommendation.csv")
LIGHT_LATEST = Path("model_c_plus_transition_conviction_overlay_011_LIGHT_latest_recommendation.csv")
TRADING_SCORES = Path("model_c_plus_full_universe_expected_returns_trading_scores.csv")

ASSETS = [
    "TQQQ", "SOXL", "SOXX", "QQQM", "IWM", "FEZ",
    "XLE", "XLB", "XLI", "XLF", "XLV", "XLP", "XLU", "XLRE",
    "TLT", "IEF", "GLD", "BIL", "XSOE",
]

DISPLAY_ORDER = {asset: i for i, asset in enumerate(ASSETS)}


def load_returns(path):
    frame = pd.read_csv(path)
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    col = "portfolio_return" if "portfolio_return" in frame.columns else frame.columns[0]
    return frame[col].astype(float)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def annualized_return(returns):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return (1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0


def annualized_volatility(returns):
    return returns.dropna().std() * np.sqrt(252.0)


def sharpe_ratio(returns):
    vol = annualized_volatility(returns)
    if vol == 0 or pd.isna(vol):
        return np.nan
    return annualized_return(returns) / vol


def max_drawdown(returns):
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    return (equity / equity.cummax() - 1.0).min()


def metrics(name, returns):
    returns = returns.dropna()
    return {
        "model": name,
        "annual_return": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "start": returns.index.min().date().isoformat(),
        "end": returns.index.max().date().isoformat(),
        "days": int(returns.shape[0]),
        "final_equity": float((1.0 + returns).prod()),
    }


def robust_gate(row):
    return (
        str(row.get("top_asset", "")) == "SOXX"
        and safe_float(row.get("growth_strength")) > 0.0
        and safe_float(row.get("crash_pressure")) < 0.50
        and (
            safe_float(row.get("industrial_strength")) > 0.0
            or safe_float(row.get("materials_strength")) > 0.0
        )
    )


def opportunistic_gate(row):
    return (
        str(row.get("top_asset", "")) == "SOXX"
        and safe_float(row.get("risk_off_strength"), 9.0) < 0.75
    )


def active_weights_text(row, prefix):
    active = []
    for asset in ASSETS:
        weight = safe_float(row.get(f"{prefix}{asset}", 0.0))
        if abs(weight) > 0.0001:
            active.append((asset, weight))
    active.sort(key=lambda item: abs(item[1]), reverse=True)
    return ", ".join(f"{asset} {weight * 100:.1f}%" for asset, weight in active) or "none"


def read_latest(path):
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    return frame.iloc[-1].to_dict()


def normalize_latest_base(base_row, expanded_row):
    out = {
        "model": "034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL",
        "signal_date": expanded_row.get("signal_date", base_row.get("date")),
        "latest_data_date": expanded_row.get("latest_data_date", base_row.get("date")),
        "gate_active": False,
        "gate_name": "robust_expanded_gate",
        "gate_reason": "Use 022F base because robust expanded gate is inactive.",
        "source_model": "022F_balanced_defense_real_estate",
        "score_gap": base_row.get("score_gap_inferred", expanded_row.get("score_gap", np.nan)),
        "base_weight_date": base_row.get("date", "N/A"),
        "expanded_top_asset": expanded_row.get("top_asset", "N/A"),
        "expanded_second_asset": expanded_row.get("second_asset", "N/A"),
        "expanded_score_gap": expanded_row.get("score_gap", np.nan),
        "expanded_growth_strength": expanded_row.get("growth_strength", np.nan),
        "expanded_crash_pressure": expanded_row.get("crash_pressure", np.nan),
        "expanded_industrial_strength": expanded_row.get("industrial_strength", np.nan),
        "expanded_materials_strength": expanded_row.get("materials_strength", np.nan),
        "latest_weights": active_weights_text(base_row, "last_w_"),
    }
    for asset in ASSETS:
        out[f"exec_w_{asset}"] = base_row.get(f"last_w_{asset}", 0.0)
    return out


def normalize_latest_expanded(expanded_row):
    out = {
        "model": "034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL",
        "signal_date": expanded_row.get("signal_date", expanded_row.get("date")),
        "latest_data_date": expanded_row.get("latest_data_date", expanded_row.get("date")),
        "gate_active": True,
        "gate_name": "robust_expanded_gate",
        "gate_reason": "Robust expanded gate active: SOXX top, growth positive, crash controlled, cyclicals confirming.",
        "source_model": "expanded_execution_candidate",
        "score_gap": expanded_row.get("score_gap", np.nan),
        "expanded_top_asset": expanded_row.get("top_asset", "N/A"),
        "expanded_second_asset": expanded_row.get("second_asset", "N/A"),
        "expanded_score_gap": expanded_row.get("score_gap", np.nan),
        "expanded_growth_strength": expanded_row.get("growth_strength", np.nan),
        "expanded_crash_pressure": expanded_row.get("crash_pressure", np.nan),
        "expanded_industrial_strength": expanded_row.get("industrial_strength", np.nan),
        "expanded_materials_strength": expanded_row.get("materials_strength", np.nan),
        "latest_weights": active_weights_text(expanded_row, "exec_w_"),
    }
    for asset in ASSETS:
        out[f"exec_w_{asset}"] = expanded_row.get(f"exec_w_{asset}", 0.0)
    return out


def score_rank(rows, score_key):
    values = [safe_float(row.get(score_key), np.nan) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return [0.0 for _ in rows]
    lo, hi = min(finite), max(finite)
    if hi <= lo:
        return [50.0 for _ in rows]
    return [100.0 * (safe_float(row.get(score_key), lo) - lo) / (hi - lo) for row in rows]


def build_execution_grade_scores(light, expanded, robust_active, opportunistic_active):
    trading = pd.read_csv(TRADING_SCORES)
    rows = []
    for _, score_row in trading.iterrows():
        asset = str(score_row["asset"])
        live_score = safe_float(light.get(f"adj_pred_{asset}", np.nan), np.nan)
        expanded_score = safe_float(expanded.get(f"adj_pred_{asset}", np.nan), np.nan)
        live_exec_w = safe_float(light.get(f"exec_w_{asset}", 0.0))
        live_signal_w = safe_float(light.get(f"signal_w_{asset}", 0.0))
        expanded_exec_w = safe_float(expanded.get(f"exec_w_{asset}", 0.0))
        expanded_signal_w = safe_float(expanded.get(f"signal_w_{asset}", 0.0))

        if np.isfinite(live_score):
            authority = "LIVE_EXECUTION_MODEL"
            execution_ok = True
            authority_multiplier = 1.00
            selected_score = live_score
            selected_weight = live_exec_w
            reason = "Covered by current production execution model."
        elif robust_active and expanded_signal_w > 0:
            authority = "ROBUST_SHADOW_EXECUTION"
            execution_ok = True
            authority_multiplier = 0.90
            selected_score = expanded_score
            selected_weight = expanded_exec_w
            reason = "Expanded ETF selected by robust validated gate."
        elif opportunistic_active and expanded_signal_w > 0:
            authority = "OPPORTUNISTIC_SHADOW_EXECUTION"
            execution_ok = False
            authority_multiplier = 0.65
            selected_score = expanded_score
            selected_weight = expanded_exec_w
            reason = "Expanded ETF selected by high-return gate, but robust gate is not active."
        else:
            authority = "RESEARCH_EXPECTED_RETURN_ONLY"
            execution_ok = False
            authority_multiplier = 0.35
            selected_score = expanded_score
            selected_weight = 0.0
            reason = "Expected-return research signal only; not execution-grade today."

        rows.append({
            "signal_date": score_row.get("signal_date"),
            "asset": asset,
            "authority": authority,
            "execution_ok": execution_ok,
            "manual_action_score_0_100": np.nan,
            "authority_multiplier": authority_multiplier,
            "selected_model_score": selected_score,
            "selected_exec_weight": selected_weight,
            "live_execution_score": live_score,
            "live_exec_weight": live_exec_w,
            "live_signal_weight": live_signal_w,
            "expanded_score": expanded_score,
            "expanded_exec_weight": expanded_exec_w,
            "expanded_signal_weight": expanded_signal_w,
            "tradable_score_0_100": safe_float(score_row.get("tradable_score_0_100")),
            "adjusted_expected_10d_return": safe_float(score_row.get("adjusted_expected_10d_return")),
            "calibrated_bucket_realized_10d_return": safe_float(score_row.get("calibrated_bucket_realized_10d_return")),
            "historical_bucket_hit_rate": safe_float(score_row.get("historical_bucket_hit_rate")),
            "expected_return_per_10d_vol": safe_float(score_row.get("expected_return_per_10d_vol")),
            "confidence_label": score_row.get("confidence_label"),
            "reason": reason,
        })

    rank_scores = score_rank(rows, "selected_model_score")
    for row, rank_score in zip(rows, rank_scores):
        tradable = safe_float(row.get("tradable_score_0_100"))
        weight_score = min(100.0, abs(safe_float(row.get("selected_exec_weight"))) * 100.0)
        raw_action = 0.55 * rank_score + 0.30 * tradable + 0.15 * weight_score
        row["manual_action_score_0_100"] = raw_action * safe_float(row["authority_multiplier"])
    rows.sort(
        key=lambda row: (
            bool(row["execution_ok"]),
            safe_float(row["manual_action_score_0_100"]),
            safe_float(row["tradable_score_0_100"]),
        ),
        reverse=True,
    )
    return pd.DataFrame(rows)


def main():
    base_returns = load_returns(BASE_RETURNS)
    expanded_returns = load_returns(EXPANDED_RETURNS)
    base_log = pd.read_csv(BASE_LOG, parse_dates=["date"]).sort_values("date")
    expanded_log = pd.read_csv(EXPANDED_LOG, parse_dates=["date"]).sort_values("date")

    idx = base_returns.index.intersection(expanded_returns.index).sort_values()
    base_returns = base_returns.loc[idx]
    expanded_returns = expanded_returns.loc[idx]
    expanded_signal = expanded_log.set_index("date").reindex(idx, method="ffill")

    gate = expanded_signal.apply(robust_gate, axis=1)
    combined = base_returns.copy()
    combined.loc[gate] = expanded_returns.loc[gate]
    pd.DataFrame({
        "portfolio_return": combined,
        "use_expanded": gate.astype(bool),
    }).to_csv(f"{OUT_PREFIX}_daily_returns.csv")

    summary = pd.DataFrame([
        metrics("034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL", combined),
        metrics("022F_BASE_COMMON", base_returns),
        metrics("EXPANDED_CANDIDATE_COMMON", expanded_returns),
    ])
    summary.to_csv(f"{OUT_PREFIX}_performance_summary.csv", index=False)

    latest_base = base_log.iloc[-1].to_dict()
    latest_expanded = read_latest(EXPANDED_LATEST)
    light = read_latest(LIGHT_LATEST)
    latest_robust = robust_gate(latest_expanded)
    latest_opportunistic = opportunistic_gate(latest_expanded)
    latest = normalize_latest_expanded(latest_expanded) if latest_robust else normalize_latest_base(latest_base, latest_expanded)
    latest["opportunistic_gate_active"] = latest_opportunistic
    latest["robust_gate_active"] = latest_robust
    pd.DataFrame([latest]).to_csv(f"{OUT_PREFIX}_latest_recommendation.csv", index=False)

    grade_scores = build_execution_grade_scores(light, latest_expanded, latest_robust, latest_opportunistic)
    grade_scores.to_csv(f"{OUT_PREFIX}_scoreboard.csv", index=False)

    diagnostics = pd.DataFrame([{
        "rule": "top SOXX + growth > 0 + crash_pressure < 0.50 + (industrial > 0 or materials > 0)",
        "active_days": int(gate.sum()),
        "active_pct": float(gate.mean()),
        "latest_robust_gate_active": latest_robust,
        "latest_opportunistic_gate_active": latest_opportunistic,
        "latest_source_model": latest["source_model"],
        "latest_weights": latest["latest_weights"],
    }])
    diagnostics.to_csv(f"{OUT_PREFIX}_diagnostics.csv", index=False)

    lines = [
        "034 EXECUTION-GRADE EXPECTED RETURN SIGNAL",
        "",
        "Purpose",
        "- Keep all-ETF expected returns visible.",
        "- Mark only production or robust-gated shadow signals as execution-grade.",
        "",
        "Performance",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['model']}: ann {row['annual_return']:.4f}, "
            f"Sharpe {row['sharpe']:.4f}, DD {row['max_drawdown']:.4f}, "
            f"final equity {row['final_equity']:.4f}"
        )
    lines.extend([
        "",
        "Latest Gate",
        f"- Robust gate active: {latest_robust}",
        f"- Opportunistic gate active: {latest_opportunistic}",
        f"- Source model: {latest['source_model']}",
        f"- Weights: {latest['latest_weights']}",
        "",
        "Top Execution-Grade Scores",
    ])
    for _, row in grade_scores.head(10).iterrows():
        lines.append(
            f"- {row['asset']}: action {row['manual_action_score_0_100']:.1f}, "
            f"authority {row['authority']}, exp10d {row['adjusted_expected_10d_return']:+.2%}, "
            f"weight {row['selected_exec_weight']:.1%}"
        )
    Path(f"{OUT_PREFIX}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(diagnostics.to_string(index=False))
    print(grade_scores.head(16)[[
        "asset", "authority", "execution_ok", "manual_action_score_0_100",
        "selected_model_score", "selected_exec_weight", "adjusted_expected_10d_return",
        "tradable_score_0_100", "reason",
    ]].to_string(index=False))
    print(f"Saved {OUT_PREFIX}_scoreboard.csv and related outputs")


if __name__ == "__main__":
    main()
