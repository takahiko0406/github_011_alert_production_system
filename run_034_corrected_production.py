"""Run recovered 034, add missing gate-switch costs, and build live authority."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import research_072b_034_hybrid_switching_cost_audit as audit
import research_022b_defensive_real_estate_allocation as r22b
from common_technology_leverage_080 import apply_common_overlay, apply_common_overlay_history


ROOT = Path(__file__).resolve().parent
PREFIX = "model_c_plus_034_execution_grade_expected_return_signal"
VALIDATED = {"QQQM", "TQQQ", "SOXX", "SOXL", "IWM", "FEZ", "XLE", "ERX", "XLB", "XLI", "UXI", "XLV", "XLP", "XLU", "XLRE", "TLT", "GLD", "XSOE", "BIL"}
FORBIDDEN = {"XLF", "IEF", "TNA", "UGL"}


def main() -> None:
    subprocess.run([sys.executable, "research_034_execution_grade_expected_return_signal.py"], cwd=ROOT, check=True)
    base = audit.load_return(ROOT / audit.BASE_RETURNS)
    expanded = audit.load_return(ROOT / audit.EXPANDED_RETURNS)
    idx = base.index.intersection(expanded.index).sort_values()
    base, expanded = base.loc[idx], expanded.loc[idx]
    base_log = pd.read_csv(ROOT / audit.BASE_LOG, parse_dates=["date"]).sort_values("date")
    expanded_log = pd.read_csv(ROOT / audit.EXPANDED_LOG, parse_dates=["date"]).sort_values("date")
    signal = expanded_log.set_index("date").reindex(idx, method="ffill")
    gate = signal.apply(audit.robust_gate, axis=1).astype(bool)
    original = base.copy()
    original.loc[gate] = expanded.loc[gate]

    base_w_log, base_assets = audit.weight_table(base_log, "last_w_")
    exp_w_log, exp_assets = audit.weight_table(expanded_log, "exec_w_")
    assets = sorted(set(base_assets).union(exp_assets))
    base_w = audit.effective_daily_weights(base_w_log, idx, base_assets).reindex(columns=assets, fill_value=0.0)
    exp_w = audit.effective_daily_weights(exp_w_log, idx, exp_assets).reindex(columns=assets, fill_value=0.0)
    final_w = base_w.copy()
    final_w.loc[gate, assets] = exp_w.loc[gate, assets]
    base_embedded = audit.embedded_cost_by_effective_date(base_log, idx, None)
    exp_embedded = audit.embedded_cost_by_effective_date(expanded_log, idx, "tx_cost_applied")
    transitions = gate.ne(gate.shift(1)).fillna(False)
    transitions.iloc[0] = False
    costs = pd.Series(0.0, index=idx)
    transition_rows = []
    for date in idx[transitions]:
        loc = idx.get_loc(date)
        turnover = float(0.5 * (final_w.loc[date] - final_w.iloc[loc - 1]).abs().sum())
        embedded = float(exp_embedded.loc[date] if gate.loc[date] else base_embedded.loc[date])
        cost = max(0.0, audit.BASE_RATE * turnover - embedded)
        costs.loc[date] = cost
        transition_rows.append({"date": date, "use_expanded": bool(gate.loc[date]), "turnover": turnover, "embedded_selected_source_cost": embedded, "missing_switch_cost": cost})
    pre_common = original - costs

    base_features = base_log.set_index("date").reindex(idx, method="ffill")
    features = signal[["growth_strength", "soxx_strength", "risk_off_strength", "crash_pressure"]].copy()
    features["total_budget"] = base_features["total_budget"]
    features["conviction_scale"] = base_features["conviction_scale"]
    if features.isna().any().any():
        raise ValueError("Common leverage historical features contain missing values")
    for asset in ["QQQM", "TQQQ", "SOXX", "SOXL"]:
        if asset not in final_w.columns:
            final_w[asset] = 0.0
    final_w = final_w.reindex(columns=sorted(final_w.columns), fill_value=0.0)
    start = (idx.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = r22b.download_prices_checked(start)
    missing_prices = [asset for asset in final_w.columns if asset not in prices.columns]
    if missing_prices:
        raise ValueError(f"Common leverage prices missing {missing_prices}")
    aligned_prices = prices[final_w.columns].reindex(prices.index.union(idx)).sort_index().ffill().reindex(idx)
    if aligned_prices.isna().any().any():
        raise ValueError("Common leverage aligned prices contain missing values")
    asset_returns = aligned_prices.pct_change().fillna(0.0)
    common_w, corrected, common_costs = apply_common_overlay_history(final_w, features, pre_common, asset_returns)
    daily = pd.DataFrame({
        "portfolio_return": corrected,
        "pre_common_return": pre_common,
        "use_expanded": gate,
        "source_switch_cost": costs,
        "common_gross_return_delta": common_costs["gross_overlay_return_delta"],
        "common_restored_embedded_cost": common_costs["restored_embedded_cost"],
        "common_final_turnover": common_costs["overlay_final_turnover"],
        "common_transaction_cost": common_costs["overlay_transaction_cost"],
        "common_slippage": common_costs["overlay_slippage"],
    })
    daily.to_csv(f"{PREFIX}_daily_returns.csv")
    common_w.assign(source_model=np.where(gate, "EXPANDED_CANDIDATE", "022F_BASE")).to_csv(f"{PREFIX}_effective_weights.csv", index_label="date")
    common_costs.to_csv(f"{PREFIX}_common_overlay_turnover_costs.csv", index_label="date")
    pd.DataFrame(transition_rows).to_csv(f"{PREFIX}_gate_transition_costs.csv", index=False)
    summary = pd.DataFrame([
        {"model": "034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL_CORRECTED", **audit.metrics(corrected)},
        {"model": "034_PRE_COMMON_TECHNOLOGY_LEVERAGE", **audit.metrics(pre_common)},
        {"model": "022F_BASE_COMMON", **audit.metrics(base)},
        {"model": "EXPANDED_CANDIDATE_COMMON", **audit.metrics(expanded)},
    ])
    summary.to_csv(f"{PREFIX}_performance_summary.csv", index=False)

    base_latest = pd.read_csv(ROOT / "model_c_plus_022F_calibrated_defense_validation_best_latest_recommendation.csv").iloc[-1]
    expanded_latest = pd.read_csv(ROOT / "model_c_plus_expanded_execution_candidate_latest_recommendation.csv").iloc[-1]
    common_date = str(base_latest["latest_data_date"])[:10]
    if str(expanded_latest["latest_data_date"])[:10] != common_date:
        raise ValueError("022F and expanded latest dates differ")
    use_expanded = audit.robust_gate(expanded_latest)
    chosen = expanded_latest if use_expanded else base_latest
    source_model = "EXPANDED_CANDIDATE" if use_expanded else "022F_BASE"
    weights = {}
    for asset in sorted(VALIDATED | FORBIDDEN):
        weights[asset] = float(chosen.get(f"exec_w_{asset}", 0.0) or 0.0)
    weights, leverage = apply_common_overlay(weights, base_latest, expanded_latest)
    invalid = {a: w for a, w in weights.items() if a in FORBIDDEN and abs(w) > 1e-12}
    if invalid:
        raise ValueError(f"Nonvalidated live weights present: {invalid}")
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError(f"034 latest weights sum to {sum(weights.values())}")
    latest = {
        "model": "034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL",
        "signal_date": common_date,
        "latest_data_date": common_date,
        "prediction_date": common_date,
        "allocation_date": common_date,
        "base_weight_date": common_date,
        "source_model": source_model,
        "gate_active": use_expanded,
        "robust_gate_active": use_expanded,
        "gate_name": "robust_expanded_gate",
        "gate_reason": "SOXX top, growth positive, crash controlled, cyclicals confirming" if use_expanded else "Robust expanded gate inactive; use 022F base",
        "expanded_top_asset": expanded_latest.get("top_asset"),
        "expanded_second_asset": expanded_latest.get("second_asset"),
        "expanded_score_gap": expanded_latest.get("score_gap"),
        "expanded_growth_strength": expanded_latest.get("growth_strength"),
        "expanded_crash_pressure": expanded_latest.get("crash_pressure"),
        "expanded_industrial_strength": expanded_latest.get("industrial_strength"),
        "expanded_materials_strength": expanded_latest.get("materials_strength"),
        "soxl_validation": "VALIDATED_COMMON_FRAMEWORK_RESEARCH_079D",
        "switch_cost_rate": audit.BASE_RATE,
    }
    latest.update(leverage)
    latest.update({f"exec_w_{asset}": weights[asset] for asset in sorted(weights)})
    latest["latest_weights"] = ", ".join(f"{a} {w:.1%}" for a, w in weights.items() if w > 1e-8)
    pd.DataFrame([latest]).to_csv(f"{PREFIX}_latest_recommendation.csv", index=False)
    print(f"Corrected 034 generated through {idx.max().date()}; live source {source_model} on {common_date}")


if __name__ == "__main__":
    main()
