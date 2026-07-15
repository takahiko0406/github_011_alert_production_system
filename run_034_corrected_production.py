"""Run recovered 034, add missing gate-switch costs, and build live authority."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import research_072b_034_hybrid_switching_cost_audit as audit
from common_technology_leverage_080 import apply_common_overlay


ROOT = Path(__file__).resolve().parent
PREFIX = "model_c_plus_034_execution_grade_expected_return_signal"
VALIDATED = {"QQQM", "TQQQ", "SOXX", "SOXL", "IWM", "FEZ", "XLE", "XLB", "XLI", "XLV", "XLP", "XLU", "XLRE", "TLT", "GLD", "XSOE", "BIL"}
FORBIDDEN = {"XLF", "IEF", "ERX", "UXI", "TNA", "UGL"}


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
    corrected = original - costs
    pd.DataFrame({"portfolio_return": corrected, "use_expanded": gate, "source_switch_cost": costs}).to_csv(f"{PREFIX}_daily_returns.csv")
    pd.DataFrame(transition_rows).to_csv(f"{PREFIX}_gate_transition_costs.csv", index=False)
    summary = pd.DataFrame([
        {"model": "034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL_CORRECTED", **audit.metrics(corrected)},
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
