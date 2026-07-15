"""Research 072b: isolate missing 034 source-model switching costs.

This is a read-only production audit.  It uses the recovered 022F and expanded
return streams (which already contain their own rebalance costs), reconstructs
the source portfolios from their saved logs, and charges only a gate-transition
cost not already embedded in the selected source stream.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PREFIX = "model_c_plus_072b_034"
BASE_RETURNS = "model_c_plus_022F_calibrated_defense_validation_best_daily_returns.csv"
EXPANDED_RETURNS = "model_c_plus_expanded_execution_candidate_daily_returns.csv"
BASE_LOG = "model_c_plus_022F_calibrated_defense_validation_best_rebalance_log.csv"
EXPANDED_LOG = "model_c_plus_expanded_execution_candidate_rebalance_log.csv"
ORIGINAL_034 = "model_c_plus_034_execution_grade_expected_return_signal_daily_returns.csv"
BASE_RATE = 0.001
SENSITIVITY_RATES = [0.0005, 0.0010, 0.0015, 0.0020]
WEIGHT_TOL = 1e-8
RETURN_TOL = 1e-12


def locate_recovered_root() -> Path:
    candidates = list(Path("C:/$Recycle.Bin").glob("S-*/$R*/research_034_execution_grade_expected_return_signal.py"))
    ranked: list[tuple[float, Path]] = []
    for script in candidates:
        root = script.parent
        required = [BASE_RETURNS, EXPANDED_RETURNS, BASE_LOG, EXPANDED_LOG, ORIGINAL_034]
        if not all((root / name).exists() for name in required):
            continue
        summary = root / "model_c_plus_034_execution_grade_expected_return_signal_performance_summary.csv"
        distance = 99.0
        if summary.exists():
            df = pd.read_csv(summary)
            row = df[df["model"].astype(str).eq("034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL")]
            if not row.empty:
                distance = abs(float(row.iloc[0]["annual_return"]) - 0.9807)
        ranked.append((distance, root))
    if not ranked:
        raise FileNotFoundError("No complete recovered 034 artifact set found")
    return min(ranked, key=lambda item: item[0])[1]


def load_return(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else "date"
    value_col = "portfolio_return"
    out = pd.Series(
        pd.to_numeric(df[value_col], errors="raise").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name=value_col,
    )
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError(f"Invalid return index: {path}")
    return out


def robust_gate(row: pd.Series) -> bool:
    """Exact recovered research_034 robust gate."""
    def num(name: str) -> float:
        value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else np.nan

    return bool(
        str(row.get("top_asset", "")) == "SOXX"
        and num("growth_strength") > 0.0
        and num("crash_pressure") < 0.50
        and (num("industrial_strength") > 0.0 or num("materials_strength") > 0.0)
    )


def weight_table(log: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in log.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No saved weight columns with prefix {prefix}")
    assets = [c[len(prefix):] for c in cols]
    weights = log[["date", *cols]].copy()
    weights = weights.rename(columns=dict(zip(cols, assets)))
    for asset in assets:
        weights[asset] = pd.to_numeric(weights[asset], errors="raise").fillna(0.0)
    return weights.sort_values("date"), assets


def effective_daily_weights(
    log_weights: pd.DataFrame, dates: pd.DatetimeIndex, assets: list[str]
) -> pd.DataFrame:
    """Weights logged at t apply to source returns beginning after t."""
    left = pd.DataFrame({"date": dates})
    daily = pd.merge_asof(
        left.sort_values("date"),
        log_weights.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("date")
    if daily[assets].isna().any(axis=None):
        bad = daily.index[daily[assets].isna().any(axis=1)]
        raise ValueError(f"Weights unavailable before {bad[0].date()}")
    sums = daily[assets].sum(axis=1)
    if not np.allclose(sums.to_numpy(), 1.0, atol=WEIGHT_TOL, rtol=0.0):
        worst = float((sums - 1.0).abs().max())
        raise ValueError(f"Saved effective weights do not sum to one; max error={worst}")
    return daily[assets]


def embedded_cost_by_effective_date(
    log: pd.DataFrame, dates: pd.DatetimeIndex, explicit_cost_col: str | None
) -> pd.Series:
    """Map each source rebalance cost to its first source-return date."""
    costs = pd.Series(0.0, index=dates)
    for _, row in log.iterrows():
        later = dates[dates > row["date"]]
        if len(later) == 0:
            continue
        effective_date = later[0]
        if explicit_cost_col and explicit_cost_col in log.columns:
            cost = float(row[explicit_cost_col])
        else:
            cost = float(row["turnover"]) * BASE_RATE
        costs.loc[effective_date] += cost
    return costs


def metrics(returns: pd.Series) -> dict[str, float | int | str]:
    r = returns.dropna()
    equity = (1.0 + r).cumprod()
    annual = float(equity.iloc[-1] ** (252.0 / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(252.0))
    return {
        "start": r.index.min().date().isoformat(),
        "end": r.index.max().date().isoformat(),
        "days": int(len(r)),
        "annual_return": annual,
        "volatility": vol,
        "sharpe": annual / vol if vol else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "final_equity": float(equity.iloc[-1]),
    }


def main() -> None:
    root = locate_recovered_root()
    base = load_return(root / BASE_RETURNS)
    expanded = load_return(root / EXPANDED_RETURNS)
    saved_034_df = pd.read_csv(root / ORIGINAL_034)
    saved_034_df["Date"] = pd.to_datetime(saved_034_df["Date"])
    saved_034 = saved_034_df.set_index("Date")["portfolio_return"].astype(float)

    idx = base.index.intersection(expanded.index).intersection(saved_034.index).sort_values()
    base, expanded, saved_034 = base.loc[idx], expanded.loc[idx], saved_034.loc[idx]
    if idx.empty:
        raise ValueError("No common historical date range")

    base_log = pd.read_csv(root / BASE_LOG, parse_dates=["date"]).sort_values("date")
    expanded_log = pd.read_csv(root / EXPANDED_LOG, parse_dates=["date"]).sort_values("date")

    # Gate timing deliberately mirrors recovered 034: same-day backward fill.
    signal = expanded_log.set_index("date").reindex(idx, method="ffill")
    gate = signal.apply(robust_gate, axis=1).astype(bool)
    reconstructed = base.copy()
    reconstructed.loc[gate] = expanded.loc[gate]
    max_return_diff = float((reconstructed - saved_034).abs().max())
    if max_return_diff > RETURN_TOL:
        raise ValueError(f"Recovered 034 return mismatch: {max_return_diff}")

    base_w_log, base_assets = weight_table(base_log, "last_w_")
    expanded_w_log, expanded_assets = weight_table(expanded_log, "exec_w_")
    assets = sorted(set(base_assets).union(expanded_assets))
    base_w = effective_daily_weights(base_w_log, idx, base_assets).reindex(columns=assets, fill_value=0.0)
    expanded_w = effective_daily_weights(expanded_w_log, idx, expanded_assets).reindex(columns=assets, fill_value=0.0)

    final_w = base_w.copy()
    final_w.loc[gate, assets] = expanded_w.loc[gate, assets]
    final_sums = final_w.sum(axis=1)
    if not np.allclose(final_sums.to_numpy(), 1.0, atol=WEIGHT_TOL, rtol=0.0):
        raise ValueError("Final 034 effective weights do not sum to 100%")

    base_embedded = embedded_cost_by_effective_date(base_log, idx, None)
    expanded_embedded = embedded_cost_by_effective_date(expanded_log, idx, "tx_cost_applied")

    transition_mask = gate.ne(gate.shift(1)).fillna(False)
    transition_mask.iloc[0] = False  # Initial source selection is not a source-model switch.
    rows: list[dict] = []
    for date in idx[transition_mask]:
        loc = idx.get_loc(date)
        prior_date = idx[loc - 1]
        old_w = final_w.loc[prior_date, assets]
        new_w = final_w.loc[date, assets]
        gross_turnover = float(0.5 * (new_w - old_w).abs().sum())
        new_source = "EXPANDED_CANDIDATE" if gate.loc[date] else "022F_BASE"
        old_source = "EXPANDED_CANDIDATE" if gate.loc[prior_date] else "022F_BASE"
        embedded = float(expanded_embedded.loc[date] if gate.loc[date] else base_embedded.loc[date])
        gross_cost = BASE_RATE * gross_turnover
        missing_cost = max(0.0, gross_cost - embedded)
        rows.append({
            "date": date.date().isoformat(),
            "previous_date": prior_date.date().isoformat(),
            "transition": f"{old_source}->{new_source}",
            "old_source": old_source,
            "new_source": new_source,
            "gross_source_switch_turnover": gross_turnover,
            "gross_source_switch_cost": gross_cost,
            "selected_source_cost_already_embedded_on_transition_date": embedded,
            "missing_source_switch_cost": missing_cost,
            "same_day_internal_rebalance": bool(embedded > 0),
            "method": "0.5*L1(previous final 034 weights,new final 034 weights); subtract selected-source cost already embedded that date",
        })
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        raise ValueError("No gate transitions found")

    missing_costs = pd.Series(0.0, index=idx)
    for _, row in transitions.iterrows():
        missing_costs.loc[pd.Timestamp(row["date"])] = float(row["missing_source_switch_cost"])
    corrected = saved_034 - missing_costs

    original_m = metrics(saved_034)
    corrected_m = metrics(corrected)
    base_m = metrics(base)
    ann_improvement = float(corrected_m["annual_return"] - base_m["annual_return"])
    sharpe_improvement = float(corrected_m["sharpe"] - base_m["sharpe"])
    dd_deterioration = max(0.0, float(base_m["max_drawdown"] - corrected_m["max_drawdown"]))
    condition_a = ann_improvement >= 0.01 and dd_deterioration <= 0.005
    condition_b = sharpe_improvement >= 0.05 and dd_deterioration <= 0.005
    decision = "PASS_CORRECTED_034_REMAINS_SUPERIOR" if condition_a or condition_b else "FAIL_CORRECTED_034_NOT_SUPERIOR"

    gate_out = pd.DataFrame({
        "date": idx.date,
        "use_expanded": gate.to_numpy(),
        "source_model": np.where(gate, "EXPANDED_CANDIDATE", "022F_BASE"),
        "gate_transition": transition_mask.to_numpy(),
        "top_asset": signal["top_asset"].to_numpy(),
        "growth_strength": signal["growth_strength"].to_numpy(),
        "crash_pressure": signal["crash_pressure"].to_numpy(),
        "industrial_strength": signal["industrial_strength"].to_numpy(),
        "materials_strength": signal["materials_strength"].to_numpy(),
    })
    effective_out = final_w.copy()
    effective_out.insert(0, "source_model", np.where(gate, "EXPANDED_CANDIDATE", "022F_BASE"))
    effective_out.insert(0, "date", idx.date)
    effective_out = effective_out.reset_index(drop=True)

    costs_out = transitions[[
        "date", "transition", "gross_source_switch_turnover", "gross_source_switch_cost",
        "selected_source_cost_already_embedded_on_transition_date", "missing_source_switch_cost",
        "same_day_internal_rebalance", "method",
    ]].copy()
    corrected_out = pd.DataFrame({
        "Date": idx.date,
        "original_034_return": saved_034.to_numpy(),
        "use_expanded": gate.to_numpy(),
        "source_switch_cost": missing_costs.to_numpy(),
        "corrected_034_return": corrected.to_numpy(),
    })

    metric_rows = []
    for name, values in (("ORIGINAL_034", original_m), ("CORRECTED_034", corrected_m), ("022F_BASE", base_m)):
        metric_rows.append({"model": name, **values})
    metrics_out = pd.DataFrame(metric_rows)

    sensitivity_rows = []
    for rate in SENSITIVITY_RATES:
        rate_costs = pd.Series(0.0, index=idx)
        for _, row in transitions.iterrows():
            cost = max(
                0.0,
                rate * float(row["gross_source_switch_turnover"])
                - float(row["selected_source_cost_already_embedded_on_transition_date"]),
            )
            rate_costs.loc[pd.Timestamp(row["date"])] = cost
        adjusted = saved_034 - rate_costs
        m = metrics(adjusted)
        sensitivity_rows.append({
            "cost_per_unit_turnover": rate,
            "total_missing_cost": float(rate_costs.sum()),
            "annualized_cost_drag": float(original_m["annual_return"] - m["annual_return"]),
            **m,
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    gate_out.to_csv(f"{PREFIX}_gate_states.csv", index=False)
    effective_out.to_csv(f"{PREFIX}_effective_weights.csv", index=False)
    transitions.to_csv(f"{PREFIX}_gate_transitions.csv", index=False)
    costs_out.to_csv(f"{PREFIX}_switching_costs.csv", index=False)
    corrected_out.to_csv(f"{PREFIX}_corrected_daily_returns.csv", index=False)
    metrics_out.to_csv(f"{PREFIX}_corrected_metrics.csv", index=False)
    sensitivity.to_csv(f"{PREFIX}_cost_sensitivity.csv", index=False)

    to_expanded = int(transitions["transition"].eq("022F_BASE->EXPANDED_CANDIDATE").sum())
    to_base = int(transitions["transition"].eq("EXPANDED_CANDIDATE->022F_BASE").sum())
    total_missing = float(transitions["missing_source_switch_cost"].sum())
    report = f"""RESEARCH 072b - 034 HYBRID SWITCHING COST AUDIT

Decision: {decision}
Recovered root: {root}
Common range: {idx.min().date()} through {idx.max().date()} ({len(idx)} trading days)
Original 034 reconstruction maximum absolute return difference: {max_return_diff:.3e}

Method
- The 022F and expanded daily streams are used unchanged, preserving their embedded internal costs.
- The robust gate is evaluated exactly as recovered: SOXX is top, growth > 0, crash pressure < 0.50, and industrial or materials strength > 0.
- Saved source weights logged at t become effective on the first source-return date after t, matching the producer hold-period code.
- A transition is charged only when the robust-gate state changes.
- Gross transition turnover is 0.5 times the L1 change from the prior final 034 portfolio to the new final 034 portfolio across {len(assets)} saved assets, including BIL and leveraged positions.
- At 0.10%, the incremental charge is max(0, 0.001 * gross transition turnover - selected-source cost already embedded on that transition date).
- Ordinary source rebalances on non-transition dates are never charged again.

Transition statistics
- 022F -> expanded transitions: {to_expanded}
- Expanded -> 022F transitions: {to_base}
- Total transitions: {len(transitions)}
- Average transition turnover: {transitions['gross_source_switch_turnover'].mean():.6f}
- Maximum transition turnover: {transitions['gross_source_switch_turnover'].max():.6f}
- Total missing cost: {total_missing:.6f}
- Annualized cost drag: {float(original_m['annual_return'] - corrected_m['annual_return']):.6%}

Performance
- Original 034: annual return {float(original_m['annual_return']):.6%}; volatility {float(original_m['volatility']):.6%}; Sharpe {float(original_m['sharpe']):.6f}; max drawdown {float(original_m['max_drawdown']):.6%}; final equity {float(original_m['final_equity']):.6f}
- Corrected 034: annual return {float(corrected_m['annual_return']):.6%}; volatility {float(corrected_m['volatility']):.6%}; Sharpe {float(corrected_m['sharpe']):.6f}; max drawdown {float(corrected_m['max_drawdown']):.6%}; final equity {float(corrected_m['final_equity']):.6f}
- 022F base: annual return {float(base_m['annual_return']):.6%}; volatility {float(base_m['volatility']):.6%}; Sharpe {float(base_m['sharpe']):.6f}; max drawdown {float(base_m['max_drawdown']):.6%}; final equity {float(base_m['final_equity']):.6f}

Superiority tests
- Annual-return improvement: {ann_improvement:.6%}; drawdown deterioration: {dd_deterioration:.6%}; condition A: {'PASS' if condition_a else 'FAIL'}
- Sharpe improvement: {sharpe_improvement:.6f}; drawdown deterioration: {dd_deterioration:.6%}; condition B: {'PASS' if condition_b else 'FAIL'}
"""
    Path(f"{PREFIX}_report.txt").write_text(report, encoding="utf-8")

    decision_payload = {
        "decision": decision,
        "recovered_root": str(root),
        "common_start": idx.min().date().isoformat(),
        "common_end": idx.max().date().isoformat(),
        "transition_count": int(len(transitions)),
        "022f_to_expanded_count": to_expanded,
        "expanded_to_022f_count": to_base,
        "total_missing_cost": total_missing,
        "annualized_cost_drag": float(original_m["annual_return"] - corrected_m["annual_return"]),
        "condition_a_pass": bool(condition_a),
        "condition_b_pass": bool(condition_b),
        "production_modified": False,
    }
    Path(f"{PREFIX}_decision.json").write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
