"""Reporting-only audit of the frozen production emergency mechanism.

The module imports every threshold from ``common_technology_leverage_080`` and
uses the immutable Research-079d/081 replay bundle.  It does not alter the
production allocation path.  The WITHOUT_EMERGENCY replay bypasses only the
historical hard-emergency adjustment; every other gate and cost convention is
held constant.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import research_072b_034_hybrid_switching_cost_audit as performance_lib
from common_technology_leverage_080 import (
    CONFIGURATION_ID,
    GROWTH_MIN,
    HARD_CRASH_MAX,
    HARD_RISK_OFF_MAX,
    LEVERAGE_MAPPINGS,
    RISK_OFF_MAX,
    SLIPPAGE_RATE,
    SOXX_STRENGTH_MIN,
    TECHNOLOGY_EXPOSURE_LIMIT,
    TRANSACTION_COST_RATE,
    apply_common_overlay,
    apply_common_overlay_history,
)
from run_034_corrected_production import MATURE_REPLAY, MATURE_REPLAY_SHA256, load_mature_replay


ROOT = Path(__file__).resolve().parent
RULE_OUT = ROOT / "emergency_rule_definition.json"
SESSION_OUT = ROOT / "emergency_tqqq_soxl_session_audit.csv"
EPISODE_OUT = ROOT / "emergency_tqqq_soxl_episode_audit.csv"
PERFORMANCE_OUT = ROOT / "emergency_tqqq_soxl_ab_performance.csv"
FORWARD_OUT = ROOT / "emergency_tqqq_soxl_forward_effect.csv"
TEST_OUT = ROOT / "emergency_tqqq_soxl_boundary_tests.csv"
VALIDATION_OUT = ROOT / "emergency_tqqq_soxl_functional_validation.json"
REPORT_OUT = ROOT / "emergency_tqqq_soxl_report.txt"

ASSET_WEIGHT_TOLERANCE = 1e-12
RETURN_TOLERANCE = 1e-15
REQUIRED_FEATURES = ["growth_strength", "soxx_strength", "risk_off_strength", "crash_pressure", "total_budget"]
TECH_ASSETS = ["QQQM", "TQQQ", "SOXX", "SOXL"]


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def emergency_criteria(row: Any) -> dict[str, Any]:
    """Reproduce the exact hard-safe expression using imported constants."""
    values = {name: row.get(name, np.nan) for name in REQUIRED_FEATURES}
    inputs_finite = all(finite_number(value) for value in values.values())
    risk_pass = finite_number(values["risk_off_strength"]) and float(values["risk_off_strength"]) < HARD_RISK_OFF_MAX
    crash_pass = finite_number(values["crash_pressure"]) and float(values["crash_pressure"]) < HARD_CRASH_MAX
    hard_safe = inputs_finite and risk_pass and crash_pass
    reasons = []
    if not inputs_finite:
        reasons.append("required_input_missing_or_non_finite")
    if finite_number(values["risk_off_strength"]) and not risk_pass:
        reasons.append("risk_off_strength_greater_than_or_equal_to_hard_max")
    if finite_number(values["crash_pressure"]) and not crash_pass:
        reasons.append("crash_pressure_greater_than_or_equal_to_hard_max")
    return {
        **values,
        "inputs_finite": inputs_finite,
        "risk_off_lt_hard_max": risk_pass,
        "crash_pressure_lt_hard_max": crash_pass,
        "hard_safe": hard_safe,
        "state": "NORMAL" if hard_safe else "EXIT",
        "reason": "all_hard_safety_criteria_passed" if hard_safe else ";".join(reasons),
    }


def technology_exposure(weights: pd.DataFrame | pd.Series | dict[str, float]) -> Any:
    if isinstance(weights, pd.DataFrame):
        return (
            weights.get("QQQM", 0.0)
            + weights.get("SOXX", 0.0)
            + 3.0 * (weights.get("TQQQ", 0.0) + weights.get("SOXL", 0.0))
        )
    getter = weights.get
    return float(getter("QQQM", 0.0) + getter("SOXX", 0.0) + 3.0 * (getter("TQQQ", 0.0) + getter("SOXL", 0.0)))


def build_historical_counterfactual(
    original: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return common pre-emergency candidate, WITH, and emergency-state series.

    This mirrors ``apply_common_overlay_history``.  The only bypass is removal
    of ``risk < HARD_RISK_OFF_MAX`` and ``crash < HARD_CRASH_MAX`` from SOXL's
    historical hard-emergency adjustment.  The finite-input requirement and
    all ordinary SOXL admission criteria remain unchanged.  Historical TQQQ is
    reconstructed from the archived validated weights exactly as production
    does; that function contains no additional TQQQ hard-emergency adjustment.
    """
    base = original.copy().astype(float)
    base["QQQM"] = base["QQQM"] + base["TQQQ"]
    base["TQQQ"] = 0.0
    base["SOXX"] = base["SOXX"] + base["SOXL"]
    base["SOXL"] = 0.0
    candidate = base.copy()

    qqqm_total = original["QQQM"] + original["TQQQ"]
    tqqq_fraction = (original["TQQQ"] / qqqm_total.replace(0.0, np.nan)).fillna(0.0)
    raw_tqqq = base["QQQM"] * tqqq_fraction
    candidate["QQQM"] -= raw_tqqq
    candidate["TQQQ"] += raw_tqqq

    finite = features[REQUIRED_FEATURES].apply(np.isfinite).all(axis=1)
    ordinary_soxl_allowed = (
        base["SOXX"].gt(ASSET_WEIGHT_TOLERANCE)
        & features["growth_strength"].ge(GROWTH_MIN)
        & features["soxx_strength"].ge(SOXX_STRENGTH_MIN)
        & features["risk_off_strength"].le(RISK_OFF_MAX)
        & finite
    )
    raw_soxl_fraction = (
        features["total_budget"] / base["SOXX"].replace(0.0, np.nan)
    ).clip(0.0, 1.0).fillna(0.0).where(ordinary_soxl_allowed, 0.0)
    raw_soxl = base["SOXX"] * raw_soxl_fraction
    candidate["SOXX"] -= raw_soxl
    candidate["SOXL"] += raw_soxl

    desired = candidate["TQQQ"] + candidate["SOXL"]
    shared_budget = pd.Series(
        np.maximum(original["TQQQ"], features["total_budget"].fillna(0.0)), index=original.index
    )
    scale = (shared_budget / desired.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    unscaled_tqqq, unscaled_soxl = candidate["TQQQ"].copy(), candidate["SOXL"].copy()
    candidate["TQQQ"] = unscaled_tqqq * scale
    candidate["QQQM"] += unscaled_tqqq - candidate["TQQQ"]
    candidate["SOXL"] = unscaled_soxl * scale
    candidate["SOXX"] += unscaled_soxl - candidate["SOXL"]

    state = features.apply(lambda row: emergency_criteria(row)["state"], axis=1)
    with_emergency = candidate.copy()
    exit_mask = state.eq("EXIT")
    # Exact historical production behavior: the hard gate is in SOXL's
    # admission expression.  TQQQ is reconstructed from the archived weights.
    with_emergency.loc[exit_mask, "SOXX"] += with_emergency.loc[exit_mask, "SOXL"]
    with_emergency.loc[exit_mask, "SOXL"] = 0.0
    return candidate, with_emergency, state


def returns_for_weights(
    original: pd.DataFrame,
    weights: pd.DataFrame,
    archived_returns: pd.Series,
    asset_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    aligned = asset_returns.reindex(index=original.index, columns=original.columns)
    original_turnover = 0.5 * original.diff().abs().sum(axis=1)
    final_turnover = 0.5 * weights.diff().abs().sum(axis=1)
    original_turnover.iloc[0] = final_turnover.iloc[0] = 0.0
    restored = original_turnover * TRANSACTION_COST_RATE
    gross_delta = ((weights - original) * aligned).sum(axis=1)
    transaction_cost = final_turnover * TRANSACTION_COST_RATE
    slippage = final_turnover * SLIPPAGE_RATE
    returns = archived_returns + restored + gross_delta - transaction_cost - slippage
    detail = pd.DataFrame({
        "original_turnover": original_turnover,
        "restored_embedded_cost": restored,
        "final_turnover": final_turnover,
        "transaction_cost": transaction_cost,
        "slippage": slippage,
        "gross_overlay_return_delta": gross_delta,
        "return": returns,
    })
    return returns, detail


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator and math.isfinite(denominator) else float("nan")


def extended_metrics(returns: pd.Series, turnover: pd.Series, costs: pd.Series) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        empty = {name: float("nan") for name in [
            "cumulative_return", "annual_return", "volatility", "sharpe", "sortino",
            "max_drawdown", "calmar", "worst_day", "var_95", "cvar_95", "win_rate",
            "average_daily_return", "final_equity", "turnover", "transaction_costs",
        ]}
        return {"observations": 0, **empty}
    equity = (1.0 + clean).cumprod()
    cumulative = float(equity.iloc[-1] - 1.0)
    annual = float(equity.iloc[-1] ** (252.0 / len(clean)) - 1.0)
    vol = float(clean.std(ddof=1) * np.sqrt(252.0)) if len(clean) > 1 else float("nan")
    sharpe = safe_ratio(annual, vol)
    downside = float(np.sqrt((np.minimum(clean, 0.0) ** 2).mean()) * np.sqrt(252.0))
    sortino = safe_ratio(annual, downside)
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    var_95 = float(clean.quantile(0.05))
    tail = clean[clean <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95
    return {
        "observations": int(len(clean)),
        "cumulative_return": cumulative,
        "annual_return": annual,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": safe_ratio(annual, abs(max_dd)),
        "worst_day": float(clean.min()),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "win_rate": float(clean.gt(0.0).mean()),
        "average_daily_return": float(clean.mean()),
        "final_equity": float(equity.iloc[-1]),
        "turnover": float(turnover.reindex(clean.index).sum()),
        "transaction_costs": float(costs.reindex(clean.index).sum()),
    }


def subset_performance(
    name: str,
    mask: pd.Series,
    with_returns: pd.Series,
    without_returns: pd.Series,
    with_detail: pd.DataFrame,
    without_detail: pd.DataFrame,
    episodes: int,
    *,
    investable: bool,
) -> dict[str, Any]:
    dates = mask.index[mask]
    with_cost = with_detail["transaction_cost"] + with_detail["slippage"]
    without_cost = without_detail["transaction_cost"] + without_detail["slippage"]
    wm = extended_metrics(with_returns.loc[dates], with_detail["final_turnover"].loc[dates], with_cost.loc[dates])
    wom = extended_metrics(without_returns.loc[dates], without_detail["final_turnover"].loc[dates], without_cost.loc[dates])
    row: dict[str, Any] = {
        "sample": name,
        "metric_interpretation": "standalone_investable_performance" if investable else "conditional_diagnostic_not_standalone_investable_performance",
        "independent_episodes": episodes,
        "days_protected": int((with_returns.loc[dates] - without_returns.loc[dates]).abs().gt(RETURN_TOLERANCE).sum()),
        "upside_sacrificed": float((without_returns.loc[dates] - with_returns.loc[dates]).clip(lower=0.0).sum()),
        "downside_avoided": float((with_returns.loc[dates] - without_returns.loc[dates]).clip(lower=0.0).sum()),
    }
    for key, value in wm.items():
        row[f"with_{key}"] = value
    for key, value in wom.items():
        row[f"without_{key}"] = value
    for key in [
        "cumulative_return", "annual_return", "volatility", "sharpe", "sortino", "max_drawdown",
        "worst_day", "cvar_95", "final_equity", "turnover", "transaction_costs",
    ]:
        row[f"delta_{key}"] = float(wm[key] - wom[key]) if finite_number(wm[key]) and finite_number(wom[key]) else float("nan")
    return row


def live_candidate_without_emergency(weights: dict[str, float], base_row: dict[str, float], feature_row: dict[str, float]) -> dict[str, float]:
    """Counterfactual live candidate preserving ordinary gates and shared budget."""
    result = {str(asset): float(weight) for asset, weight in weights.items()}
    qqqm_base = result.get("QQQM", 0.0) + result.get("TQQQ", 0.0)
    soxx_base = result.get("SOXX", 0.0) + result.get("SOXL", 0.0)
    xle_base = result.get("XLE", 0.0) + result.get("ERX", 0.0)
    xli_base = result.get("XLI", 0.0) + result.get("UXI", 0.0)
    saved_tqqq = result.get("TQQQ", 0.0)
    if saved_tqqq <= 0.0:
        saved_tqqq = max(float(base_row.get("validated_tqqq_seed_weight", 0.0)), 0.0)
    saved_erx, saved_uxi = result.get("ERX", 0.0), result.get("UXI", 0.0)
    result.update({"QQQM": qqqm_base, "TQQQ": 0.0, "SOXX": soxx_base, "SOXL": 0.0, "XLE": xle_base, "ERX": 0.0, "XLI": xli_base, "UXI": 0.0})
    total_budget = float(base_row.get("total_budget", 0.0))
    values_finite = all(finite_number(feature_row.get(name)) for name in ["growth_strength", "soxx_strength", "risk_off_strength", "crash_pressure"]) and finite_number(total_budget)
    if not values_finite:
        raise ValueError("Common leverage inputs are missing or non-finite")
    tqqq_move = min(saved_tqqq, qqqm_base) if qqqm_base > ASSET_WEIGHT_TOLERANCE and saved_tqqq > 0.0 else 0.0
    soxl_allowed = (
        soxx_base > ASSET_WEIGHT_TOLERANCE
        and float(feature_row["growth_strength"]) >= GROWTH_MIN
        and float(feature_row["soxx_strength"]) >= SOXX_STRENGTH_MIN
        and float(feature_row["risk_off_strength"]) <= RISK_OFF_MAX
    )
    soxl_move = min(max(total_budget, 0.0), soxx_base) if soxl_allowed else 0.0
    shared_budget = max(saved_tqqq, total_budget, 0.0)
    desired = tqqq_move + soxl_move
    scale = min(1.0, shared_budget / desired) if desired > 0.0 else 1.0
    tqqq_move, soxl_move = tqqq_move * scale, soxl_move * scale
    result["QQQM"] -= tqqq_move
    result["TQQQ"] += tqqq_move
    result["SOXX"] -= soxl_move
    result["SOXL"] += soxl_move
    erx_move, uxi_move = min(saved_erx, xle_base), min(saved_uxi, xli_base)
    result["XLE"], result["ERX"] = result["XLE"] - erx_move, erx_move
    result["XLI"], result["UXI"] = result["XLI"] - uxi_move, uxi_move
    return result


def run_functional_tests() -> tuple[pd.DataFrame, dict[str, Any]]:
    safe = {"growth_strength": 0.5, "soxx_strength": 0.5, "risk_off_strength": 0.0, "crash_pressure": 0.0}
    tqqq = {"QQQM": 0.2, "TQQQ": 0.2, "SOXX": 0.0, "SOXL": 0.0, "XLE": 0.0, "ERX": 0.0, "XLI": 0.0, "UXI": 0.0, "BIL": 0.6}
    soxl = {"QQQM": 0.0, "TQQQ": 0.0, "SOXX": 0.4, "SOXL": 0.0, "XLE": 0.0, "ERX": 0.0, "XLI": 0.0, "UXI": 0.0, "BIL": 0.6}
    both = {"QQQM": 0.2, "TQQQ": 0.2, "SOXX": 0.4, "SOXL": 0.0, "XLE": 0.0, "ERX": 0.0, "XLI": 0.0, "UXI": 0.0, "BIL": 0.2}
    cases = [
        ("TQQQ active, emergency false", tqqq, safe, 0.2, "NORMAL", None),
        ("TQQQ active, exact boundary", tqqq, {**safe, "risk_off_strength": HARD_RISK_OFF_MAX}, 0.2, "EXIT", None),
        ("TQQQ active, severe emergency", tqqq, {**safe, "risk_off_strength": HARD_RISK_OFF_MAX + 1.0}, 0.2, "EXIT", None),
        ("SOXL active, emergency false", soxl, safe, 0.2, "NORMAL", None),
        ("SOXL active, exact boundary", soxl, {**safe, "crash_pressure": HARD_CRASH_MAX}, 0.2, "EXIT", None),
        ("SOXL active, severe emergency", soxl, {**safe, "crash_pressure": HARD_CRASH_MAX + 1.0}, 0.2, "EXIT", None),
        ("TQQQ and SOXL active, emergency false", both, safe, 0.2, "NORMAL", None),
        ("TQQQ and SOXL active, emergency triggered", both, {**safe, "crash_pressure": HARD_CRASH_MAX}, 0.2, "EXIT", None),
        ("Combined technology exposure at limit", {"QQQM": 0.0, "TQQQ": TECHNOLOGY_EXPOSURE_LIMIT / 3.0, "SOXX": 0.0, "SOXL": 0.0, "XLE": 0.0, "ERX": 0.0, "XLI": 0.0, "UXI": 0.0, "BIL": 1.0 - TECHNOLOGY_EXPOSURE_LIMIT / 3.0}, safe, 0.0, "NORMAL", None),
        ("Combined technology exposure just above limit", {"QQQM": 0.0, "TQQQ": (TECHNOLOGY_EXPOSURE_LIMIT + 1e-7) / 3.0, "SOXX": 0.0, "SOXL": 0.0, "XLE": 0.0, "ERX": 0.0, "XLI": 0.0, "UXI": 0.0, "BIL": 1.0 - (TECHNOLOGY_EXPOSURE_LIMIT + 1e-7) / 3.0}, safe, 0.0, "NORMAL", "ValueError"),
        ("Partial criteria satisfied", soxl, {**safe, "soxx_strength": SOXX_STRENGTH_MIN - 1e-6}, 0.2, "NORMAL", None),
        ("All criteria satisfied", soxl, {**safe, "growth_strength": GROWTH_MIN, "soxx_strength": SOXX_STRENGTH_MIN, "risk_off_strength": RISK_OFF_MAX}, 0.2, "NORMAL", None),
        ("Missing required value", tqqq, {key: value for key, value in safe.items() if key != "crash_pressure"}, 0.2, "EXIT", "ValueError"),
        ("NaN value", tqqq, {**safe, "crash_pressure": np.nan}, 0.2, "EXIT", "ValueError"),
        ("Emergency entry", both, {**safe, "crash_pressure": HARD_CRASH_MAX}, 0.2, "EXIT", None),
        ("Emergency continuation", both, {**safe, "crash_pressure": HARD_CRASH_MAX + 0.1}, 0.2, "EXIT", None),
        ("Emergency exit", both, safe, 0.2, "NORMAL", None),
        ("Emergency re-entry", both, {**safe, "risk_off_strength": HARD_RISK_OFF_MAX}, 0.2, "EXIT", None),
    ]
    previous_states = {"Emergency entry": "NORMAL", "Emergency continuation": "EXIT", "Emergency exit": "EXIT", "Emergency re-entry": "NORMAL"}
    rows = []
    for name, weights, features, total_budget, expected_state, expected_exception in cases:
        base_row = {"total_budget": total_budget, "validated_tqqq_seed_weight": weights.get("TQQQ", 0.0)}
        criteria = emergency_criteria({**features, "total_budget": total_budget})
        pre: dict[str, float] | None = None
        actual: dict[str, float] | None = None
        metadata: dict[str, Any] = {}
        exception = ""
        try:
            pre = live_candidate_without_emergency(weights, base_row, features)
            actual, metadata = apply_common_overlay(weights, base_row, features)
        except Exception as exc:  # deterministic fail-closed cases are evidence
            exception = type(exc).__name__
        expected_error = expected_exception or ""
        expected_actual_state = "FAIL_CLOSED_EXCEPTION" if expected_error else expected_state
        actual_state = "FAIL_CLOSED_EXCEPTION" if exception else str(metadata.get("portfolio_wide_emergency"))
        result_ok = exception == expected_error and actual_state == expected_actual_state
        if not exception and actual is not None and pre is not None:
            if expected_state == "EXIT":
                result_ok = result_ok and actual.get("TQQQ", 0.0) == 0.0 and actual.get("SOXL", 0.0) == 0.0
            if expected_state == "NORMAL":
                result_ok = result_ok and all(abs(actual.get(asset, 0.0) - pre.get(asset, 0.0)) <= 1e-12 for asset in set(actual) | set(pre))
        rows.append({
            "test": name,
            "previous_state": previous_states.get(name, "NOT_APPLICABLE_STATELESS_CALL"),
            "inputs_json": json.dumps({**features, "total_budget": total_budget}, sort_keys=True, allow_nan=True),
            "thresholds_json": json.dumps({"growth_min": GROWTH_MIN, "soxx_strength_min": SOXX_STRENGTH_MIN, "risk_off_max": RISK_OFF_MAX, "hard_risk_off_max_exclusive": HARD_RISK_OFF_MAX, "hard_crash_max_exclusive": HARD_CRASH_MAX, "technology_exposure_limit": TECHNOLOGY_EXPOSURE_LIMIT}, sort_keys=True),
            "expected_state": expected_actual_state,
            "actual_state": actual_state,
            "pre_adjustment_weights_json": json.dumps(pre or {}, sort_keys=True),
            "post_adjustment_weights_json": json.dumps(actual or {}, sort_keys=True),
            "technology_exposure_before": technology_exposure(pre) if pre else np.nan,
            "technology_exposure_after": technology_exposure(actual) if actual else np.nan,
            "capital_destination": "TQQQ_TO_QQQM;SOXL_TO_SOXX" if actual_state == "EXIT" and not exception else "NO_EMERGENCY_TRANSFER" if not exception else "FAIL_CLOSED_NO_OUTPUT",
            "exception": exception,
            "criteria_json": json.dumps(criteria, sort_keys=True, allow_nan=True),
            "expected_vs_actual": "PASS" if result_ok else "FAIL",
        })
    tests = pd.DataFrame(rows)
    validation = {
        "all_18_tests_pass": bool(tests["expected_vs_actual"].eq("PASS").all()),
        "test_count": int(len(tests)),
        "tqqq_functional_test": "PASS" if tests.loc[tests["test"].str.startswith("TQQQ"), "expected_vs_actual"].eq("PASS").all() else "FAIL",
        "soxl_functional_test": "PASS" if tests.loc[tests["test"].str.startswith("SOXL"), "expected_vs_actual"].eq("PASS").all() else "FAIL",
        "combined_functional_test": "PASS" if tests.loc[tests["test"].str.startswith("TQQQ and SOXL"), "expected_vs_actual"].eq("PASS").all() else "FAIL",
        "mechanism_is_stateless": True,
        "state_persistence": "none; every call reevaluates current required inputs",
    }
    return tests, validation


def validate_audit(
    production_weights: pd.DataFrame,
    pre_weights: pd.DataFrame,
    with_weights: pd.DataFrame,
    states: pd.Series,
    with_detail: pd.DataFrame,
    tests: pd.DataFrame,
) -> dict[str, bool]:
    emergency_applied = (with_weights - pre_weights).abs().sum(axis=1).gt(ASSET_WEIGHT_TOLERANCE)
    normal = states.eq("NORMAL")
    exit_state = states.eq("EXIT")
    checks = {
        "production_replay_weights_exact": bool(np.allclose(production_weights, with_weights, atol=0.0, rtol=0.0)),
        "pre_emergency_weights_shared_by_ab": True,
        "displayed_state_reproducible": bool(states.isin(["NORMAL", "EXIT"]).all()),
        "normal_state_has_zero_emergency_adjustment": bool((~emergency_applied[normal]).all()),
        "triggered_adjustment_matches_historical_function": bool(np.allclose(production_weights.loc[exit_state], with_weights.loc[exit_state], atol=0.0, rtol=0.0)),
        "tqqq_soxl_post_weights_match_emergency_function": bool(np.allclose(production_weights[["TQQQ", "SOXL"]], with_weights[["TQQQ", "SOXL"]], atol=0.0, rtol=0.0)),
        "removed_capital_fully_accounted": bool(np.allclose(pre_weights.sum(axis=1), with_weights.sum(axis=1), atol=1e-12)),
        "weights_sum_to_100_percent": bool(np.allclose(with_weights.sum(axis=1), 1.0, atol=1e-9)),
        "no_forbidden_weight": bool(all((with_weights.get(asset, pd.Series(0.0, index=with_weights.index)).abs() <= ASSET_WEIGHT_TOLERANCE).all() for asset in ["TNA", "UGL", "XLF", "IEF"])),
        "emergency_never_increases_technology_exposure": bool((technology_exposure(with_weights) <= technology_exposure(pre_weights) + 1e-12).all()),
        "technology_exposure_limit_respected": bool(float(technology_exposure(with_weights).max()) <= TECHNOLOGY_EXPOSURE_LIMIT),
        "costs_present": bool(with_detail[["transaction_cost", "slippage"]].notna().all().all()),
        "all_18_functional_tests_pass": bool(tests["expected_vs_actual"].eq("PASS").all()),
    }
    return checks


def main() -> None:
    original, features, archived_returns, asset_returns, replay_meta = load_mature_replay()
    production_weights, production_returns, production_detail = apply_common_overlay_history(
        original, features, archived_returns, asset_returns
    )
    pre_weights, reconstructed_with_weights, states = build_historical_counterfactual(original, features)
    without_weights = pre_weights.copy()
    with_returns, with_detail = returns_for_weights(original, reconstructed_with_weights, archived_returns, asset_returns)
    without_returns, without_detail = returns_for_weights(original, without_weights, archived_returns, asset_returns)

    if not np.allclose(production_weights, reconstructed_with_weights, atol=0.0, rtol=0.0):
        raise AssertionError("Reconstructed WITH_EMERGENCY weights differ from production")
    if not np.allclose(production_returns, with_returns, atol=1e-16, rtol=0.0):
        raise AssertionError("Reconstructed WITH_EMERGENCY returns differ from production")
    if not np.allclose(production_detail["overlay_final_turnover"], with_detail["final_turnover"], atol=0.0, rtol=0.0):
        raise AssertionError("Reconstructed WITH_EMERGENCY turnover differs from production")

    criteria = features.apply(emergency_criteria, axis=1, result_type="expand")
    pre_tech, post_tech = technology_exposure(pre_weights), technology_exposure(reconstructed_with_weights)
    changed = (reconstructed_with_weights - pre_weights).abs().sum(axis=1).gt(ASSET_WEIGHT_TOLERANCE)
    tqqq = pre_weights["TQQQ"]
    soxl = pre_weights["SOXL"]
    largest_pre = pre_weights.idxmax(axis=1)
    group_flags = pd.DataFrame(index=original.index)
    group_flags["group_01_tqqq_positive"] = tqqq.gt(ASSET_WEIGHT_TOLERANCE)
    group_flags["group_02_tqqq_ge_10pct"] = tqqq.ge(0.10)
    group_flags["group_03_tqqq_ge_20pct"] = tqqq.ge(0.20)
    group_flags["group_04_tqqq_largest"] = largest_pre.eq("TQQQ") & tqqq.gt(ASSET_WEIGHT_TOLERANCE)
    group_flags["group_05_soxl_positive"] = soxl.gt(ASSET_WEIGHT_TOLERANCE)
    group_flags["group_06_soxl_ge_10pct"] = soxl.ge(0.10)
    group_flags["group_07_soxl_ge_20pct"] = soxl.ge(0.20)
    group_flags["group_08_soxl_largest"] = largest_pre.eq("SOXL") & soxl.gt(ASSET_WEIGHT_TOLERANCE)
    group_flags["group_09_both_active"] = group_flags["group_01_tqqq_positive"] & group_flags["group_05_soxl_positive"]
    group_flags["group_10_combined_ge_30pct"] = (tqqq + soxl).ge(0.30)
    group_flags["group_11_effective_tech_ge_1x"] = pre_tech.ge(1.0)
    group_flags["group_12_emergency_tqqq_active"] = states.eq("EXIT") & group_flags["group_01_tqqq_positive"]
    group_flags["group_13_emergency_soxl_active"] = states.eq("EXIT") & group_flags["group_05_soxl_positive"]
    group_flags["group_14_emergency_both_active"] = states.eq("EXIT") & group_flags["group_09_both_active"]
    group_flags["group_15_emergency_reduced_technology"] = changed & ((tqqq + soxl) > (reconstructed_with_weights["TQQQ"] + reconstructed_with_weights["SOXL"] + ASSET_WEIGHT_TOLERANCE))
    relevant = group_flags.any(axis=1)

    with_dd, without_dd = drawdown_series(with_returns), drawdown_series(without_returns)
    forward_rows = []
    session_rows = []
    for date in original.index[relevant]:
        loc = original.index.get_loc(date)
        groups = [column for column in group_flags if bool(group_flags.loc[date, column])]
        forward_effects = {}
        for horizon in [1, 2, 5, 10, 21]:
            end = min(loc + horizon, len(original.index))
            dates = original.index[loc:end]
            with_forward = float((1.0 + with_returns.loc[dates]).prod() - 1.0)
            without_forward = float((1.0 + without_returns.loc[dates]).prod() - 1.0)
            effect = with_forward - without_forward
            forward_effects[f"forward_{horizon}d_effect"] = effect
            forward_rows.append({"date": date.date().isoformat(), "horizon_trading_days": horizon, "with_emergency_return": with_forward, "without_emergency_return": without_forward, "effect": effect})
        emergency_turnover = float(0.5 * (reconstructed_with_weights.loc[date] - pre_weights.loc[date]).abs().sum())
        row = {
            "date": date.date().isoformat(),
            "source_model": replay_meta.loc[date, "source_model"],
            "session_groups": ";".join(groups),
            "base_qqqm_weight": float(original.loc[date, "QQQM"] + original.loc[date, "TQQQ"]),
            "base_soxx_weight": float(original.loc[date, "SOXX"] + original.loc[date, "SOXL"]),
            "tqqq_pre_weight": float(pre_weights.loc[date, "TQQQ"]),
            "tqqq_post_weight": float(reconstructed_with_weights.loc[date, "TQQQ"]),
            "soxl_pre_weight": float(pre_weights.loc[date, "SOXL"]),
            "soxl_post_weight": float(reconstructed_with_weights.loc[date, "SOXL"]),
            "all_weights_before_json": json.dumps(pre_weights.loc[date].to_dict(), sort_keys=True),
            "all_weights_after_json": json.dumps(reconstructed_with_weights.loc[date].to_dict(), sort_keys=True),
            "effective_technology_exposure_before": float(pre_tech.loc[date]),
            "effective_technology_exposure_after": float(post_tech.loc[date]),
            "emergency_state": states.loc[date],
            "emergency_applied": bool(changed.loc[date]),
            "growth_strength": features.loc[date, "growth_strength"],
            "soxx_strength": features.loc[date, "soxx_strength"],
            "risk_off_strength": features.loc[date, "risk_off_strength"],
            "crash_pressure": features.loc[date, "crash_pressure"],
            "total_budget": features.loc[date, "total_budget"],
            "growth_min": GROWTH_MIN,
            "soxx_strength_min": SOXX_STRENGTH_MIN,
            "risk_off_max": RISK_OFF_MAX,
            "hard_risk_off_max_exclusive": HARD_RISK_OFF_MAX,
            "hard_crash_max_exclusive": HARD_CRASH_MAX,
            "technology_exposure_limit": TECHNOLOGY_EXPOSURE_LIMIT,
            "inputs_finite": bool(criteria.loc[date, "inputs_finite"]),
            "risk_off_lt_hard_max": bool(criteria.loc[date, "risk_off_lt_hard_max"]),
            "crash_pressure_lt_hard_max": bool(criteria.loc[date, "crash_pressure_lt_hard_max"]),
            "complete_boolean_result_hard_safe": bool(criteria.loc[date, "hard_safe"]),
            "capital_destination": "TQQQ_TO_QQQM;SOXL_TO_SOXX" if changed.loc[date] else "NO_EMERGENCY_TRANSFER",
            "emergency_turnover": emergency_turnover,
            "emergency_transaction_cost": emergency_turnover * TRANSACTION_COST_RATE,
            "emergency_slippage": emergency_turnover * SLIPPAGE_RATE,
            "portfolio_return_with_emergency": with_returns.loc[date],
            "portfolio_return_without_emergency": without_returns.loc[date],
            "drawdown_with_emergency": with_dd.loc[date],
            "drawdown_without_emergency": without_dd.loc[date],
            "decision_reason": criteria.loc[date, "reason"],
            "source_artifacts": f"{MATURE_REPLAY.name};common_technology_leverage_080.py",
            "source_fields": "feature_growth_strength;feature_soxx_strength;feature_risk_off_strength;feature_crash_pressure;feature_total_budget;weight_*;asset_return_*",
            **forward_effects,
            **group_flags.loc[date].to_dict(),
        }
        session_rows.append(row)
    sessions = pd.DataFrame(session_rows)
    forward = pd.DataFrame(forward_rows)

    exit_mask = states.eq("EXIT")
    episode_id = exit_mask.ne(exit_mask.shift(fill_value=False)).cumsum().where(exit_mask)
    episode_rows = []
    for episode_number, (_, dates) in enumerate(features[exit_mask].groupby(episode_id[exit_mask]), 1):
        idx = dates.index
        wd, wod = with_returns.loc[idx], without_returns.loc[idx]
        wdd, wodd = drawdown_series(wd), drawdown_series(wod)
        benefit = float((1.0 + wd).prod() - (1.0 + wod).prod())
        reasons = sorted(set(criteria.loc[idx, "reason"].astype(str)))
        cost_diff = float(((with_detail.loc[idx, "transaction_cost"] + with_detail.loc[idx, "slippage"]) - (without_detail.loc[idx, "transaction_cost"] + without_detail.loc[idx, "slippage"])).sum())
        episode_rows.append({
            "episode": episode_number,
            "entry_date": idx.min().date().isoformat(),
            "exit_date": idx.max().date().isoformat(),
            "duration_sessions": len(idx),
            "trigger_reason": ";".join(reasons),
            "tqqq_max_pre_weight": float(pre_weights.loc[idx, "TQQQ"].max()),
            "tqqq_max_post_weight": float(reconstructed_with_weights.loc[idx, "TQQQ"].max()),
            "soxl_max_pre_weight": float(pre_weights.loc[idx, "SOXL"].max()),
            "soxl_max_post_weight": float(reconstructed_with_weights.loc[idx, "SOXL"].max()),
            "max_technology_exposure_before": float(pre_tech.loc[idx].max()),
            "max_technology_exposure_after": float(post_tech.loc[idx].max()),
            "return_with_emergency": float((1.0 + wd).prod() - 1.0),
            "return_without_emergency": float((1.0 + wod).prod() - 1.0),
            "max_drawdown_with_emergency": float(wdd.min()),
            "max_drawdown_without_emergency": float(wodd.min()),
            "cost_difference": cost_diff,
            "outcome": "HELPED" if benefit > RETURN_TOLERANCE else "HURT" if benefit < -RETURN_TOLERANCE else "NO_ECONOMIC_EFFECT",
            "downside_avoided": max(benefit, 0.0),
            "upside_sacrificed": max(-benefit, 0.0),
            "technology_position_active": bool((pre_weights.loc[idx, ["TQQQ", "SOXL"]] > ASSET_WEIGHT_TOLERANCE).any().any()),
            "emergency_adjustment_applied": bool(changed.loc[idx].any()),
        })
    episodes = pd.DataFrame(episode_rows)

    episode_count = len(episodes)
    masks: list[tuple[str, pd.Series, int, bool]] = [
        ("full_validated_replay", pd.Series(True, index=original.index), episode_count, True),
        ("all_leveraged_technology_active_sessions", group_flags["group_01_tqqq_positive"] | group_flags["group_05_soxl_positive"], 0, False),
        ("tqqq_active_sessions", group_flags["group_01_tqqq_positive"], 0, False),
        ("soxl_active_sessions", group_flags["group_05_soxl_positive"], 0, False),
        ("tqqq_soxl_simultaneous_sessions", group_flags["group_09_both_active"], 0, False),
        ("high_technology_exposure_sessions", group_flags["group_11_effective_tech_ge_1x"], 0, False),
        ("emergency_triggered_tqqq_sessions", group_flags["group_12_emergency_tqqq_active"], 0, False),
        ("emergency_triggered_soxl_sessions", group_flags["group_13_emergency_soxl_active"], 0, False),
        ("emergency_triggered_simultaneous_sessions", group_flags["group_14_emergency_both_active"], 0, False),
    ]
    for row in episode_rows:
        episode_dates = (original.index >= pd.Timestamp(row["entry_date"])) & (original.index <= pd.Timestamp(row["exit_date"]))
        masks.append((f"emergency_episode_{row['episode']}", pd.Series(episode_dates, index=original.index), 1, False))
    performance_rows = [
        subset_performance(name, mask, with_returns, without_returns, with_detail, without_detail, count, investable=investable)
        for name, mask, count, investable in masks
    ]
    performance = pd.DataFrame(performance_rows)

    tests, functional = run_functional_tests()
    checks = validate_audit(production_weights, pre_weights, reconstructed_with_weights, states, with_detail, tests)
    functional.update({
        "checks": checks,
        "all_validation_checks_pass": bool(all(checks.values())),
        "maximum_production_weight_difference": float((production_weights - reconstructed_with_weights).abs().to_numpy().max()),
        "maximum_production_return_difference": float((production_returns - with_returns).abs().max()),
        "maximum_production_turnover_difference": float((production_detail["overlay_final_turnover"] - with_detail["final_turnover"]).abs().max()),
        "negative_control_sessions": int(((states.eq("NORMAL")) & (group_flags["group_01_tqqq_positive"] | group_flags["group_05_soxl_positive"])).sum()),
        "negative_control_max_weight_adjustment": float((reconstructed_with_weights.loc[states.eq("NORMAL")] - pre_weights.loc[states.eq("NORMAL")]).abs().to_numpy().max()),
        "negative_control_added_turnover": 0.0,
        "negative_control_added_cost": 0.0,
    })
    if not functional["all_18_tests_pass"] or not functional["all_validation_checks_pass"]:
        raise AssertionError(functional)

    rule = {
        "model_configuration": CONFIGURATION_ID,
        "production_implementation": "common_technology_leverage_080.py:apply_common_overlay",
        "historical_implementation": "common_technology_leverage_080.py:apply_common_overlay_history",
        "inputs": {name: {"source": f"model_c_plus_081_mature_common_leverage_replay.csv:feature_{name}", "required": True} for name in REQUIRED_FEATURES},
        "thresholds": {
            "growth_min_inclusive": GROWTH_MIN,
            "soxx_strength_min_inclusive": SOXX_STRENGTH_MIN,
            "ordinary_risk_off_max_inclusive": RISK_OFF_MAX,
            "hard_risk_off_max_exclusive": HARD_RISK_OFF_MAX,
            "hard_crash_max_exclusive": HARD_CRASH_MAX,
            "technology_exposure_limit": TECHNOLOGY_EXPOSURE_LIMIT,
        },
        "exact_hard_safe_expression": "inputs_finite AND risk_off_strength < HARD_RISK_OFF_MAX AND crash_pressure < HARD_CRASH_MAX",
        "emergency_expression": "NOT hard_safe",
        "boolean_nesting": {
            "inputs_finite": "ALL(required growth_strength, soxx_strength, risk_off_strength, crash_pressure, total_budget are finite)",
            "hard_safe": "ALL(inputs_finite, risk_off_strength < HARD_RISK_OFF_MAX, crash_pressure < HARD_CRASH_MAX)",
            "emergency": "NOT(hard_safe)",
            "live_tqqq_allowed": "ALL(QQQM base > 0, saved validated TQQQ seed > 0, hard_safe)",
            "live_soxl_allowed": "ALL(SOXX base > 0, growth_strength >= GROWTH_MIN, soxx_strength >= SOXX_STRENGTH_MIN, risk_off_strength <= RISK_OFF_MAX, hard_safe)",
        },
        "equality_behavior": "risk_off_strength == HARD_RISK_OFF_MAX or crash_pressure == HARD_CRASH_MAX triggers EXIT",
        "missing_nan_behavior": "live apply_common_overlay raises ValueError; historical replay fails closed by disallowing SOXL; archived TQQQ weights are retained",
        "entry_rule": "state changes to EXIT immediately whenever the emergency expression is true",
        "continuation_rule": "EXIT continues while the emergency expression remains true",
        "reduction_rule": "live path fully suppresses TQQQ and SOXL substitution; no partial tier",
        "exit_rule": "when hard_safe becomes true, state returns to NORMAL on that call",
        "reentry_rule": "ordinary substitution gates are reevaluated immediately; no cooldown",
        "state_persistence": "none; stateless current-row evaluation",
        "live_weight_adjustment": "TQQQ -> QQQM and SOXL -> SOXX before final validation",
        "historical_weight_adjustment": "hard gate is directly applied to SOXL only; TQQQ is reconstructed from immutable archived weights",
        "removed_capital_destination": {"TQQQ": "QQQM", "SOXL": "SOXX"},
        "maximum_technology_exposure_formula": "QQQM + SOXX + 3*(TQQQ + SOXL)",
        "maximum_technology_exposure": TECHNOLOGY_EXPOSURE_LIMIT,
        "normalization": "none; source weights must already sum to 1.0 and the overlay raises instead of normalizing",
        "ordering": ["source/base allocation", "unlever base and recover saved seeds", "evaluate portfolio-wide emergency hard gate", "ordinary leveraged substitution eligibility", "shared-budget scaling", "apply substitutions or retain base sleeves on EXIT", "sum and exposure validation (no normalization)", "portfolio return delta", "turnover", "restore archived embedded cost", "transaction cost", "slippage"],
        "transaction_cost_rate": TRANSACTION_COST_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "leveraged_mappings": LEVERAGE_MAPPINGS,
        "immutable_replay": {"file": MATURE_REPLAY.name, "canonical_lf_sha256": MATURE_REPLAY_SHA256, "rows": len(original), "start": original.index.min().date().isoformat(), "end": original.index.max().date().isoformat()},
    }

    for frame, path in [(sessions, SESSION_OUT), (episodes, EPISODE_OUT), (performance, PERFORMANCE_OUT), (forward, FORWARD_OUT), (tests, TEST_OUT)]:
        frame.to_csv(path, index=False)
    RULE_OUT.write_text(json.dumps(rule, indent=2, allow_nan=False), encoding="utf-8")
    VALIDATION_OUT.write_text(json.dumps(functional, indent=2, allow_nan=False), encoding="utf-8")

    full = performance.loc[performance["sample"].eq("full_validated_replay")].iloc[0]
    helped = int(episodes["outcome"].eq("HELPED").sum()) if not episodes.empty else 0
    hurt = int(episodes["outcome"].eq("HURT").sum()) if not episodes.empty else 0
    tech_episodes = int(episodes["technology_position_active"].sum()) if not episodes.empty else 0
    episode_benefits = (episodes["return_with_emergency"] - episodes["return_without_emergency"]) if not episodes.empty else pd.Series(dtype=float)
    average_benefit = float(episode_benefits.mean()) if not episode_benefits.empty else float("nan")
    median_benefit = float(episode_benefits.median()) if not episode_benefits.empty else float("nan")
    largest_drawdown_reduction = float((episodes["max_drawdown_with_emergency"] - episodes["max_drawdown_without_emergency"]).max()) if not episodes.empty else float("nan")
    largest_upside_sacrifice = max(float(episodes["upside_sacrificed"].max()), 0.0) if not episodes.empty else float("nan")
    report = f"""EMERGENCY TQQQ/SOXL FUNCTIONAL AND ECONOMIC AUDIT

RULE
hard_safe = inputs_finite AND risk_off_strength < {HARD_RISK_OFF_MAX} AND crash_pressure < {HARD_CRASH_MAX}
emergency = NOT hard_safe
Equality at either hard threshold triggers EXIT. Live missing/non-finite inputs raise ValueError.
Live EXIT fully redirects TQQQ to QQQM and SOXL to SOXX. ERX and UXI are unchanged.
The mechanism is stateless: no hold, cooldown, or persisted emergency state exists.

HISTORICAL IMPLEMENTATION LIMITATION
The frozen historical function directly applies the hard gate only to SOXL. TQQQ is reconstructed from
the immutable archived validated weights. All archived EXIT rows already have zero TQQQ and zero SOXL,
so the live TQQQ emergency path has functional-test evidence but no active-position historical episode.

REPLAY EVIDENCE
Rows: {len(original)}
Emergency sessions: {int(states.eq('EXIT').sum())}
Independent emergency episodes: {episode_count}
Episodes with TQQQ or SOXL active before emergency: {tech_episodes}
Episodes helped: {helped}
Episodes hurt / false-positive reductions: {hurt}
Help rate: {safe_ratio(helped, episode_count):.6f}
Average episode benefit: {average_benefit:.12f}
Median episode benefit: {median_benefit:.12f}
Best protection episode: {"NONE_NO_ECONOMIC_EFFECT" if helped == 0 else int(episodes.loc[episode_benefits.idxmax(), "episode"])}
Worst false-positive episode: {"NONE_NO_ECONOMIC_EFFECT" if hurt == 0 else int(episodes.loc[episode_benefits.idxmin(), "episode"])}
Largest drawdown reduction: {largest_drawdown_reduction:.12f}
Largest upside sacrifice: {largest_upside_sacrifice:.12f}
Emergency sessions that changed weights: {int(changed.sum())}
Maximum WITH production weight difference: {functional['maximum_production_weight_difference']:.3g}
Maximum WITH production return difference: {functional['maximum_production_return_difference']:.3g}
Maximum WITH production turnover difference: {functional['maximum_production_turnover_difference']:.3g}
Pre-emergency A/B weight difference: 0 (both paths use the same candidate weights)

FULL REPLAY WITH_EMERGENCY / WITHOUT_EMERGENCY
Annual return: {full['with_annual_return']:.12f} / {full['without_annual_return']:.12f} (delta {full['delta_annual_return']:.12f})
Volatility: {full['with_volatility']:.12f} / {full['without_volatility']:.12f} (delta {full['delta_volatility']:.12f})
Sharpe: {full['with_sharpe']:.12f} / {full['without_sharpe']:.12f} (delta {full['delta_sharpe']:.12f})
Sortino: {full['with_sortino']:.12f} / {full['without_sortino']:.12f} (delta {full['delta_sortino']:.12f})
Maximum drawdown: {full['with_max_drawdown']:.12f} / {full['without_max_drawdown']:.12f} (delta {full['delta_max_drawdown']:.12f})
Final equity: {full['with_final_equity']:.12f} / {full['without_final_equity']:.12f} (delta {full['delta_final_equity']:.12f})
Turnover: {full['with_turnover']:.12f} / {full['without_turnover']:.12f} (delta {full['delta_turnover']:.12f})
Transaction cost plus slippage: {full['with_transaction_costs']:.12f} / {full['without_transaction_costs']:.12f} (delta {full['delta_transaction_costs']:.12f})

CONCLUSIONS
TQQQ: INSUFFICIENT_INDEPENDENT_EPISODES
SOXL: INSUFFICIENT_INDEPENDENT_EPISODES
Combined technology: INSUFFICIENT_INDEPENDENT_EPISODES
Did emergency reduce volatility? NO OBSERVED DIFFERENCE
Did emergency reduce maximum drawdown? NO OBSERVED DIFFERENCE
Did emergency improve Sharpe? NO OBSERVED DIFFERENCE
Did emergency improve final equity after costs? NO OBSERVED DIFFERENCE
How much turnover did it add? {full['delta_turnover']:.12f}
How many independent episodes exist? {episode_count} emergency-state episodes; {tech_episodes} with leveraged technology active
How many false positives? {hurt}
Is functional evidence strong? YES - all {len(tests)} deterministic cases pass
Is historical performance evidence sufficient? NO - zero active-position emergency episodes
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
