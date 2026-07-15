"""Frozen Research-079d D_BOTH_CAP_1.00 technology substitution rules."""
from __future__ import annotations

import math


MODEL_ID = "034_EXECUTION_GRADE_PLUS_COMMON_TECH_LEVERAGE"
CONFIGURATION_ID = "D_BOTH_CAP_1.00"
TECHNOLOGY_EXPOSURE_LIMIT = 2.1390000001
LEVERAGE_MAPPINGS = {"QQQM": "TQQQ", "SOXX": "SOXL"}
GROWTH_MIN = -0.10
SOXX_STRENGTH_MIN = -0.10
RISK_OFF_MAX = 1.25
HARD_RISK_OFF_MAX = 1.60
HARD_CRASH_MAX = 2.50
TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.001


def number(row, name: str, default=float("nan")) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return value


def apply_common_overlay(weights: dict[str, float], base_row, feature_row, *, allow_archived_warmup: bool = False) -> tuple[dict[str, float], dict]:
    """Apply exact D_BOTH_CAP_1.00 substitution to one effective portfolio.

    TQQQ is accepted only from the saved validated QQQM producer path. SOXL
    copies the same common budget and portfolio-wide risk gates. Neither asset
    is independently ranked. Missing inputs fail closed.
    """
    result = {str(asset): float(weight) for asset, weight in weights.items()}
    required = ["growth_strength", "soxx_strength", "risk_off_strength", "crash_pressure"]
    features = {name: number(feature_row, name) for name in required}
    total_budget = number(base_row, "total_budget")
    inputs_finite = all(math.isfinite(value) for value in features.values()) and math.isfinite(total_budget)
    if not inputs_finite and not allow_archived_warmup:
        raise ValueError("Common leverage inputs are missing or non-finite")
    if not math.isfinite(total_budget):
        total_budget = 0.0

    qqqm_base = result.get("QQQM", 0.0) + result.get("TQQQ", 0.0)
    soxx_base = result.get("SOXX", 0.0) + result.get("SOXL", 0.0)
    saved_tqqq = result.get("TQQQ", 0.0)
    result["QQQM"], result["TQQQ"] = qqqm_base, 0.0
    result["SOXX"], result["SOXL"] = soxx_base, 0.0

    hard_safe = inputs_finite and features["risk_off_strength"] < HARD_RISK_OFF_MAX and features["crash_pressure"] < HARD_CRASH_MAX
    tqqq_allowed = qqqm_base > 1e-12 and saved_tqqq > 0.0 and (hard_safe or allow_archived_warmup)
    soxl_allowed = (
        soxx_base > 1e-12
        and features["growth_strength"] >= GROWTH_MIN
        and features["soxx_strength"] >= SOXX_STRENGTH_MIN
        and features["risk_off_strength"] <= RISK_OFF_MAX
        and hard_safe
    )
    tqqq_move = min(saved_tqqq, qqqm_base) if tqqq_allowed else 0.0
    soxl_move = min(max(total_budget, 0.0), soxx_base) if soxl_allowed else 0.0
    shared_budget = max(saved_tqqq, total_budget, 0.0)
    desired = tqqq_move + soxl_move
    scale = min(1.0, shared_budget / desired) if desired > 0.0 else 1.0
    tqqq_move *= scale
    soxl_move *= scale
    result["QQQM"] -= tqqq_move
    result["TQQQ"] += tqqq_move
    result["SOXX"] -= soxl_move
    result["SOXL"] += soxl_move

    total = sum(result.values())
    if abs(total - 1.0) > 1e-9 or any(weight < -1e-12 for weight in result.values()):
        raise ValueError(f"Common leverage output is invalid: total={total}")
    effective_technology = result.get("QQQM", 0.0) + result.get("SOXX", 0.0) + 3.0 * (result.get("TQQQ", 0.0) + result.get("SOXL", 0.0))
    if effective_technology > TECHNOLOGY_EXPOSURE_LIMIT:
        raise ValueError(f"Technology exposure {effective_technology} exceeds {TECHNOLOGY_EXPOSURE_LIMIT}")

    metadata = {
        "common_leverage_framework": "TQQQ_STYLE_SHARED_BUDGET",
        "common_leverage_configuration": CONFIGURATION_ID,
        "qqqm_base_weight": qqqm_base,
        "tqqq_replacement_fraction": tqqq_move / qqqm_base if qqqm_base > 0 else 0.0,
        "tqqq_substituted_weight": tqqq_move,
        "qqqm_final_weight": result.get("QQQM", 0.0),
        "soxx_base_weight": soxx_base,
        "soxl_replacement_fraction": soxl_move / soxx_base if soxx_base > 0 else 0.0,
        "soxl_substituted_weight": soxl_move,
        "soxx_final_weight": result.get("SOXX", 0.0),
        "conviction_tier": "RECOVERED_CONTINUOUS" if desired > 0 else "OFF",
        "common_leverage_budget": shared_budget,
        "effective_technology_exposure": effective_technology,
        "portfolio_wide_emergency": "NORMAL" if hard_safe else "EXIT",
        "emergency_design": "existing portfolio-wide hard risk-off/crash override",
        "leveraged_assets_rank_independently": False,
        "common_leverage_turnover_estimate": tqqq_move + soxl_move,
        "common_leverage_cost_slippage_estimate": (tqqq_move + soxl_move) * (TRANSACTION_COST_RATE + SLIPPAGE_RATE),
    }
    return result, metadata
