"""Frozen Research-079d D_BOTH_CAP_1.00 technology substitution rules."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


MODEL_ID = "034_EXECUTION_GRADE_PLUS_COMMON_TECH_LEVERAGE"
CONFIGURATION_ID = "D_BOTH_CAP_1.00"
TECHNOLOGY_EXPOSURE_LIMIT = 2.1390000001
LEVERAGE_MAPPINGS = {"QQQM": "TQQQ", "SOXX": "SOXL", "XLE": "ERX", "XLI": "UXI"}
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
    xle_base = result.get("XLE", 0.0) + result.get("ERX", 0.0)
    xli_base = result.get("XLI", 0.0) + result.get("UXI", 0.0)
    saved_tqqq = result.get("TQQQ", 0.0)
    saved_erx = result.get("ERX", 0.0)
    saved_uxi = result.get("UXI", 0.0)
    # The live 022F builder intentionally publishes an unlevered base portfolio.
    # Preserve its authoritative pre-conversion allocations in dedicated fields
    # so substitution does not depend on an already-zeroed exec_w value.
    if saved_tqqq <= 0.0:
        saved_tqqq = max(number(base_row, "validated_tqqq_seed_weight", 0.0), 0.0)
    result["QQQM"], result["TQQQ"] = qqqm_base, 0.0
    result["SOXX"], result["SOXL"] = soxx_base, 0.0
    result["XLE"], result["ERX"] = xle_base, 0.0
    result["XLI"], result["UXI"] = xli_base, 0.0

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
    erx_move = min(saved_erx, xle_base)
    uxi_move = min(saved_uxi, xli_base)
    result["XLE"] -= erx_move
    result["ERX"] += erx_move
    result["XLI"] -= uxi_move
    result["UXI"] += uxi_move

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
        "xle_base_weight": xle_base,
        "erx_replacement_fraction": erx_move / xle_base if xle_base > 0 else 0.0,
        "erx_substituted_weight": erx_move,
        "xle_final_weight": result.get("XLE", 0.0),
        "xli_base_weight": xli_base,
        "uxi_replacement_fraction": uxi_move / xli_base if xli_base > 0 else 0.0,
        "uxi_substituted_weight": uxi_move,
        "xli_final_weight": result.get("XLI", 0.0),
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


def apply_common_overlay_history(
    original: pd.DataFrame,
    features: pd.DataFrame,
    archived_returns: pd.Series,
    asset_returns: pd.DataFrame,
    *,
    transaction_cost: float = TRANSACTION_COST_RATE,
    slippage: float = SLIPPAGE_RATE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Replay exact Research-079d D_BOTH_CAP_1.00 across all dates.

    ``archived_returns`` already contains the source models' internal costs and
    corrected 034 source-switch cost. Research 079d adds back the original
    0.10% final-weight turnover estimate and then charges the complete common-
    overlay final turnover once at transaction cost plus slippage. This is the
    frozen validated accounting path and must not be simplified.
    """
    required_assets = ["QQQM", "TQQQ", "SOXX", "SOXL"]
    required_features = ["growth_strength", "soxx_strength", "risk_off_strength", "crash_pressure", "total_budget"]
    if not original.index.equals(features.index) or not original.index.equals(archived_returns.index):
        raise ValueError("Common leverage historical inputs have different indexes")
    missing_assets = [asset for asset in original.columns if asset not in asset_returns.columns]
    missing_assets += [asset for asset in required_assets if asset not in original.columns]
    missing_features = [name for name in required_features if name not in features.columns]
    if missing_assets or missing_features:
        raise ValueError(f"Common leverage historical inputs missing assets={missing_assets}, features={missing_features}")
    if asset_returns.reindex(original.index)[original.columns].isna().any().any():
        raise ValueError("Common leverage historical asset returns contain missing values")

    base = original.copy().astype(float)
    base["QQQM"] = base["QQQM"] + base["TQQQ"]
    base["TQQQ"] = 0.0
    base["SOXX"] = base["SOXX"] + base["SOXL"]
    base["SOXL"] = 0.0
    final = base.copy()

    qqqm_total = original["QQQM"] + original["TQQQ"]
    tqqq_fraction = (original["TQQQ"] / qqqm_total.replace(0.0, np.nan)).fillna(0.0)
    tqqq_move = base["QQQM"] * tqqq_fraction
    final["QQQM"] -= tqqq_move
    final["TQQQ"] += tqqq_move

    finite = features[required_features].apply(np.isfinite).all(axis=1)
    soxl_allowed = (
        base["SOXX"].gt(1e-12)
        & features["growth_strength"].ge(GROWTH_MIN)
        & features["soxx_strength"].ge(SOXX_STRENGTH_MIN)
        & features["risk_off_strength"].le(RISK_OFF_MAX)
        & features["risk_off_strength"].lt(HARD_RISK_OFF_MAX)
        & features["crash_pressure"].lt(HARD_CRASH_MAX)
        & finite
    )
    soxl_fraction = (features["total_budget"] / base["SOXX"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0).where(soxl_allowed, 0.0)
    soxl_move = base["SOXX"] * soxl_fraction
    final["SOXX"] -= soxl_move
    final["SOXL"] += soxl_move

    desired = final["TQQQ"] + final["SOXL"]
    shared_budget = pd.Series(np.maximum(original["TQQQ"], features["total_budget"].fillna(0.0)), index=original.index)
    scale = (shared_budget / desired.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    raw_tqqq, raw_soxl = final["TQQQ"].copy(), final["SOXL"].copy()
    final["TQQQ"] = raw_tqqq * scale
    final["QQQM"] += raw_tqqq - final["TQQQ"]
    final["SOXL"] = raw_soxl * scale
    final["SOXX"] += raw_soxl - final["SOXL"]

    if not np.allclose(final.sum(axis=1), 1.0, atol=1e-9) or (final < -1e-12).any().any():
        raise ValueError("Common leverage historical weights are invalid")
    technology_exposure = final["QQQM"] + final["SOXX"] + 3.0 * (final["TQQQ"] + final["SOXL"])
    if float(technology_exposure.max()) > TECHNOLOGY_EXPOSURE_LIMIT:
        raise ValueError(f"Historical technology exposure {technology_exposure.max()} exceeds {TECHNOLOGY_EXPOSURE_LIMIT}")

    aligned_asset_returns = asset_returns.reindex(index=original.index, columns=original.columns)
    gross_delta = ((final - original) * aligned_asset_returns).sum(axis=1)
    original_turnover = 0.5 * original.diff().abs().sum(axis=1)
    final_turnover = 0.5 * final.diff().abs().sum(axis=1)
    original_turnover.iloc[0] = final_turnover.iloc[0] = 0.0
    restored_embedded_cost = original_turnover * TRANSACTION_COST_RATE
    transaction_costs = final_turnover * transaction_cost
    slippage_costs = final_turnover * slippage
    adjusted = archived_returns + restored_embedded_cost + gross_delta - transaction_costs - slippage_costs
    audit = pd.DataFrame({
        "original_turnover": original_turnover,
        "restored_embedded_cost": restored_embedded_cost,
        "overlay_final_turnover": final_turnover,
        "overlay_transaction_cost": transaction_costs,
        "overlay_slippage": slippage_costs,
        "gross_overlay_return_delta": gross_delta,
        "final_overlay_return": adjusted,
        "effective_technology_exposure": technology_exposure,
    })
    return final, adjusted, audit
