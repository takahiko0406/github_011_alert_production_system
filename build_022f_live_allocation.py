"""Build a same-date live allocation from the exact recovered 022F logic."""

from pathlib import Path

import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_019a_commodity_cyclical_confirmation as r19a
import research_020b_fez_validation as r20b
import research_020c_fez_stress_guard as r20c
import research_022b_defensive_real_estate_allocation as r22b
import research_022f_calibrated_defense_validation as r22f


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model_c_plus_022F_calibrated_defense_validation_best_latest_recommendation.csv"


def main() -> None:
    cfg, cfg_message = m.load_best_cfg()
    if "CONVICTION_00051" not in cfg_message:
        raise RuntimeError(f"Required CONVICTION_00051 was not loaded: {cfg_message}")

    raw = m.load_rebalance_log()
    live_path = ROOT / "model_c_plus_transition_conviction_overlay_011_LIGHT_latest_recommendation.csv"
    live = pd.read_csv(live_path)
    if live.empty:
        raise ValueError("Empty LIGHT latest recommendation")
    live_row = live.iloc[-1].copy()
    signal_date = pd.Timestamp(live_row.get("latest_data_date", live_row.get("signal_date"))).normalize()
    live_row["date"] = signal_date

    start = (raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = r22b.download_prices_checked(start)
    market_date = pd.Timestamp(prices.index.max()).normalize()
    if signal_date not in prices.index:
        raise ValueError(f"LIGHT common date {signal_date.date()} is missing from auxiliary prices")
    auxiliary_gap = len(pd.bdate_range(signal_date + pd.Timedelta(days=1), market_date))
    if market_date < signal_date or auxiliary_gap > 1:
        raise ValueError(f"LIGHT common date {signal_date.date()} is stale versus auxiliary market date {market_date.date()}")

    combined = pd.concat([raw, pd.DataFrame([live_row])], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = m.align_rebalance_dates(combined, prices[m.PRICE_ASSETS])
    combined = m.add_transition_features(combined)
    combined = r13c.add_soxx_market_filters(combined, prices)
    combined = r14b.add_iwm_peer_features(combined, prices)
    candidates = combined[combined["aligned_date"].eq(signal_date)]
    if candidates.empty:
        raise ValueError(f"Could not build 022F live feature row for {signal_date.date()}")
    row = candidates.iloc[-1]

    realized_vol = m.build_realized_vol(prices[m.PRICE_ASSETS])
    weights, details = m.build_dynamic_exec_weights(row, cfg, realized_vol)
    for asset in r22b.EXEC_ASSETS_EXT:
        weights.setdefault(asset, 0.0)
    weights, soxx_on, soxl_on, soxl_fraction = r19a.r18c.r13e.apply_overlay(weights, row, r19a.r18c.r13h.PARAMS_013F)
    weights, iwm_on, tna_on, iwm_score = r19a.r18c.r14b.apply_iwm_tna(weights, row, r19a.r18c.r15a.PARAMS_014D)

    defensive_features = r22b.build_defensive_regime_features(prices)
    bond_features = r19a.r18c.r17d.build_daily_bond_features(prices)
    gold_features = r19a.r18a.build_daily_gold_features(prices)
    commodity_features = r19a.build_daily_commodity_features(prices)
    europe_features = r22b.r20a.build_daily_europe_features(prices)

    defensive_on = False
    defensive_regime = "blocked_by_volatility"
    defensive_destination = ""
    if pd.isna(details.get("realized_vol_now")) or float(details["realized_vol_now"]) <= r22f.BEST_VOL_MAX:
        weights, defensive_on, defensive_regime, defensive_destination = r22b.apply_regime_defensive_overlay(
            weights, defensive_features.loc[signal_date], r22f.BEST_DEFENSIVE_PARAMS
        )
    weights, tlt_on, tlt_score = r19a.r18c.r17d.apply_tlt_overlay(weights, bond_features.loc[signal_date], row, r19a.r18c.tlt_params(0.15))
    weights, gld_on, gld_score = r19a.r18a.apply_gld_overlay(weights, gold_features.loc[signal_date], row, r19a.r18c.gld_params(0.10))
    weights, commodity_on, commodity_score = r19a.apply_commodity_overlay(weights, commodity_features.loc[signal_date], row, r20b.commodity_params())
    weights, europe_on, europe_score, europe_block_reason = r20c.apply_guarded_europe_overlay(
        weights, europe_features.loc[signal_date], row, r22f.BEST_EUROPE_PARAMS, defensive_on
    )
    for asset in r22b.EXEC_ASSETS_EXT:
        weights.setdefault(asset, 0.0)
    total = sum(float(v) for v in weights.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"022F live weights sum to {total}")
    if weights.get("XLF", 0.0) > 1e-12 or weights.get("IEF", 0.0) > 1e-12:
        raise ValueError("Nonvalidated XLF or IEF received 022F live weight")

    output = {
        "model": "022F_CALIBRATED_DEFENSE_VALIDATION_BEST",
        "configuration": "CONVICTION_00051",
        "signal_date": signal_date.date().isoformat(),
        "latest_data_date": signal_date.date().isoformat(),
        "allocation_date": signal_date.date().isoformat(),
        "auxiliary_price_date": market_date.date().isoformat(),
        "auxiliary_price_gap_sessions": auxiliary_gap,
        "source_rebalance_date": str(raw["date"].max().date()),
        "defensive_regime": defensive_regime,
        "defensive_destination": defensive_destination,
        "defensive_active": bool(defensive_on),
        "soxx_overlay_active": bool(soxx_on),
        "soxl_overlay_active": False,
        "soxl_validation": "NOT_VALIDATED_DISABLED",
        "iwm_overlay_active": bool(iwm_on),
        "tlt_overlay_active": bool(tlt_on),
        "gld_overlay_active": bool(gld_on),
        "commodity_overlay_active": bool(commodity_on),
        "europe_overlay_active": bool(europe_on),
        "europe_block_reason": europe_block_reason,
        "tlt_score": tlt_score,
        "gld_score": gld_score,
        "commodity_score": commodity_score,
        "europe_score": europe_score,
        "iwm_score": iwm_score,
        **details,
    }
    # Publish the 022F source as an unlevered base while retaining the exact
    # authoritative leverage seeds for the validated substitution framework.
    # Conversion preserves 100% total weight and prevents leveraged ETFs from
    # receiving independent ranking authority.
    output["validated_tqqq_seed_weight"] = float(weights.get("TQQQ", 0.0))
    output["validated_tqqq_seed_allowed"] = bool(row.get("allowed_TQQQ", False))
    output["validated_soxl_seed_weight"] = float(weights.get("SOXL", 0.0))
    # ERX and UXI are already final outputs of the mature 011 leverage producer;
    # preserve them directly rather than reconstructing them downstream.
    mappings = {"TQQQ": "QQQM", "SOXL": "SOXX", "TNA": "IWM"}
    for asset, underlying in mappings.items():
        removed = float(weights.get(asset, 0.0))
        weights[underlying] = float(weights.get(underlying, 0.0)) + removed
        weights[asset] = 0.0
    direct_total = sum(float(weights.get(a, 0.0)) for a in r22b.EXEC_ASSETS_EXT)
    if abs(direct_total - 1.0) > 1e-8:
        raise ValueError(f"Unlevered 022F live weights sum to {direct_total}")
    for asset in sorted(r22b.EXEC_ASSETS_EXT):
        output[f"exec_w_{asset}"] = float(weights.get(asset, 0.0))
    pd.DataFrame([output]).to_csv(OUT, index=False)
    print(f"Saved {OUT.name} for {market_date.date()}")


if __name__ == "__main__":
    main()
