import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e
import research_013g_soxl_instability_brake as r13g


OUT_PREFIX = "model_c_plus_013H_final_soxx_soxl_validation"
PARAMS_013F = {
    "soxx_lead_min": 0.25,
    "qqqm_to_soxx": 0.90,
    "soxl_cap": 0.30,
    "soxl_risk_max": 0.75,
    "ma63_min": -0.05,
    "small_lead": 0.30,
    "medium_lead": 0.40,
    "large_lead": 0.50,
    "small_mom": 0.03,
    "medium_mom": 0.06,
    "large_mom": 0.10,
    "small_frac": 0.15,
    "medium_frac": 0.65,
    "large_frac": 0.80,
}
PARAMS_013G = {
    "qqqm_to_soxx": 0.90,
    "soxl_cap": 0.30,
    "small_frac": 0.15,
    "medium_frac": 0.65,
    "large_frac": 0.80,
    "transition_cut": 0.25,
    "transition_multiplier": 0.70,
    "leader_flip_cut": 0.50,
    "flip_multiplier": 0.50,
    "entropy_cut": 0.50,
    "entropy_multiplier": 0.60,
}


def period_metrics(name, returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    row = m.summary(name, r)
    return {
        "model": name,
        "annual_return": row["annual_return"],
        "sharpe": row["sharpe"],
        "max_drawdown": row["max_drawdown"],
        "final_equity": row["final_equity"],
        "days": row["days"],
    }


def apply_013f_latest_overlay(live_row, cfg, realized_vol, prices):
    exec_weights, details = m.build_dynamic_exec_weights(live_row, cfg, realized_vol)
    for asset in m.EXEC_ASSETS + ["SOXX", "SOXL"]:
        exec_weights.setdefault(asset, 0.0)
    exec_weights, soxx_on, soxl_on, soxl_frac = r13e.apply_overlay(exec_weights, live_row, PARAMS_013F)
    latest = {
        "model": "013F_SOXX_SOXL_PRODUCTION_CANDIDATE",
        "signal_date": live_row.get("date", live_row.get("aligned_date")),
        "aligned_date": live_row.get("aligned_date"),
        "soxx_overlay_on": soxx_on,
        "soxl_overlay_on": soxl_on,
        "soxl_ladder_fraction": soxl_frac,
        **details,
    }
    for col in [
        "top_asset", "second_asset", "top_score", "second_score", "score_gap",
        "growth_strength", "soxx_strength", "risk_off_strength", "crash_pressure",
        "breakdown_score", "soxx_63d_return", "qqqm_63d_return", "soxx_ma63_ratio",
    ]:
        if col in live_row.index:
            latest[col] = live_row[col]
    for asset in m.EXEC_ASSETS + ["SOXX", "SOXL"]:
        latest[f"exec_w_{asset}"] = exec_weights.get(asset, 0.0)
    return pd.DataFrame([latest])


def build_live_row(rebalance_raw, prices):
    latest_path = m.SCRIPT_DIR / f"{m.CURRENT_PREFIX}_latest_recommendation.csv"
    live_df = pd.read_csv(latest_path)
    live_input = live_df.iloc[-1].copy()
    if "signal_date" in live_input.index:
        live_date = pd.to_datetime(live_input["signal_date"])
    elif "latest_data_date" in live_input.index:
        live_date = pd.to_datetime(live_input["latest_data_date"])
    elif "date" in live_input.index:
        live_date = pd.to_datetime(live_input["date"])
    else:
        raise ValueError("Latest current-best recommendation has no date column")
    live_input["date"] = live_date
    combined = pd.concat([rebalance_raw.copy(), pd.DataFrame([live_input])], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = m.align_rebalance_dates(combined, prices[m.PRICE_ASSETS])
    combined = m.add_transition_features(combined)
    combined = r13c.add_soxx_market_filters(combined, prices)
    candidates = combined[combined["aligned_date"] <= live_date]
    if candidates.empty:
        raise ValueError(f"No live row available for {live_date.date()}")
    return candidates.iloc[-1]


def main():
    cfg, cfg_message = m.load_best_cfg()
    print(cfg_message)

    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    prices = r13c.download_extended_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)

    current_returns = m.load_current_returns_optional()
    base_returns, base_log, base_turn = m.run_backtest(rebalance, prices[m.PRICE_ASSETS], cfg, m.DEFAULT_TRANSACTION_COST)
    common = current_returns.index.intersection(base_returns.index)
    common_start, common_end = common.min(), common.max()
    early_start, early_end = pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")
    recent_start, recent_end = pd.Timestamp("2025-01-01"), common_end

    rows = []
    period_rows = []
    for tx_cost in [0.0, 0.0005, 0.001, 0.002, 0.003]:
        returns_013f, log_013f, avg_013f = r13e.run_backtest(rebalance, prices, cfg, PARAMS_013F)
        if tx_cost != m.DEFAULT_TRANSACTION_COST:
            returns_013f, log_013f, avg_013f = run_013f_with_cost(rebalance, prices, cfg, tx_cost)
        row = m.summary(f"013F_COST_{tx_cost:.4f}", returns_013f, avg_013f, len(log_013f), tx_cost)
        rows.append(row)

    returns_013f, log_013f, avg_013f = r13e.run_backtest(rebalance, prices, cfg, PARAMS_013F)
    returns_013g, log_013g, avg_013g = r13g.run_backtest(rebalance, prices, cfg, PARAMS_013G)

    for label, returns in [
        ("011_BASELINE", base_returns),
        ("013F_RECOMMENDED", returns_013f),
        ("013G_SMOOTHER", returns_013g),
    ]:
        period_rows.append(period_metrics(f"{label}_COMMON", returns, common_start, common_end))
        period_rows.append(period_metrics(f"{label}_EARLY_2023_2024", returns, early_start, early_end))
        period_rows.append(period_metrics(f"{label}_RECENT_2025_2026", returns, recent_start, recent_end))

    validation = pd.DataFrame(rows)
    periods = pd.DataFrame(period_rows)
    validation.to_csv(f"{OUT_PREFIX}_cost_sensitivity.csv", index=False)
    periods.to_csv(f"{OUT_PREFIX}_period_validation.csv", index=False)
    log_013f.to_csv(f"{OUT_PREFIX}_013F_rebalance_log.csv", index=False)
    returns_013f.to_csv(f"{OUT_PREFIX}_013F_daily_returns.csv", header=["portfolio_return"])

    live_row = build_live_row(rebalance_raw, prices)
    realized_vol = m.build_realized_vol(prices[m.PRICE_ASSETS])
    latest = apply_013f_latest_overlay(live_row, cfg, realized_vol, prices)
    latest.to_csv(f"{OUT_PREFIX}_013F_latest_recommendation.csv", index=False)

    exposure = pd.DataFrame([
        {
            "model": "013F_RECOMMENDED",
            "soxx_overlay_rebalances": int(log_013f["soxx_overlay_on"].sum()),
            "soxl_overlay_rebalances": int(log_013f["soxl_overlay_on"].sum()),
            "soxx_avg_weight": float(log_013f["exec_w_SOXX"].mean()),
            "soxl_avg_weight": float(log_013f["exec_w_SOXL"].mean()),
            "soxl_max_weight": float(log_013f["exec_w_SOXL"].max()),
            "avg_turnover": avg_013f,
        },
        {
            "model": "013G_SMOOTHER",
            "soxx_overlay_rebalances": int(log_013g["soxx_overlay_on"].sum()),
            "soxl_overlay_rebalances": int(log_013g["soxl_overlay_on"].sum()),
            "soxx_avg_weight": float(log_013g["exec_w_SOXX"].mean()),
            "soxl_avg_weight": float(log_013g["exec_w_SOXL"].mean()),
            "soxl_max_weight": float(log_013g["exec_w_SOXL"].max()),
            "avg_turnover": avg_013g,
        },
    ])
    exposure.to_csv(f"{OUT_PREFIX}_exposure_summary.csv", index=False)

    print("\n=== COST SENSITIVITY ===")
    print(validation.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== PERIOD VALIDATION ===")
    print(periods.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== EXPOSURE ===")
    print(exposure.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== LATEST 013F RECOMMENDATION ===")
    latest_cols = [c for c in latest.columns if c.startswith("exec_w_") or c in [
        "model", "signal_date", "aligned_date", "soxx_overlay_on", "soxl_overlay_on",
        "soxl_ladder_fraction", "growth_strength", "soxx_strength", "risk_off_strength",
        "crash_pressure", "breakdown_score", "soxx_63d_return", "qqqm_63d_return", "soxx_ma63_ratio",
    ]]
    print(latest[latest_cols].to_string(index=False))


def run_013f_with_cost(rebalance, prices, cfg, tx_cost):
    returns, log, avg_turn = r13e.run_backtest(rebalance, prices, cfg, PARAMS_013F)
    adjustment = (tx_cost - m.DEFAULT_TRANSACTION_COST)
    if abs(adjustment) < 1e-12:
        return returns, log, avg_turn
    adjusted = returns.copy()
    for _, row in log.iterrows():
        date = pd.to_datetime(row["date"])
        later_dates = adjusted.index[adjusted.index > date]
        if len(later_dates) > 0:
            adjusted.loc[later_dates[0]] -= float(row["turnover"]) * adjustment
    return adjusted, log, avg_turn


if __name__ == "__main__":
    main()
