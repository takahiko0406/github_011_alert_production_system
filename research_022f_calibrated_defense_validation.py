import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_020b_fez_validation as r20b
import research_020c_fez_stress_guard as r20c
import research_022b_defensive_real_estate_allocation as r22b
import research_022e_defensive_allocation_calibration as r22e


OUT_PREFIX = "model_c_plus_022F_calibrated_defense_validation"

PERIODS = {
    "common": None,
    "early_2023_2024": (pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")),
    "recent_2025_2026": (pd.Timestamp("2025-01-01"), None),
    "rate_stress_2024H1": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31")),
    "defense_2024_summer": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-31")),
    "april_2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30")),
    "europe_2025H1": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
}

BEST_DEFENSIVE_PARAMS = r22b.params(
    source_take=0.35,
    xlp_margin=0.010,
    xlp_cap=0.10,
    xlre_cap=0.08,
    use_bil=True,
)
BEST_DEFENSIVE_PARAMS["xlu_cap"] = 0.05
BEST_DEFENSIVE_PARAMS["broad_breadth_min"] = 0.875
BEST_DEFENSIVE_PARAMS["risk_losses_min"] = 3
BEST_EUROPE_PARAMS = r20c.guarded_europe_params(0.08, 0.25, False)
BEST_VOL_MAX = 9.99


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {
        "ann": m.annualized_return(r),
        "sharpe": m.sharpe_ratio(r),
        "dd": m.max_drawdown(r),
        "days": len(r),
    }


def exposure_summary(log):
    row = {
        "rebalance_rows": len(log),
        "defensive_days": int(log["daily_defensive_regime_days"].sum()),
        "xlv_days": int(log["daily_xlv_days"].sum()),
        "xlp_days": int(log["daily_xlp_days"].sum()),
        "xlu_days": int(log["daily_xlu_days"].sum()),
        "xlre_days": int(log["daily_xlre_days"].sum()),
        "blocked_vol_days": int(log["daily_defensive_blocked_vol_days"].sum()),
    }
    for asset in ["XLV", "XLP", "XLU", "XLRE", "FEZ", "TLT", "GLD", "SOXX", "SOXL", "TQQQ", "BIL"]:
        col = f"last_w_{asset}"
        if col in log.columns:
            row[f"avg_{asset.lower()}_weight"] = float(log[col].mean())
            row[f"max_{asset.lower()}_weight"] = float(log[col].max())
    return row


def write_report(period_df, cost_df, exposure_df):
    common = period_df[period_df["period"].eq("common")].iloc[0]
    cost_2x = cost_df[cost_df["cost_multiplier"].eq(2.0)].iloc[0]
    cost_5x = cost_df[cost_df["cost_multiplier"].eq(5.0)].iloc[0]
    rate = period_df[period_df["period"].eq("rate_stress_2024H1")].iloc[0]
    april = period_df[period_df["period"].eq("april_2025")].iloc[0]
    exposure = exposure_df.iloc[0]

    lines = []
    lines.append("022F CALIBRATED DEFENSIVE / REAL ESTATE VALIDATION")
    lines.append("")
    lines.append("Design Being Validated")
    lines.append("- 022E tightened defensive breadth from 0.750 to 0.875.")
    lines.append("- XLP still represents consumer-staples recession defense, but weaker XLP calls are filtered out.")
    lines.append("- XLRE still requires real-estate strength, rate relief, and credit confirmation.")
    lines.append("- XLU remains a rate-relief defensive sleeve.")
    lines.append("- The volatility block was tested but rejected; best model uses no practical vol block.")
    lines.append("")
    lines.append("Common Period Versus 020C")
    lines.append(f"- Annual return delta: {common['delta_ann_vs_020c']:+.4f}")
    lines.append(f"- Sharpe delta: {common['delta_sharpe_vs_020c']:+.4f}")
    lines.append(f"- Max drawdown delta: {common['delta_dd_vs_020c']:+.4f}")
    lines.append("")
    lines.append("Cost Sensitivity")
    lines.append(f"- 2x cost Sharpe delta: {cost_2x['delta_sharpe_vs_020c']:+.4f}")
    lines.append(f"- 5x cost Sharpe delta: {cost_5x['delta_sharpe_vs_020c']:+.4f}")
    lines.append("")
    lines.append("Weak Windows")
    lines.append(f"- Rate-stress 2024H1 Sharpe delta: {rate['delta_sharpe_vs_020c']:+.4f}")
    lines.append(f"- April 2025 Sharpe delta: {april['delta_sharpe_vs_020c']:+.4f}")
    lines.append("")
    lines.append("Exposure")
    lines.append(f"- Defensive active days: {int(exposure['defensive_days'])}")
    lines.append(f"- XLP days: {int(exposure['xlp_days'])}")
    lines.append(f"- XLRE days: {int(exposure['xlre_days'])}")
    lines.append(f"- XLU days: {int(exposure['xlu_days'])}")
    lines.append(f"- XLV days: {int(exposure['xlv_days'])}")
    lines.append("")
    lines.append("Recommendation")
    if common["delta_sharpe_vs_020c"] > 0 and common["delta_dd_vs_020c"] > 0 and cost_2x["delta_sharpe_vs_020c"] > 0:
        lines.append("- 022E passes normal validation as the best defensive/real-estate allocation candidate so far.")
        lines.append("- It is a balanced/risk-quality candidate, not a clean replacement for every market window.")
    else:
        lines.append("- 022E does not pass validation. Keep it as lab logic only.")
    with open(f"{OUT_PREFIX}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = r22b.download_prices_checked(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, _, _ = r20c.run_backtest(
        rebalance,
        prices,
        cfg,
        r20b.commodity_params(),
        r20c.guarded_europe_params(0.08, 0.25, True),
        tx_cost=m.DEFAULT_TRANSACTION_COST,
    )
    model_returns, model_log, _ = r22e.run_backtest(
        rebalance,
        prices,
        cfg,
        BEST_DEFENSIVE_PARAMS,
        BEST_EUROPE_PARAMS,
        BEST_VOL_MAX,
        tx_cost=m.DEFAULT_TRANSACTION_COST,
    )
    common = current_returns.index.intersection(baseline_returns.index).intersection(model_returns.index)
    common_start, common_end = common.min(), common.max()
    baseline_returns = baseline_returns.loc[common]
    model_returns = model_returns.loc[common]

    period_rows = []
    for period, bounds in PERIODS.items():
        if bounds is None:
            start_p, end_p = common_start, common_end
        else:
            start_p = bounds[0]
            end_p = bounds[1] if bounds[1] is not None else common_end
        base_m = period_metrics(baseline_returns, start_p, end_p)
        model_m = period_metrics(model_returns, start_p, end_p)
        period_rows.append({
            "period": period,
            "ann_return": model_m["ann"],
            "sharpe": model_m["sharpe"],
            "max_drawdown": model_m["dd"],
            "days": model_m["days"],
            "base_ann_return": base_m["ann"],
            "base_sharpe": base_m["sharpe"],
            "base_max_drawdown": base_m["dd"],
            "delta_ann_vs_020c": model_m["ann"] - base_m["ann"],
            "delta_sharpe_vs_020c": model_m["sharpe"] - base_m["sharpe"],
            "delta_dd_vs_020c": model_m["dd"] - base_m["dd"],
        })

    cost_rows = []
    for mult in [0.0, 1.0, 2.0, 5.0]:
        tx_cost = m.DEFAULT_TRANSACTION_COST * mult
        base_r, _, _ = r20c.run_backtest(
            rebalance,
            prices,
            cfg,
            r20b.commodity_params(),
            r20c.guarded_europe_params(0.08, 0.25, True),
            tx_cost=tx_cost,
        )
        model_r, _, _ = r22e.run_backtest(
            rebalance,
            prices,
            cfg,
            BEST_DEFENSIVE_PARAMS,
            BEST_EUROPE_PARAMS,
            BEST_VOL_MAX,
            tx_cost=tx_cost,
        )
        idx = common.intersection(base_r.index).intersection(model_r.index)
        base_m = period_metrics(base_r.loc[idx], idx.min(), idx.max())
        model_m = period_metrics(model_r.loc[idx], idx.min(), idx.max())
        cost_rows.append({
            "cost_multiplier": mult,
            "tx_cost": tx_cost,
            "ann_return": model_m["ann"],
            "sharpe": model_m["sharpe"],
            "max_drawdown": model_m["dd"],
            "base_ann_return": base_m["ann"],
            "base_sharpe": base_m["sharpe"],
            "base_max_drawdown": base_m["dd"],
            "delta_ann_vs_020c": model_m["ann"] - base_m["ann"],
            "delta_sharpe_vs_020c": model_m["sharpe"] - base_m["sharpe"],
            "delta_dd_vs_020c": model_m["dd"] - base_m["dd"],
        })

    period_df = pd.DataFrame(period_rows)
    cost_df = pd.DataFrame(cost_rows)
    exposure_df = pd.DataFrame([exposure_summary(model_log)])
    period_df.to_csv(f"{OUT_PREFIX}_period_validation.csv", index=False)
    cost_df.to_csv(f"{OUT_PREFIX}_cost_sensitivity.csv", index=False)
    exposure_df.to_csv(f"{OUT_PREFIX}_exposure_summary.csv", index=False)
    model_returns.to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])
    model_log.to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    write_report(period_df, cost_df, exposure_df)

    print("PERIOD VALIDATION")
    print(period_df.to_string(index=False))
    print("\nCOST SENSITIVITY")
    print(cost_df.to_string(index=False))
    print("\nEXPOSURE SUMMARY")
    print(exposure_df.to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_*.csv and {OUT_PREFIX}.txt")


if __name__ == "__main__":
    main()
