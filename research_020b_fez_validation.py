import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_019a_commodity_cyclical_confirmation as r19a
import research_020a_fez_europe_leadership as r20a


OUT_PREFIX = "model_c_plus_020B_fez_validation"

PERIODS = {
    "common": None,
    "early_2023_2024": (pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")),
    "recent_2025_2026": (pd.Timestamp("2025-01-01"), None),
    "rate_stress_2024H1": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31")),
    "defense_2024_summer": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-31")),
    "april_2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30")),
    "europe_2025H1": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
}


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {"ann": m.annualized_return(r), "sharpe": m.sharpe_ratio(r), "dd": m.max_drawdown(r)}


def commodity_params():
    return {
        "cyclical_cap": 0.08,
        "source_take": 0.35,
        "commodity_edge_5_min": 0.005,
        "commodity_edge_10_min": 0.000,
        "cyclical_edge_5_min": 0.000,
        "commodity_wins_5_min": 2,
        "cyclical_5_min": 0.005,
        "credit_min": -0.50,
        "risk_max": 1.25,
        "crash_max": 1.75,
        "score_min": 0.003,
    }


def europe_params(cap=0.08):
    return {
        "fez_cap": cap,
        "source_take": 0.35,
        "fez_5_min": -0.005,
        "fez_10_min": 0.000,
        "edge_growth_10_min": 0.000,
        "edge_growth_21_min": 0.000,
        "edge_cyc_10_min": -0.005,
        "breadth_10_min": 3,
        "wins_10_min": 2,
        "eur_usd_10_min": -0.010,
        "ma63_min": -0.020,
        "credit_min": -0.25,
        "risk_max": 1.25,
        "crash_max": 1.75,
        "score_min": 0.002,
    }


def summarize_periods(name, returns, baseline, common_start, common_end):
    rows = []
    for period, bounds in PERIODS.items():
        if bounds is None:
            start, end = common_start, common_end
        else:
            start = bounds[0]
            end = bounds[1] if bounds[1] is not None else common_end
        model_m = period_metrics(returns, start, end)
        base_m = period_metrics(baseline, start, end)
        rows.append({
            "model": name,
            "period": period,
            "ann_return": model_m["ann"],
            "sharpe": model_m["sharpe"],
            "max_drawdown": model_m["dd"],
            "delta_ann_vs_019a": model_m["ann"] - base_m["ann"],
            "delta_sharpe_vs_019a": model_m["sharpe"] - base_m["sharpe"],
            "delta_dd_vs_019a": model_m["dd"] - base_m["dd"],
        })
    return rows


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = r20a.download_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, _, _ = r19a.run_backtest(rebalance, prices, cfg, commodity_params())
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()

    baseline_by_cost = {}
    for cost_mult in [0.0, 1.0, 2.0, 5.0]:
        tx_cost = m.DEFAULT_TRANSACTION_COST * cost_mult
        baseline_by_cost[cost_mult], _, _ = r19a.run_backtest(rebalance, prices, cfg, commodity_params(), tx_cost=tx_cost)

    cost_rows = []
    period_rows = []
    for cap in [0.03, 0.05, 0.08]:
        name = f"020B_FEZ_cap{cap:.2f}"
        for cost_mult in [0.0, 1.0, 2.0, 5.0]:
            tx_cost = m.DEFAULT_TRANSACTION_COST * cost_mult
            returns, log, avg_turn = r20a.run_backtest(
                rebalance, prices, cfg, commodity_params(), europe_params(cap), tx_cost=tx_cost
            )
            common_m = period_metrics(returns, common_start, common_end)
            base_m = period_metrics(baseline_by_cost[cost_mult], common_start, common_end)
            cost_rows.append({
                "model": name,
                "cost_multiplier": cost_mult,
                "tx_cost": tx_cost,
                "ann_return": common_m["ann"],
                "sharpe": common_m["sharpe"],
                "max_drawdown": common_m["dd"],
                "delta_ann_vs_019a": common_m["ann"] - base_m["ann"],
                "delta_sharpe_vs_019a": common_m["sharpe"] - base_m["sharpe"],
                "delta_dd_vs_019a": common_m["dd"] - base_m["dd"],
                "europe_active_days": int(log["daily_europe_active_days"].sum()),
                "europe_rebalance_count": int((log["daily_europe_active_days"] > 0).sum()),
                "fez_last_avg_weight": float(log["last_w_FEZ"].mean()),
                "avg_turnover": avg_turn,
            })
            if cost_mult == 1.0:
                period_rows.extend(summarize_periods(name, returns, baseline_returns, common_start, common_end))

    cost_df = pd.DataFrame(cost_rows)
    period_df = pd.DataFrame(period_rows)
    cost_df.to_csv(f"{OUT_PREFIX}_cost_sensitivity.csv", index=False)
    period_df.to_csv(f"{OUT_PREFIX}_period_validation.csv", index=False)

    print("COST SENSITIVITY")
    print(cost_df.sort_values(["cost_multiplier", "delta_sharpe_vs_019a"], ascending=[True, False]).to_string(index=False))
    print("\nPERIOD VALIDATION")
    print(period_df.to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_cost_sensitivity.csv, {OUT_PREFIX}_period_validation.csv")


if __name__ == "__main__":
    main()
