import itertools

import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_019a_commodity_cyclical_confirmation as r19a
import research_020b_fez_validation as r20b
import research_020c_fez_stress_guard as r20c
import research_022b_defensive_real_estate_allocation as r22b


OUT_PREFIX = "model_c_plus_022E_defensive_allocation_calibration"
PERIODS = r22b.PERIODS


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {"ann": m.annualized_return(r), "sharpe": m.sharpe_ratio(r), "dd": m.max_drawdown(r)}


def run_backtest(rebalance, prices, cfg, defensive_params, europe_params, vol_max, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + r19a.r18a.EXTRA_ASSETS + r19a.COMMODITY_ASSETS + r14b.PEERS + r22b.r20a.EUROPE_ASSETS + r22b.DEFENSIVE_ASSETS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    defensive_features = r22b.build_defensive_regime_features(prices)
    bond_features = r19a.r18c.r17d.build_daily_bond_features(prices)
    gold_features = r19a.r18a.build_daily_gold_features(prices)
    commodity_features = r19a.build_daily_commodity_features(prices)
    europe_features = r22b.r20a.build_daily_europe_features(prices)
    realized_vol = m.build_realized_vol(prices[m.PRICE_ASSETS])
    dates = prices.index
    rebal_dates = [date for date in rebalance["aligned_date"] if date in dates]
    date_to_loc = {date: i for i, date in enumerate(dates)}
    old_rebalance_weights = {asset: 0.0 for asset in r22b.EXEC_ASSETS_EXT}
    old_rebalance_weights["BIL"] = 1.0
    portfolio = pd.Series(index=dates, dtype=float)
    logs = []
    turnovers = []
    tlt_params = r19a.r18c.tlt_params(0.15)
    gld_params = r19a.r18c.gld_params(0.10)
    commodity_params = r20b.commodity_params()

    for i, date in enumerate(rebal_dates):
        row = rebalance.loc[rebalance["aligned_date"].eq(date)].iloc[0]
        base_weights, details = m.build_dynamic_exec_weights(row, cfg, realized_vol)
        for asset in r22b.EXEC_ASSETS_EXT:
            base_weights.setdefault(asset, 0.0)
        base_weights, soxx_on, soxl_on, soxl_frac = r19a.r18c.r13e.apply_overlay(base_weights, row, r19a.r18c.r13h.PARAMS_013F)
        base_weights, iwm_on, tna_on, iwm_score = r19a.r18c.r14b.apply_iwm_tna(base_weights, row, r19a.r18c.r15a.PARAMS_014D)

        rebalance_turnover = sum(abs(base_weights.get(asset, 0.0) - old_rebalance_weights.get(asset, 0.0)) for asset in r22b.EXEC_ASSETS_EXT)
        turnovers.append(rebalance_turnover)
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        hold_dates = dates[date_to_loc[date] + 1: date_to_loc.get(next_date, len(dates) - 1) + 1]
        if len(hold_dates) == 0:
            old_rebalance_weights = base_weights.copy()
            continue

        previous_daily_weights = base_weights.copy()
        defensive_days = 0
        xlv_days = 0
        xlp_days = 0
        xlu_days = 0
        xlre_days = 0
        blocked_vol_days = 0
        regimes = {}
        period_returns = []
        for j, hold_date in enumerate(hold_dates):
            day_weights = base_weights.copy()
            defense_on = False
            defense_regime = "no_defensive_allocation"
            defense_destination = ""
            if pd.notna(details.get("realized_vol_now")) and float(details.get("realized_vol_now")) > vol_max:
                blocked_vol_days += 1
            else:
                day_weights, defense_on, defense_regime, defense_destination = r22b.apply_regime_defensive_overlay(
                    day_weights, defensive_features.loc[hold_date], defensive_params
                )
            day_weights, tlt_on, tlt_score = r19a.r18c.r17d.apply_tlt_overlay(day_weights, bond_features.loc[hold_date], row, tlt_params)
            day_weights, gld_on, gld_score = r19a.r18a.apply_gld_overlay(day_weights, gold_features.loc[hold_date], row, gld_params)
            day_weights, commodity_on, commodity_score = r19a.apply_commodity_overlay(day_weights, commodity_features.loc[hold_date], row, commodity_params)
            day_weights, europe_on, europe_score, block_reason = r20c.apply_guarded_europe_overlay(
                day_weights, europe_features.loc[hold_date], row, europe_params, defense_on
            )

            for asset in r22b.EXEC_ASSETS_EXT:
                day_weights.setdefault(asset, 0.0)
            daily_turnover = sum(abs(day_weights.get(asset, 0.0) - previous_daily_weights.get(asset, 0.0)) for asset in r22b.EXEC_ASSETS_EXT)
            if j == 0:
                daily_turnover += rebalance_turnover
            ret = sum(weight * asset_returns.at[hold_date, asset] for asset, weight in day_weights.items() if abs(weight) > 1e-12)
            ret -= daily_turnover * tx_cost
            period_returns.append(ret)
            previous_daily_weights = day_weights

            if defense_on:
                defensive_days += 1
                regimes[defense_regime] = regimes.get(defense_regime, 0) + 1
                xlv_days += int(defense_destination == "XLV")
                xlp_days += int(defense_destination == "XLP")
                xlu_days += int(defense_destination == "XLU")
                xlre_days += int(defense_destination == "XLRE")

        portfolio.loc[hold_dates] = period_returns
        log = {
            "date": date,
            "turnover": rebalance_turnover,
            "daily_defensive_regime_days": defensive_days,
            "daily_xlv_days": xlv_days,
            "daily_xlp_days": xlp_days,
            "daily_xlu_days": xlu_days,
            "daily_xlre_days": xlre_days,
            "daily_defensive_blocked_vol_days": blocked_vol_days,
            "daily_defensive_regime_counts": str(regimes),
            **details,
        }
        for asset in r22b.EXEC_ASSETS_EXT:
            log[f"last_w_{asset}"] = previous_daily_weights.get(asset, 0.0)
        logs.append(log)
        old_rebalance_weights = base_weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(sum(turnovers) / len(turnovers)) if turnovers else float("nan")


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
    )
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    base_common = period_metrics(baseline_returns, common_start, common_end)

    rows = []
    period_rows = []
    returns_by_name = {}
    logs_by_name = {}
    specs = itertools.product(
        [0.07, 0.10],
        [0.05, 0.08],
        [0.0, 0.03, 0.05],
        [0.75, 0.875],
        [3, 4],
        [0.20, 0.25, 0.35, 9.99],
    )
    for xlp_cap, xlre_cap, xlu_cap, breadth, risk_losses, vol_max in specs:
        defensive_params = r22b.params(0.35, 0.010, xlp_cap, xlre_cap, True)
        defensive_params["xlu_cap"] = xlu_cap
        defensive_params["broad_breadth_min"] = breadth
        defensive_params["risk_losses_min"] = risk_losses
        europe_params = r20c.guarded_europe_params(0.08, 0.25, False)
        name = f"022E_xlp{xlp_cap:.2f}_xlre{xlre_cap:.2f}_xlu{xlu_cap:.2f}_b{breadth:.3f}_rl{risk_losses}_v{vol_max:.2f}"
        returns, log, avg_turn = run_backtest(rebalance, prices, cfg, defensive_params, europe_params, vol_max)
        common_m = period_metrics(returns, common_start, common_end)
        rate_m = period_metrics(returns, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31"))
        april_m = period_metrics(returns, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30"))
        rate_b = period_metrics(baseline_returns, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31"))
        april_b = period_metrics(baseline_returns, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30"))
        rows.append({
            "model": name,
            "common_ann_return": common_m["ann"],
            "common_sharpe": common_m["sharpe"],
            "common_max_drawdown": common_m["dd"],
            "delta_ann_vs_020c": common_m["ann"] - base_common["ann"],
            "delta_sharpe_vs_020c": common_m["sharpe"] - base_common["sharpe"],
            "delta_dd_vs_020c": common_m["dd"] - base_common["dd"],
            "rate_stress_delta_sharpe": rate_m["sharpe"] - rate_b["sharpe"],
            "april_2025_delta_sharpe": april_m["sharpe"] - april_b["sharpe"],
            "defensive_days": int(log["daily_defensive_regime_days"].sum()),
            "xlp_days": int(log["daily_xlp_days"].sum()),
            "xlu_days": int(log["daily_xlu_days"].sum()),
            "xlre_days": int(log["daily_xlre_days"].sum()),
            "blocked_vol_days": int(log["daily_defensive_blocked_vol_days"].sum()),
            "selection_score": (
                (common_m["sharpe"] - base_common["sharpe"])
                + 0.5 * (common_m["ann"] - base_common["ann"])
                + 2.0 * (common_m["dd"] - base_common["dd"])
                + 0.20 * min(0.0, rate_m["sharpe"] - rate_b["sharpe"])
                + 0.20 * min(0.0, april_m["sharpe"] - april_b["sharpe"])
            ),
            "xlp_cap": xlp_cap,
            "xlre_cap": xlre_cap,
            "xlu_cap": xlu_cap,
            "broad_breadth_min": breadth,
            "risk_losses_min": risk_losses,
            "vol_max": vol_max,
        })
        for period, bounds in PERIODS.items():
            if bounds is None:
                start_p, end_p = common_start, common_end
            else:
                start_p = bounds[0]
                end_p = bounds[1] if bounds[1] is not None else common_end
            model_m = period_metrics(returns, start_p, end_p)
            base_m = period_metrics(baseline_returns, start_p, end_p)
            period_rows.append({
                "model": name,
                "period": period,
                "ann_return": model_m["ann"],
                "sharpe": model_m["sharpe"],
                "max_drawdown": model_m["dd"],
                "delta_ann_vs_020c": model_m["ann"] - base_m["ann"],
                "delta_sharpe_vs_020c": model_m["sharpe"] - base_m["sharpe"],
                "delta_dd_vs_020c": model_m["dd"] - base_m["dd"],
            })
        returns_by_name[name] = returns
        logs_by_name[name] = log

    results = pd.DataFrame(rows).sort_values(["selection_score", "common_sharpe"], ascending=[False, False])
    period_df = pd.DataFrame(period_rows)
    best_name = results.iloc[0]["model"]
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    period_df.to_csv(f"{OUT_PREFIX}_period_validation.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)

    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "delta_ann_vs_020c", "delta_sharpe_vs_020c", "delta_dd_vs_020c",
        "rate_stress_delta_sharpe", "april_2025_delta_sharpe",
        "defensive_days", "xlp_days", "xlu_days", "xlre_days", "blocked_vol_days",
    ]
    print("BASE_020C_COMMON", base_common)
    print(results[cols].head(25).to_string(index=False))
    print("\nBEST PERIODS")
    print(period_df[period_df["model"].eq(best_name)].to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv, {OUT_PREFIX}_period_validation.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
