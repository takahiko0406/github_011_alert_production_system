import time

import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_019a_commodity_cyclical_confirmation as r19a
import research_020a_fez_europe_leadership as r20a
import research_020b_fez_validation as r20b


OUT_PREFIX = "model_c_plus_020C_fez_stress_guard"
EXEC_ASSETS_EXT = r20a.EXEC_ASSETS_EXT


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


def download_prices_checked(start):
    required = sorted(set(m.PRICE_ASSETS + r19a.r18a.EXTRA_ASSETS + r19a.COMMODITY_ASSETS + r14b.PEERS + r20a.EUROPE_ASSETS))
    last_bad = []
    for attempt in range(3):
        prices = r20a.download_prices(start)
        last_bad = [
            asset for asset in required
            if asset not in prices.columns or prices[asset].dropna().empty
        ]
        if not last_bad:
            return prices
        time.sleep(5)
    raise ValueError(f"Download failed or all-NaN prices after retries: {last_bad}")


def guarded_europe_params(cap, hedge_max, block_defense):
    p = r20b.europe_params(cap)
    p["stress_hedge_max"] = hedge_max
    p["block_when_daily_defense_on"] = block_defense
    return p


def apply_guarded_europe_overlay(weights, day_features, macro_row, params, defense_on):
    weights = dict(weights)
    stress_hedge_weight = weights.get("BIL", 0.0) + weights.get("GLD", 0.0) + weights.get("TLT", 0.0)
    if stress_hedge_weight > params["stress_hedge_max"]:
        return weights, False, 0.0, "stress_hedge"
    if params["block_when_daily_defense_on"] and defense_on:
        return weights, False, 0.0, "daily_defense"
    weights, europe_on, score = r20a.apply_europe_overlay(weights, day_features, macro_row, params)
    return weights, europe_on, score, ""


def run_backtest(rebalance, prices, cfg, commodity_params, europe_params, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + r19a.r18a.EXTRA_ASSETS + r19a.COMMODITY_ASSETS + r14b.PEERS + r20a.EUROPE_ASSETS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    defensive_features = r19a.r15e.build_daily_defensive_features(prices)
    bond_features = r19a.r18c.r17d.build_daily_bond_features(prices)
    gold_features = r19a.r18a.build_daily_gold_features(prices)
    commodity_features = r19a.build_daily_commodity_features(prices)
    europe_features = r20a.build_daily_europe_features(prices)
    realized_vol = m.build_realized_vol(prices[m.PRICE_ASSETS])
    dates = prices.index
    rebal_dates = [date for date in rebalance["aligned_date"] if date in dates]
    date_to_loc = {date: i for i, date in enumerate(dates)}
    old_rebalance_weights = {asset: 0.0 for asset in EXEC_ASSETS_EXT}
    old_rebalance_weights["BIL"] = 1.0
    portfolio = pd.Series(index=dates, dtype=float)
    logs = []
    turnovers = []
    tlt_params = r19a.r18c.tlt_params(0.15)
    gld_params = r19a.r18c.gld_params(0.10)

    for i, date in enumerate(rebal_dates):
        row = rebalance.loc[rebalance["aligned_date"].eq(date)].iloc[0]
        base_weights, details = m.build_dynamic_exec_weights(row, cfg, realized_vol)
        for asset in EXEC_ASSETS_EXT:
            base_weights.setdefault(asset, 0.0)
        base_weights, soxx_on, soxl_on, soxl_frac = r19a.r18c.r13e.apply_overlay(base_weights, row, r19a.r18c.r13h.PARAMS_013F)
        base_weights, iwm_on, tna_on, iwm_score = r19a.r18c.r14b.apply_iwm_tna(base_weights, row, r19a.r18c.r15a.PARAMS_014D)

        rebalance_turnover = sum(abs(base_weights.get(asset, 0.0) - old_rebalance_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
        turnovers.append(rebalance_turnover)
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        hold_dates = dates[date_to_loc[date] + 1: date_to_loc.get(next_date, len(dates) - 1) + 1]
        if len(hold_dates) == 0:
            old_rebalance_weights = base_weights.copy()
            continue

        previous_daily_weights = base_weights.copy()
        europe_days = 0
        blocked_stress_days = 0
        blocked_defense_days = 0
        period_returns = []
        for j, hold_date in enumerate(hold_dates):
            day_weights = base_weights.copy()
            day_weights, defense_on, destination, defense_score, cyc = r19a.r15e.apply_daily_brake(
                day_weights, defensive_features.loc[hold_date], row, r19a.r16a.PARAMS_015H
            )
            day_weights, tlt_on, tlt_score = r19a.r18c.r17d.apply_tlt_overlay(day_weights, bond_features.loc[hold_date], row, tlt_params)
            day_weights, gld_on, gld_score = r19a.r18a.apply_gld_overlay(day_weights, gold_features.loc[hold_date], row, gld_params)
            day_weights, commodity_on, commodity_score = r19a.apply_commodity_overlay(
                day_weights, commodity_features.loc[hold_date], row, commodity_params
            )
            day_weights, europe_on, europe_score, block_reason = apply_guarded_europe_overlay(
                day_weights, europe_features.loc[hold_date], row, europe_params, defense_on
            )

            for asset in EXEC_ASSETS_EXT:
                day_weights.setdefault(asset, 0.0)
            daily_turnover = sum(abs(day_weights.get(asset, 0.0) - previous_daily_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
            if j == 0:
                daily_turnover += rebalance_turnover
            ret = sum(weight * asset_returns.at[hold_date, asset] for asset, weight in day_weights.items() if abs(weight) > 1e-12)
            ret -= daily_turnover * tx_cost
            period_returns.append(ret)
            previous_daily_weights = day_weights
            if europe_on:
                europe_days += 1
            elif block_reason == "stress_hedge":
                blocked_stress_days += 1
            elif block_reason == "daily_defense":
                blocked_defense_days += 1

        portfolio.loc[hold_dates] = period_returns
        log = {
            "date": date,
            "turnover": rebalance_turnover,
            "daily_europe_active_days": europe_days,
            "daily_europe_blocked_stress_days": blocked_stress_days,
            "daily_europe_blocked_defense_days": blocked_defense_days,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"last_w_{asset}"] = previous_daily_weights.get(asset, 0.0)
        logs.append(log)
        old_rebalance_weights = base_weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(sum(turnovers) / len(turnovers)) if turnovers else float("nan")


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = download_prices_checked(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, _, _ = r19a.run_backtest(rebalance, prices, cfg, r20b.commodity_params())
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    base_common = period_metrics(baseline_returns, common_start, common_end)

    rows = []
    period_rows = []
    specs = [
        ("cap05_hedge10_defense", 0.05, 0.10, True),
        ("cap05_hedge15_defense", 0.05, 0.15, True),
        ("cap05_hedge20_defense", 0.05, 0.20, True),
        ("cap05_hedge25_defense", 0.05, 0.25, True),
        ("cap08_hedge10_defense", 0.08, 0.10, True),
        ("cap08_hedge15_defense", 0.08, 0.15, True),
        ("cap08_hedge20_defense", 0.08, 0.20, True),
        ("cap08_hedge25_defense", 0.08, 0.25, True),
        ("cap05_hedge15_only", 0.05, 0.15, False),
        ("cap08_hedge15_only", 0.08, 0.15, False),
    ]
    returns_by_name = {}
    logs_by_name = {}
    for label, cap, hedge_max, block_defense in specs:
        name = f"020C_{label}"
        returns, log, avg_turn = run_backtest(
            rebalance, prices, cfg, r20b.commodity_params(), guarded_europe_params(cap, hedge_max, block_defense)
        )
        common_m = period_metrics(returns, common_start, common_end)
        rows.append({
            "model": name,
            "common_ann_return": common_m["ann"],
            "common_sharpe": common_m["sharpe"],
            "common_max_drawdown": common_m["dd"],
            "delta_ann_vs_019a": common_m["ann"] - base_common["ann"],
            "delta_sharpe_vs_019a": common_m["sharpe"] - base_common["sharpe"],
            "delta_dd_vs_019a": common_m["dd"] - base_common["dd"],
            "europe_active_days": int(log["daily_europe_active_days"].sum()),
            "blocked_stress_days": int(log["daily_europe_blocked_stress_days"].sum()),
            "blocked_defense_days": int(log["daily_europe_blocked_defense_days"].sum()),
            "fez_last_avg_weight": float(log["last_w_FEZ"].mean()),
            "avg_turnover": avg_turn,
            "cap": cap,
            "stress_hedge_max": hedge_max,
            "block_defense": block_defense,
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
                "delta_ann_vs_019a": model_m["ann"] - base_m["ann"],
                "delta_sharpe_vs_019a": model_m["sharpe"] - base_m["sharpe"],
                "delta_dd_vs_019a": model_m["dd"] - base_m["dd"],
            })
        returns_by_name[name] = returns
        logs_by_name[name] = log

    results = pd.DataFrame(rows).sort_values(["delta_sharpe_vs_019a", "delta_ann_vs_019a"], ascending=[False, False])
    period_df = pd.DataFrame(period_rows)
    best_name = results.iloc[0]["model"]
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    period_df.to_csv(f"{OUT_PREFIX}_period_validation.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)

    print("BASE_019A_COMMON", base_common)
    print(results.to_string(index=False))
    print("\nBEST PERIODS")
    print(period_df[period_df["model"].eq(best_name)].to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv, {OUT_PREFIX}_period_validation.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
