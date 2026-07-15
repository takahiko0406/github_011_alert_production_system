import itertools

import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e
import research_013h_final_soxx_soxl_validation as r13h
import research_014b_iwm_peer_leadership_tna as r14b
import research_015a_xlv_defensive_overlay as r15a
import research_015e_daily_macro_defense_brake as r15e
import research_016a_xlf_credit_cycle as r16a
import research_017d_tlt_cap_stability as r17d
import research_018a_gld_macro_hedge as r18a


OUT_PREFIX = "model_c_plus_018C_tlt_gld_combined"
EXEC_ASSETS_EXT = sorted(set(r17d.EXEC_ASSETS_EXT + r18a.EXEC_ASSETS_EXT + ["TLT", "GLD"]))


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {"ann": m.annualized_return(r), "sharpe": m.sharpe_ratio(r), "dd": m.max_drawdown(r)}


def run_backtest(rebalance, prices, cfg, tlt_params, gld_params, order, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + r17d.EXTRA_ASSETS + r18a.EXTRA_ASSETS + r14b.PEERS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    defensive_features = r15e.build_daily_defensive_features(prices)
    bond_features = r17d.build_daily_bond_features(prices)
    gold_features = r18a.build_daily_gold_features(prices)
    realized_vol = m.build_realized_vol(prices[m.PRICE_ASSETS])
    dates = prices.index
    rebal_dates = [date for date in rebalance["aligned_date"] if date in dates]
    date_to_loc = {date: i for i, date in enumerate(dates)}
    old_rebalance_weights = {asset: 0.0 for asset in EXEC_ASSETS_EXT}
    old_rebalance_weights["BIL"] = 1.0
    portfolio = pd.Series(index=dates, dtype=float)
    logs = []
    turnovers = []

    for i, date in enumerate(rebal_dates):
        row = rebalance.loc[rebalance["aligned_date"].eq(date)].iloc[0]
        base_weights, details = m.build_dynamic_exec_weights(row, cfg, realized_vol)
        for asset in EXEC_ASSETS_EXT:
            base_weights.setdefault(asset, 0.0)
        base_weights, soxx_on, soxl_on, soxl_frac = r13e.apply_overlay(base_weights, row, r13h.PARAMS_013F)
        base_weights, iwm_on, tna_on, iwm_score = r14b.apply_iwm_tna(base_weights, row, r15a.PARAMS_014D)

        rebalance_turnover = sum(abs(base_weights.get(asset, 0.0) - old_rebalance_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
        turnovers.append(rebalance_turnover)
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        hold_dates = dates[date_to_loc[date] + 1: date_to_loc.get(next_date, len(dates) - 1) + 1]
        if len(hold_dates) == 0:
            old_rebalance_weights = base_weights.copy()
            continue

        previous_daily_weights = base_weights.copy()
        defense_days = 0
        tlt_days = 0
        gld_days = 0
        period_returns = []
        for j, hold_date in enumerate(hold_dates):
            day_weights = base_weights.copy()
            day_weights, defense_on, destination, defense_score, cyc = r15e.apply_daily_brake(
                day_weights, defensive_features.loc[hold_date], row, r16a.PARAMS_015H
            )
            if order == "TLT_THEN_GLD":
                day_weights, tlt_on, tlt_score = r17d.apply_tlt_overlay(day_weights, bond_features.loc[hold_date], row, tlt_params)
                day_weights, gld_on, gld_score = r18a.apply_gld_overlay(day_weights, gold_features.loc[hold_date], row, gld_params)
            else:
                day_weights, gld_on, gld_score = r18a.apply_gld_overlay(day_weights, gold_features.loc[hold_date], row, gld_params)
                day_weights, tlt_on, tlt_score = r17d.apply_tlt_overlay(day_weights, bond_features.loc[hold_date], row, tlt_params)

            for asset in EXEC_ASSETS_EXT:
                day_weights.setdefault(asset, 0.0)
            daily_turnover = sum(abs(day_weights.get(asset, 0.0) - previous_daily_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
            if j == 0:
                daily_turnover += rebalance_turnover
            ret = sum(weight * asset_returns.at[hold_date, asset] for asset, weight in day_weights.items() if abs(weight) > 1e-12)
            ret -= daily_turnover * tx_cost
            period_returns.append(ret)
            previous_daily_weights = day_weights
            if defense_on:
                defense_days += 1
            if tlt_on:
                tlt_days += 1
            if gld_on:
                gld_days += 1

        portfolio.loc[hold_dates] = period_returns
        log = {
            "date": date,
            "turnover": rebalance_turnover,
            "soxx_overlay_on": soxx_on,
            "soxl_overlay_on": soxl_on,
            "iwm_overlay_on": iwm_on,
            "daily_defense_active_days": defense_days,
            "daily_tlt_active_days": tlt_days,
            "daily_gld_active_days": gld_days,
            "iwm_leadership_score": iwm_score,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"base_w_{asset}"] = base_weights.get(asset, 0.0)
            log[f"last_w_{asset}"] = previous_daily_weights.get(asset, 0.0)
        logs.append(log)
        old_rebalance_weights = base_weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(sum(turnovers) / len(turnovers)) if turnovers else float("nan")


def tlt_params(cap):
    return {
        "tlt_cap": cap,
        "source_take": 0.50 if cap >= 0.15 else 0.35,
        "tlt_5_min": 0.000,
        "tlt_10_min": 0.000,
        "tlt_vs_bil_5_min": 0.000,
        "tlt_vs_risk_5_min": 0.005,
        "rate_easing_10_min": 0.000,
        "risk_min": 0.00,
        "risk_max": 1.75,
        "crash_max": 2.0,
        "breakdown_min": 1.0,
        "qqqm_5_max": -0.012,
        "soxx_5_max": -0.025,
        "score_min": 0.003,
        "source_group": "all_risk",
    }


def gld_params(cap):
    return {
        "gld_cap": cap,
        "source_take": 0.25,
        "gld_5_min": 0.000,
        "gld_10_min": -0.005,
        "gld_vs_bil_5_min": 0.000,
        "gld_vs_risk_5_min": 0.005,
        "gld_vs_uup_10_min": -0.005,
        "gld_vs_tlt_5_min": -0.005,
        "require_gld_beats_tlt": False,
        "risk_min": 0.00,
        "risk_max": 1.75,
        "crash_max": 2.0,
        "breakdown_min": 1.0,
        "qqqm_5_max": -0.012,
        "soxx_5_max": -0.025,
        "score_min": 0.003,
        "source_group": "all_risk",
    }


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = r18a.download_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, _, _ = r15e.run_backtest(rebalance, prices, cfg, r16a.PARAMS_015H)
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    early_start, early_end = pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")
    recent_start, recent_end = pd.Timestamp("2025-01-01"), common_end
    base_common = period_metrics(baseline_returns, common_start, common_end)
    base_early = period_metrics(baseline_returns, early_start, early_end)
    base_recent = period_metrics(baseline_returns, recent_start, recent_end)

    rows = []
    logs_by_name = {}
    returns_by_name = {}
    for tlt_cap, gld_cap, order in itertools.product(
        [0.10, 0.12, 0.15],
        [0.05, 0.08, 0.10],
        ["TLT_THEN_GLD", "GLD_THEN_TLT"],
    ):
        name = f"018C_tlt{tlt_cap:.2f}_gld{gld_cap:.2f}_{order}"
        returns, log, avg_turn = run_backtest(rebalance, prices, cfg, tlt_params(tlt_cap), gld_params(gld_cap), order)
        common_m = period_metrics(returns, common_start, common_end)
        early_m = period_metrics(returns, early_start, early_end)
        recent_m = period_metrics(returns, recent_start, recent_end)
        row = m.summary(name, returns, avg_turn, len(log), m.DEFAULT_TRANSACTION_COST)
        row.update({
            "common_ann_return": common_m["ann"],
            "common_sharpe": common_m["sharpe"],
            "common_max_drawdown": common_m["dd"],
            "delta_common_ann_vs_015h": common_m["ann"] - base_common["ann"],
            "delta_common_sharpe_vs_015h": common_m["sharpe"] - base_common["sharpe"],
            "delta_common_dd_vs_015h": common_m["dd"] - base_common["dd"],
            "early_sharpe": early_m["sharpe"],
            "recent_sharpe": recent_m["sharpe"],
            "delta_early_sharpe_vs_015h": early_m["sharpe"] - base_early["sharpe"],
            "delta_recent_sharpe_vs_015h": recent_m["sharpe"] - base_recent["sharpe"],
            "selection_score": (
                (common_m["ann"] - base_common["ann"])
                + 0.8 * (common_m["sharpe"] - base_common["sharpe"])
                + 3.0 * min(0.0, common_m["dd"] - base_common["dd"])
            ),
            "tlt_cap": tlt_cap,
            "gld_cap": gld_cap,
            "order": order,
            "tlt_active_days": int(log["daily_tlt_active_days"].sum()),
            "gld_active_days": int(log["daily_gld_active_days"].sum()),
            "tlt_last_avg_weight": float(log["last_w_TLT"].mean()),
            "gld_last_avg_weight": float(log["last_w_GLD"].mean()),
        })
        rows.append(row)
        logs_by_name[name] = log
        returns_by_name[name] = returns

    results = pd.DataFrame(rows).sort_values(["selection_score", "common_sharpe"], ascending=[False, False])
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    best_name = results.iloc[0]["model"]
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])
    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "delta_common_ann_vs_015h", "delta_common_sharpe_vs_015h", "delta_common_dd_vs_015h",
        "early_sharpe", "recent_sharpe", "tlt_cap", "gld_cap", "order",
        "tlt_active_days", "gld_active_days", "tlt_last_avg_weight", "gld_last_avg_weight",
    ]
    print("BASE_015H_COMMON", base_common)
    print(results[cols].to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
