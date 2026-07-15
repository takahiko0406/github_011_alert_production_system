import itertools

import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e


OUT_PREFIX = "model_c_plus_013G_soxl_instability_brake"
EXEC_ASSETS_EXT = m.EXEC_ASSETS + ["SOXX", "SOXL"]


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {
        "ann": m.annualized_return(r),
        "sharpe": m.sharpe_ratio(r),
        "dd": m.max_drawdown(r),
    }


def ladder_fraction(row, params):
    growth = m.safe_float(row, "growth_strength", 0.0)
    soxx = m.safe_float(row, "soxx_strength", 0.0)
    risk = m.safe_float(row, "risk_off_strength", 0.0)
    leadership = soxx - growth
    rel_mom = m.safe_float(row, "soxx_63d_return", 0.0) - m.safe_float(row, "qqqm_63d_return", 0.0)
    ma63 = m.safe_float(row, "soxx_ma63_ratio", 0.0)

    if risk > 0.75 or ma63 < -0.05:
        return 0.0
    if leadership >= 0.50 and rel_mom >= 0.10:
        frac = params["large_frac"]
    elif leadership >= 0.40 and rel_mom >= 0.06:
        frac = params["medium_frac"]
    elif leadership >= 0.30 and rel_mom >= 0.03:
        frac = params["small_frac"]
    else:
        return 0.0

    transition_z = m.safe_float(row, "transition_instability_z", 0.0)
    leader_flip = m.safe_float(row, "leader_flip_4", 0.0)
    entropy = m.safe_float(row, "signal_entropy", 0.0)
    if transition_z >= params["transition_cut"]:
        frac *= params["transition_multiplier"]
    if leader_flip >= params["leader_flip_cut"]:
        frac *= params["flip_multiplier"]
    if entropy >= params["entropy_cut"]:
        frac *= params["entropy_multiplier"]
    return frac


def apply_overlay(weights, row, params):
    weights = dict(weights)
    growth = m.safe_float(row, "growth_strength", 0.0)
    soxx = m.safe_float(row, "soxx_strength", 0.0)
    risk = m.safe_float(row, "risk_off_strength", 0.0)
    crash = m.safe_float(row, "crash_pressure", 0.0)
    breakdown = m.safe_float(row, "breakdown_score", 0.0)
    leadership = soxx - growth

    soxx_on = (
        soxx >= 0.0
        and growth >= 0.0
        and risk <= 0.75
        and crash <= 1.25
        and breakdown <= 2.0
        and leadership >= 0.25
    )
    if not soxx_on:
        return weights, False, False, 0.0

    qqqm_move = weights.get("QQQM", 0.0) * params["qqqm_to_soxx"]
    if qqqm_move > 0:
        weights["QQQM"] -= qqqm_move
        weights["SOXX"] = weights.get("SOXX", 0.0) + qqqm_move

    soxl_frac = ladder_fraction(row, params)
    soxl_on = False
    if soxl_frac > 0:
        tqqq_move = weights.get("TQQQ", 0.0) * soxl_frac
        if tqqq_move > 0:
            move = min(tqqq_move, max(0.0, params["soxl_cap"] - weights.get("SOXL", 0.0)))
            weights["TQQQ"] -= move
            weights["SOXL"] = weights.get("SOXL", 0.0) + move
            soxl_on = move > 1e-12

    return r13c.normalize(weights), True, soxl_on, soxl_frac


def run_backtest(rebalance, prices, cfg, params):
    assets = sorted(set(m.PRICE_ASSETS + ["SOXX", "SOXL"]))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    realized_vol = m.build_realized_vol(prices[m.PRICE_ASSETS])
    dates = prices.index
    rebal_dates = [date for date in rebalance["aligned_date"] if date in dates]
    date_to_loc = {date: i for i, date in enumerate(dates)}
    old_weights = {asset: 0.0 for asset in EXEC_ASSETS_EXT}
    old_weights["BIL"] = 1.0
    portfolio = pd.Series(index=dates, dtype=float)
    logs = []
    turnovers = []

    for i, date in enumerate(rebal_dates):
        row = rebalance.loc[rebalance["aligned_date"].eq(date)].iloc[0]
        weights, details = m.build_dynamic_exec_weights(row, cfg, realized_vol)
        for asset in EXEC_ASSETS_EXT:
            weights.setdefault(asset, 0.0)
        weights, soxx_on, soxl_on, soxl_frac = apply_overlay(weights, row, params)
        turnover = sum(abs(weights.get(asset, 0.0) - old_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
        turnovers.append(turnover)

        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        hold_dates = dates[date_to_loc[date] + 1: date_to_loc.get(next_date, len(dates) - 1) + 1]
        if len(hold_dates) == 0:
            old_weights = weights.copy()
            continue

        returns = pd.Series(0.0, index=hold_dates)
        for asset, weight in weights.items():
            if abs(weight) > 1e-12:
                returns = returns.add(weight * asset_returns[asset].reindex(hold_dates).fillna(0.0), fill_value=0.0)
        returns.iloc[0] -= turnover * m.DEFAULT_TRANSACTION_COST
        portfolio.loc[hold_dates] = returns.values

        log = {
            "date": date,
            "turnover": turnover,
            "soxx_overlay_on": soxx_on,
            "soxl_overlay_on": soxl_on,
            "soxl_ladder_fraction": soxl_frac,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"exec_w_{asset}"] = weights.get(asset, 0.0)
        logs.append(log)
        old_weights = weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(sum(turnovers) / len(turnovers)) if turnovers else float("nan")


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

    base_returns, _, _ = m.run_backtest(rebalance, prices[m.PRICE_ASSETS], cfg, m.DEFAULT_TRANSACTION_COST)
    common = current_returns.index.intersection(base_returns.index)
    common_start, common_end = common.min(), common.max()
    early_start, early_end = pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")
    recent_start, recent_end = pd.Timestamp("2025-01-01"), common_end

    base_common = period_metrics(base_returns, common_start, common_end)
    base_early = period_metrics(base_returns, early_start, early_end)
    base_recent = period_metrics(base_returns, recent_start, recent_end)

    rows = []
    logs_by_name = {}
    returns_by_name = {}
    for qfrac, cap, transition_cut, transition_mult, flip_cut, flip_mult, entropy_cut, entropy_mult in itertools.product(
        [0.75, 0.90],
        [0.25, 0.30],
        [0.25, 0.50, 0.75],
        [0.50, 0.70],
        [0.25, 0.50],
        [0.50, 0.75],
        [0.50, 0.60],
        [0.60, 0.80],
    ):
        params = {
            "qqqm_to_soxx": qfrac,
            "soxl_cap": cap,
            "small_frac": 0.15,
            "medium_frac": 0.65,
            "large_frac": 0.80,
            "transition_cut": transition_cut,
            "transition_multiplier": transition_mult,
            "leader_flip_cut": flip_cut,
            "flip_multiplier": flip_mult,
            "entropy_cut": entropy_cut,
            "entropy_multiplier": entropy_mult,
        }
        name = (
            f"013G_q{qfrac:.2f}_cap{cap:.2f}_tc{transition_cut:.2f}x{transition_mult:.2f}_"
            f"fc{flip_cut:.2f}x{flip_mult:.2f}_ec{entropy_cut:.2f}x{entropy_mult:.2f}"
        )
        returns, log, avg_turn = run_backtest(rebalance, prices, cfg, params)
        common_metrics = period_metrics(returns, common_start, common_end)
        early_metrics = period_metrics(returns, early_start, early_end)
        recent_metrics = period_metrics(returns, recent_start, recent_end)

        row = m.summary(name, returns, avg_turn, len(log), m.DEFAULT_TRANSACTION_COST)
        row.update({
            "common_ann_return": common_metrics["ann"],
            "common_sharpe": common_metrics["sharpe"],
            "common_max_drawdown": common_metrics["dd"],
            "early_ann_return": early_metrics["ann"],
            "early_sharpe": early_metrics["sharpe"],
            "recent_ann_return": recent_metrics["ann"],
            "recent_sharpe": recent_metrics["sharpe"],
            "beats_common_return": common_metrics["ann"] > base_common["ann"],
            "beats_common_sharpe": common_metrics["sharpe"] > base_common["sharpe"],
            "no_worse_common_dd": common_metrics["dd"] >= base_common["dd"] - 1e-9,
            "beats_early_sharpe": early_metrics["sharpe"] > base_early["sharpe"],
            "beats_recent_sharpe": recent_metrics["sharpe"] > base_recent["sharpe"],
            "maturity_score": (
                (common_metrics["ann"] - base_common["ann"]) * 1.20
                + (common_metrics["sharpe"] - base_common["sharpe"]) * 0.55
                + (early_metrics["sharpe"] - base_early["sharpe"]) * 0.80
                + min(0.0, common_metrics["dd"] - base_common["dd"]) * 1.25
            ),
            "soxx_overlay_rebalances": int(log["soxx_overlay_on"].sum()),
            "soxl_overlay_rebalances": int(log["soxl_overlay_on"].sum()),
            "soxx_avg_weight": float(log["exec_w_SOXX"].mean()),
            "soxl_avg_weight": float(log["exec_w_SOXL"].mean()),
            "soxl_max_weight": float(log["exec_w_SOXL"].max()),
            **params,
        })
        rows.append(row)
        logs_by_name[name] = log
        returns_by_name[name] = returns

    results = pd.DataFrame(rows).sort_values(
        [
            "beats_common_return",
            "beats_common_sharpe",
            "no_worse_common_dd",
            "beats_early_sharpe",
            "maturity_score",
            "common_sharpe",
        ],
        ascending=[False, False, False, False, False, False],
    )
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    best_name = results.iloc[0]["model"]
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])

    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "early_ann_return", "early_sharpe", "recent_ann_return", "recent_sharpe",
        "maturity_score", "soxx_overlay_rebalances", "soxl_overlay_rebalances",
        "soxx_avg_weight", "soxl_avg_weight", "soxl_max_weight",
    ]
    print("BASE_COMMON", base_common)
    print("BASE_EARLY", base_early)
    print("BASE_RECENT", base_recent)
    print(results[cols].head(25).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
