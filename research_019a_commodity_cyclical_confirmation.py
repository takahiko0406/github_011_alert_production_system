import itertools

import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_015e_daily_macro_defense_brake as r15e
import research_016a_xlf_credit_cycle as r16a
import research_018a_gld_macro_hedge as r18a
import research_018c_tlt_gld_combined as r18c


OUT_PREFIX = "model_c_plus_019A_commodity_cyclical_confirmation"
COMMODITY_ASSETS = ["CPER", "DBC", "USO"]
EXEC_ASSETS_EXT = sorted(set(r18c.EXEC_ASSETS_EXT + ["GLD", "TLT"]))


def download_prices(start_date):
    assets = sorted(set(m.PRICE_ASSETS + r18a.EXTRA_ASSETS + COMMODITY_ASSETS + r14b.PEERS))
    prices = yf.download(assets, start=start_date, auto_adjust=True, progress=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"]
    prices = prices.sort_index().dropna(how="all").ffill()
    missing = [asset for asset in assets if asset not in prices.columns]
    if missing:
        raise ValueError(f"Missing prices: {missing}")
    return prices


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {"ann": m.annualized_return(r), "sharpe": m.sharpe_ratio(r), "dd": m.max_drawdown(r)}


def normalize(weights):
    total = sum(weights.values())
    return {asset: weight / total for asset, weight in weights.items()} if total > 0 else weights


def build_daily_commodity_features(prices):
    d = pd.DataFrame(index=prices.index)
    for window in [5, 10, 21]:
        for asset in COMMODITY_ASSETS + ["XLE", "XLB", "XLI", "QQQM", "SOXX", "IWM", "BIL"]:
            d[f"{asset.lower()}_ret_{window}"] = prices[asset].pct_change(window)
        commodity_cols = [f"{asset.lower()}_ret_{window}" for asset in COMMODITY_ASSETS]
        cyclical_cols = [f"{asset.lower()}_ret_{window}" for asset in ["XLE", "XLB", "XLI"]]
        risk_cols = [f"{asset.lower()}_ret_{window}" for asset in ["QQQM", "SOXX", "IWM"]]
        d[f"commodity_avg_ret_{window}"] = d[commodity_cols].mean(axis=1)
        d[f"cyclical_avg_ret_{window}"] = d[cyclical_cols].mean(axis=1)
        d[f"risk_avg_ret_{window}"] = d[risk_cols].mean(axis=1)
        d[f"commodity_edge_{window}"] = d[f"commodity_avg_ret_{window}"] - d[f"risk_avg_ret_{window}"]
        d[f"cyclical_edge_{window}"] = d[f"cyclical_avg_ret_{window}"] - d[f"risk_avg_ret_{window}"]
        d[f"commodity_win_count_{window}"] = sum((d[f"{asset.lower()}_ret_{window}"] > 0.0).astype(int) for asset in COMMODITY_ASSETS)
        leader_cols = [f"{asset.lower()}_ret_{window}" for asset in ["XLE", "XLB"]]
        d[f"best_cyclical_asset_{window}"] = (
            d[leader_cols]
            .fillna(-999.0)
            .idxmax(axis=1)
            .str.replace("_ret_" + str(window), "", regex=False)
            .str.upper()
        )
    return d.fillna(0.0)


def commodity_signal(day_features, macro_row, params):
    risk_off = m.safe_float(macro_row, "risk_off_strength", 0.0)
    crash = m.safe_float(macro_row, "crash_pressure", 0.0)
    credit = m.safe_float(macro_row, "credit_strength", m.safe_float(macro_row, "hyg_strength", 0.0))
    copper = m.safe_float(macro_row, "copper_strength", 0.0)
    materials = m.safe_float(macro_row, "materials_strength", 0.0)
    industrial = m.safe_float(macro_row, "industrial_strength", 0.0)

    commodity_edge_5d = float(day_features.get("commodity_edge_5", 0.0))
    commodity_edge_10d = float(day_features.get("commodity_edge_10", 0.0))
    cyclical_edge_5d = float(day_features.get("cyclical_edge_5", 0.0))
    commodity_wins_5d = float(day_features.get("commodity_win_count_5", 0.0))
    xle_5d = float(day_features.get("xle_ret_5", 0.0))
    xlb_5d = float(day_features.get("xlb_ret_5", 0.0))

    macro_support = 0.35 * copper + 0.25 * materials + 0.20 * industrial + 0.20 * credit
    score = (
        0.25 * commodity_edge_5d
        + 0.20 * commodity_edge_10d
        + 0.20 * cyclical_edge_5d
        + 0.20 * max(0.0, macro_support)
        + 0.15 * max(0.0, max(xle_5d, xlb_5d))
    )
    on = (
        commodity_edge_5d >= params["commodity_edge_5_min"]
        and commodity_edge_10d >= params["commodity_edge_10_min"]
        and cyclical_edge_5d >= params["cyclical_edge_5_min"]
        and commodity_wins_5d >= params["commodity_wins_5_min"]
        and max(xle_5d, xlb_5d) >= params["cyclical_5_min"]
        and credit >= params["credit_min"]
        and risk_off <= params["risk_max"]
        and crash <= params["crash_max"]
        and score >= params["score_min"]
    )
    return on, score


def apply_commodity_overlay(weights, day_features, macro_row, params):
    weights = dict(weights)
    on, score = commodity_signal(day_features, macro_row, params)
    if not on:
        return weights, False, score

    target = day_features.get("best_cyclical_asset_5", "XLE")
    if target not in ["XLE", "XLB"]:
        target = "XLE"
    sources = ["BIL", "QQQM", "TQQQ", "SOXX", "SOXL", "GLD", "TLT"]
    source_total = sum(weights.get(asset, 0.0) for asset in sources)
    if source_total <= 0:
        return weights, False, score
    move = min(source_total * params["source_take"], params["cyclical_cap"])
    if move <= 1e-12:
        return weights, False, score

    for asset in sources:
        weight = weights.get(asset, 0.0)
        if weight > 0:
            cut = move * weight / source_total
            weights[asset] = weight - cut
            weights[target] = weights.get(target, 0.0) + cut
    return normalize(weights), True, score


def run_backtest(rebalance, prices, cfg, params, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + r18a.EXTRA_ASSETS + COMMODITY_ASSETS + r14b.PEERS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    defensive_features = r15e.build_daily_defensive_features(prices)
    bond_features = r18c.r17d.build_daily_bond_features(prices)
    gold_features = r18a.build_daily_gold_features(prices)
    commodity_features = build_daily_commodity_features(prices)
    realized_vol = m.build_realized_vol(prices[m.PRICE_ASSETS])
    dates = prices.index
    rebal_dates = [date for date in rebalance["aligned_date"] if date in dates]
    date_to_loc = {date: i for i, date in enumerate(dates)}
    old_rebalance_weights = {asset: 0.0 for asset in EXEC_ASSETS_EXT}
    old_rebalance_weights["BIL"] = 1.0
    portfolio = pd.Series(index=dates, dtype=float)
    logs = []
    turnovers = []
    tlt_params = r18c.tlt_params(0.15)
    gld_params = r18c.gld_params(0.10)

    for i, date in enumerate(rebal_dates):
        row = rebalance.loc[rebalance["aligned_date"].eq(date)].iloc[0]
        base_weights, details = m.build_dynamic_exec_weights(row, cfg, realized_vol)
        for asset in EXEC_ASSETS_EXT:
            base_weights.setdefault(asset, 0.0)
        base_weights, soxx_on, soxl_on, soxl_frac = r18c.r13e.apply_overlay(base_weights, row, r18c.r13h.PARAMS_013F)
        base_weights, iwm_on, tna_on, iwm_score = r18c.r14b.apply_iwm_tna(base_weights, row, r18c.r15a.PARAMS_014D)

        rebalance_turnover = sum(abs(base_weights.get(asset, 0.0) - old_rebalance_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
        turnovers.append(rebalance_turnover)
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        hold_dates = dates[date_to_loc[date] + 1: date_to_loc.get(next_date, len(dates) - 1) + 1]
        if len(hold_dates) == 0:
            old_rebalance_weights = base_weights.copy()
            continue

        previous_daily_weights = base_weights.copy()
        commodity_days = 0
        tlt_days = 0
        gld_days = 0
        defense_days = 0
        period_returns = []
        for j, hold_date in enumerate(hold_dates):
            day_weights = base_weights.copy()
            day_weights, defense_on, destination, defense_score, cyc = r15e.apply_daily_brake(
                day_weights, defensive_features.loc[hold_date], row, r16a.PARAMS_015H
            )
            day_weights, tlt_on, tlt_score = r18c.r17d.apply_tlt_overlay(day_weights, bond_features.loc[hold_date], row, tlt_params)
            day_weights, gld_on, gld_score = r18a.apply_gld_overlay(day_weights, gold_features.loc[hold_date], row, gld_params)
            day_weights, commodity_on, commodity_score = apply_commodity_overlay(day_weights, commodity_features.loc[hold_date], row, params)

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
            if commodity_on:
                commodity_days += 1

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
            "daily_commodity_active_days": commodity_days,
            "iwm_leadership_score": iwm_score,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"base_w_{asset}"] = base_weights.get(asset, 0.0)
            log[f"last_w_{asset}"] = previous_daily_weights.get(asset, 0.0)
        logs.append(log)
        old_rebalance_weights = base_weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(sum(turnovers) / len(turnovers)) if turnovers else float("nan")


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = download_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, _, _ = r18c.run_backtest(
        rebalance, prices, cfg, r18c.tlt_params(0.15), r18c.gld_params(0.10), "TLT_THEN_GLD"
    )
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    base_common = period_metrics(baseline_returns, common_start, common_end)

    rows = []
    logs_by_name = {}
    returns_by_name = {}
    for cap, take, edge5, edge10, wins, cyc5 in itertools.product(
        [0.03, 0.05, 0.08],
        [0.20, 0.35],
        [0.005, 0.010],
        [0.000, 0.005],
        [2, 3],
        [0.000, 0.005],
    ):
        params = {
            "cyclical_cap": cap,
            "source_take": take,
            "commodity_edge_5_min": edge5,
            "commodity_edge_10_min": edge10,
            "cyclical_edge_5_min": 0.000,
            "commodity_wins_5_min": wins,
            "cyclical_5_min": cyc5,
            "credit_min": -0.50,
            "risk_max": 1.25,
            "crash_max": 1.75,
            "score_min": 0.003,
        }
        name = f"019A_cap{cap:.2f}_take{take:.2f}_e{edge5:.3f}-{edge10:.3f}_w{wins}_c{cyc5:.3f}"
        returns, log, avg_turn = run_backtest(rebalance, prices, cfg, params)
        common_m = period_metrics(returns, common_start, common_end)
        row = m.summary(name, returns, avg_turn, len(log), m.DEFAULT_TRANSACTION_COST)
        row.update({
            "common_ann_return": common_m["ann"],
            "common_sharpe": common_m["sharpe"],
            "common_max_drawdown": common_m["dd"],
            "delta_common_ann_vs_018c": common_m["ann"] - base_common["ann"],
            "delta_common_sharpe_vs_018c": common_m["sharpe"] - base_common["sharpe"],
            "delta_common_dd_vs_018c": common_m["dd"] - base_common["dd"],
            "selection_score": (
                (common_m["ann"] - base_common["ann"])
                + 0.8 * (common_m["sharpe"] - base_common["sharpe"])
                + 3.0 * min(0.0, common_m["dd"] - base_common["dd"])
            ),
            "commodity_rebalance_count": int((log["daily_commodity_active_days"] > 0).sum()),
            "commodity_active_days": int(log["daily_commodity_active_days"].sum()),
            "xle_last_avg_weight": float(log["last_w_XLE"].mean()),
            "xlb_last_avg_weight": float(log["last_w_XLB"].mean()),
            **params,
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
        "delta_common_ann_vs_018c", "delta_common_sharpe_vs_018c", "delta_common_dd_vs_018c",
        "commodity_rebalance_count", "commodity_active_days", "xle_last_avg_weight", "xlb_last_avg_weight",
        "cyclical_cap", "source_take", "commodity_edge_5_min", "commodity_edge_10_min",
        "commodity_wins_5_min", "cyclical_5_min",
    ]
    print("BASE_018C_COMMON", base_common)
    print(results[cols].head(40).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
