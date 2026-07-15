import itertools

import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_019a_commodity_cyclical_confirmation as r19a


OUT_PREFIX = "model_c_plus_020A_fez_europe_leadership"
EUROPE_ASSETS = ["FEZ", "VGK", "EWG", "EWQ", "FXE", "UUP"]
EXEC_ASSETS_EXT = sorted(set(r19a.EXEC_ASSETS_EXT + ["FEZ"]))


def download_prices(start_date):
    assets = sorted(set(m.PRICE_ASSETS + r19a.r18a.EXTRA_ASSETS + r19a.COMMODITY_ASSETS + r14b.PEERS + EUROPE_ASSETS))
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


def build_daily_europe_features(prices):
    d = pd.DataFrame(index=prices.index)
    europe_market_assets = ["FEZ", "VGK", "EWG", "EWQ"]
    us_growth_assets = ["QQQM", "SOXX"]
    us_cyclical_assets = ["IWM", "XLI", "XLB", "XLE"]

    for window in [5, 10, 21, 63]:
        for asset in europe_market_assets + us_growth_assets + us_cyclical_assets + ["FXE", "UUP", "BIL"]:
            d[f"{asset.lower()}_ret_{window}"] = prices[asset].pct_change(window)

        europe_cols = [f"{asset.lower()}_ret_{window}" for asset in europe_market_assets]
        growth_cols = [f"{asset.lower()}_ret_{window}" for asset in us_growth_assets]
        cyclical_cols = [f"{asset.lower()}_ret_{window}" for asset in us_cyclical_assets]
        d[f"europe_avg_ret_{window}"] = d[europe_cols].mean(axis=1)
        d[f"us_growth_avg_ret_{window}"] = d[growth_cols].mean(axis=1)
        d[f"us_cyclical_avg_ret_{window}"] = d[cyclical_cols].mean(axis=1)
        d[f"fez_vs_growth_edge_{window}"] = d[f"fez_ret_{window}"] - d[f"us_growth_avg_ret_{window}"]
        d[f"fez_vs_cyclical_edge_{window}"] = d[f"fez_ret_{window}"] - d[f"us_cyclical_avg_ret_{window}"]
        d[f"europe_breadth_count_{window}"] = sum((d[f"{asset.lower()}_ret_{window}"] > 0.0).astype(int) for asset in europe_market_assets)
        d[f"europe_win_count_{window}"] = sum(
            (d[f"{asset.lower()}_ret_{window}"] > d[f"us_growth_avg_ret_{window}"]).astype(int)
            for asset in europe_market_assets
        )
        d[f"eur_vs_usd_{window}"] = d[f"fxe_ret_{window}"] - d[f"uup_ret_{window}"]

    d["fez_ma63_ratio"] = prices["FEZ"] / prices["FEZ"].rolling(63).mean() - 1.0
    return d.fillna(0.0)


def europe_signal(day_features, macro_row, params):
    risk_off = m.safe_float(macro_row, "risk_off_strength", 0.0)
    crash = m.safe_float(macro_row, "crash_pressure", 0.0)
    credit = m.safe_float(macro_row, "credit_strength", m.safe_float(macro_row, "hyg_strength", 0.0))
    industrial = m.safe_float(macro_row, "industrial_strength", 0.0)
    materials = m.safe_float(macro_row, "materials_strength", 0.0)

    fez_5 = float(day_features.get("fez_ret_5", 0.0))
    fez_10 = float(day_features.get("fez_ret_10", 0.0))
    edge_growth_10 = float(day_features.get("fez_vs_growth_edge_10", 0.0))
    edge_growth_21 = float(day_features.get("fez_vs_growth_edge_21", 0.0))
    edge_cyc_10 = float(day_features.get("fez_vs_cyclical_edge_10", 0.0))
    breadth_10 = float(day_features.get("europe_breadth_count_10", 0.0))
    wins_10 = float(day_features.get("europe_win_count_10", 0.0))
    eur_usd_10 = float(day_features.get("eur_vs_usd_10", 0.0))
    ma63 = float(day_features.get("fez_ma63_ratio", 0.0))

    macro_support = 0.35 * credit + 0.30 * industrial + 0.20 * materials + 0.15 * max(0.0, eur_usd_10)
    score = (
        0.25 * edge_growth_10
        + 0.20 * edge_growth_21
        + 0.15 * edge_cyc_10
        + 0.15 * max(0.0, fez_10)
        + 0.15 * max(0.0, macro_support)
        + 0.10 * max(0.0, ma63)
    )
    on = (
        fez_5 >= params["fez_5_min"]
        and fez_10 >= params["fez_10_min"]
        and edge_growth_10 >= params["edge_growth_10_min"]
        and edge_growth_21 >= params["edge_growth_21_min"]
        and edge_cyc_10 >= params["edge_cyc_10_min"]
        and breadth_10 >= params["breadth_10_min"]
        and wins_10 >= params["wins_10_min"]
        and eur_usd_10 >= params["eur_usd_10_min"]
        and ma63 >= params["ma63_min"]
        and credit >= params["credit_min"]
        and risk_off <= params["risk_max"]
        and crash <= params["crash_max"]
        and score >= params["score_min"]
    )
    return on, score


def apply_europe_overlay(weights, day_features, macro_row, params):
    weights = dict(weights)
    on, score = europe_signal(day_features, macro_row, params)
    if not on:
        return weights, False, score

    sources = ["BIL", "QQQM", "TQQQ", "SOXX", "SOXL", "IWM", "XLI", "XLB", "XLE"]
    source_total = sum(weights.get(asset, 0.0) for asset in sources)
    if source_total <= 0:
        return weights, False, score
    move = min(source_total * params["source_take"], params["fez_cap"])
    if move <= 1e-12:
        return weights, False, score

    for asset in sources:
        weight = weights.get(asset, 0.0)
        if weight > 0:
            cut = move * weight / source_total
            weights[asset] = weight - cut
            weights["FEZ"] = weights.get("FEZ", 0.0) + cut
    return normalize(weights), True, score


def run_backtest(rebalance, prices, cfg, commodity_params, europe_params, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + r19a.r18a.EXTRA_ASSETS + r19a.COMMODITY_ASSETS + r14b.PEERS + EUROPE_ASSETS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    defensive_features = r19a.r15e.build_daily_defensive_features(prices)
    bond_features = r19a.r18c.r17d.build_daily_bond_features(prices)
    gold_features = r19a.r18a.build_daily_gold_features(prices)
    commodity_features = r19a.build_daily_commodity_features(prices)
    europe_features = build_daily_europe_features(prices)
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
        commodity_days = 0
        europe_days = 0
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
            day_weights, europe_on, europe_score = apply_europe_overlay(
                day_weights, europe_features.loc[hold_date], row, europe_params
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
            if commodity_on:
                commodity_days += 1
            if europe_on:
                europe_days += 1

        portfolio.loc[hold_dates] = period_returns
        log = {
            "date": date,
            "turnover": rebalance_turnover,
            "soxx_overlay_on": soxx_on,
            "soxl_overlay_on": soxl_on,
            "iwm_overlay_on": iwm_on,
            "daily_commodity_active_days": commodity_days,
            "daily_europe_active_days": europe_days,
            "iwm_leadership_score": iwm_score,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"base_w_{asset}"] = base_weights.get(asset, 0.0)
            log[f"last_w_{asset}"] = previous_daily_weights.get(asset, 0.0)
        logs.append(log)
        old_rebalance_weights = base_weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(sum(turnovers) / len(turnovers)) if turnovers else float("nan")


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

    baseline_returns, _, _ = r19a.run_backtest(rebalance, prices, cfg, commodity_params())
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    base_common = period_metrics(baseline_returns, common_start, common_end)
    early_start, early_end = pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")
    recent_start, recent_end = pd.Timestamp("2025-01-01"), common_end
    base_early = period_metrics(baseline_returns, early_start, early_end)
    base_recent = period_metrics(baseline_returns, recent_start, recent_end)

    candidate_specs = [
        ("loose_lab", 0.03, 0.20, 0.000, -0.005, -0.005, 2, 2, -0.010, -0.020),
        ("balanced_small", 0.03, 0.20, 0.000, 0.000, -0.005, 3, 2, -0.010, -0.020),
        ("balanced_mid", 0.05, 0.20, 0.000, 0.000, -0.005, 3, 2, -0.010, -0.020),
        ("balanced_large", 0.08, 0.20, 0.000, 0.000, -0.005, 3, 2, -0.010, -0.020),
        ("growth_edge_small", 0.03, 0.20, 0.005, 0.000, -0.005, 3, 2, -0.010, -0.020),
        ("growth_edge_mid", 0.05, 0.20, 0.005, 0.000, -0.005, 3, 2, -0.010, -0.020),
        ("currency_confirmed", 0.05, 0.20, 0.000, 0.000, -0.005, 3, 2, 0.000, -0.020),
        ("strict_breadth", 0.05, 0.20, 0.005, 0.000, 0.000, 3, 3, -0.010, 0.000),
        ("strict_currency", 0.05, 0.20, 0.005, 0.000, 0.000, 3, 3, 0.000, 0.000),
        ("faster_take", 0.05, 0.35, 0.000, 0.000, -0.005, 3, 2, -0.010, -0.020),
        ("faster_large", 0.08, 0.35, 0.000, 0.000, -0.005, 3, 2, -0.010, -0.020),
        ("strict_small", 0.03, 0.20, 0.005, 0.005, 0.000, 3, 3, 0.000, 0.000),
    ]

    rows = []
    logs_by_name = {}
    returns_by_name = {}
    for label, cap, take, edge10, edge21, cyc_edge, breadth, wins, eur_usd, ma63 in candidate_specs:
        p = {
            "fez_cap": cap,
            "source_take": take,
            "fez_5_min": -0.005,
            "fez_10_min": 0.000,
            "edge_growth_10_min": edge10,
            "edge_growth_21_min": edge21,
            "edge_cyc_10_min": cyc_edge,
            "breadth_10_min": breadth,
            "wins_10_min": wins,
            "eur_usd_10_min": eur_usd,
            "ma63_min": ma63,
            "credit_min": -0.25,
            "risk_max": 1.25,
            "crash_max": 1.75,
            "score_min": 0.002,
        }
        name = (
            f"020A_{label}_cap{cap:.2f}_take{take:.2f}_eg{edge10:.3f}-{edge21:.3f}_"
            f"ec{cyc_edge:.3f}_b{breadth}_w{wins}_fx{eur_usd:.3f}_ma{ma63:.3f}"
        )
        returns, log, avg_turn = run_backtest(rebalance, prices, cfg, commodity_params(), p)
        common_m = period_metrics(returns, common_start, common_end)
        early_m = period_metrics(returns, early_start, early_end)
        recent_m = period_metrics(returns, recent_start, recent_end)
        row = m.summary(name, returns, avg_turn, len(log), m.DEFAULT_TRANSACTION_COST)
        row.update({
            "common_ann_return": common_m["ann"],
            "common_sharpe": common_m["sharpe"],
            "common_max_drawdown": common_m["dd"],
            "delta_ann_vs_019a": common_m["ann"] - base_common["ann"],
            "delta_sharpe_vs_019a": common_m["sharpe"] - base_common["sharpe"],
            "delta_dd_vs_019a": common_m["dd"] - base_common["dd"],
            "early_sharpe": early_m["sharpe"],
            "recent_sharpe": recent_m["sharpe"],
            "delta_early_sharpe_vs_019a": early_m["sharpe"] - base_early["sharpe"],
            "delta_recent_sharpe_vs_019a": recent_m["sharpe"] - base_recent["sharpe"],
            "europe_rebalance_count": int((log["daily_europe_active_days"] > 0).sum()),
            "europe_active_days": int(log["daily_europe_active_days"].sum()),
            "fez_last_avg_weight": float(log["last_w_FEZ"].mean()),
            "selection_score": (
                (common_m["ann"] - base_common["ann"])
                + 0.8 * (common_m["sharpe"] - base_common["sharpe"])
                + 0.4 * (early_m["sharpe"] - base_early["sharpe"])
                + 0.4 * (recent_m["sharpe"] - base_recent["sharpe"])
                + 3.0 * min(0.0, common_m["dd"] - base_common["dd"])
            ),
            **p,
        })
        rows.append(row)
        logs_by_name[name] = log
        returns_by_name[name] = returns

    results = pd.DataFrame(rows).sort_values(["selection_score", "common_sharpe"], ascending=[False, False])
    best_name = results.iloc[0]["model"]
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])

    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "delta_ann_vs_019a", "delta_sharpe_vs_019a", "delta_dd_vs_019a",
        "early_sharpe", "delta_early_sharpe_vs_019a", "recent_sharpe", "delta_recent_sharpe_vs_019a",
        "europe_rebalance_count", "europe_active_days", "fez_last_avg_weight",
        "fez_cap", "source_take", "edge_growth_10_min", "edge_growth_21_min",
        "edge_cyc_10_min", "breadth_10_min", "wins_10_min", "eur_usd_10_min", "ma63_min",
    ]
    print("BASE_019A_COMMON", base_common)
    print("BASE_019A_EARLY", base_early)
    print("BASE_019A_RECENT", base_recent)
    print(results[cols].head(40).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
