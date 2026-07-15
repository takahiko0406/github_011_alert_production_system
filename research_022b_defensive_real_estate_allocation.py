import itertools
import time

import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_014b_iwm_peer_leadership_tna as r14b
import research_019a_commodity_cyclical_confirmation as r19a
import research_020a_fez_europe_leadership as r20a
import research_020b_fez_validation as r20b
import research_020c_fez_stress_guard as r20c


OUT_PREFIX = "model_c_plus_022B_defensive_real_estate_allocation"
DEFENSIVE_ASSETS = ["XLV", "XLP", "XLU", "XLRE"]
RISK_ASSETS = ["QQQM", "SOXX", "IWM", "XLI", "XLB", "XLE"]
EXEC_ASSETS_EXT = sorted(set(r20a.EXEC_ASSETS_EXT + DEFENSIVE_ASSETS))


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


def normalize(weights):
    total = sum(weights.values())
    return {asset: weight / total for asset, weight in weights.items()} if total > 0 else weights


def download_prices_checked(start):
    required = sorted(
        set(
            m.PRICE_ASSETS
            + r19a.r18a.EXTRA_ASSETS
            + r19a.COMMODITY_ASSETS
            + r14b.PEERS
            + r20a.EUROPE_ASSETS
            + DEFENSIVE_ASSETS
            + ["IEF", "SHY", "HYG", "KRE", "XLF"]
        )
    )
    last_bad = []
    for _ in range(3):
        prices = yf.download(required, start=start, auto_adjust=True, progress=False)
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices["Close"]
        prices = prices.sort_index().dropna(how="all").ffill()
        last_bad = [asset for asset in required if asset not in prices.columns or prices[asset].dropna().empty]
        if not last_bad:
            return prices
        time.sleep(5)
    raise ValueError(f"Download failed or all-NaN prices after retries: {last_bad}")


def build_defensive_regime_features(prices):
    d = pd.DataFrame(index=prices.index)
    all_assets = sorted(set(DEFENSIVE_ASSETS + RISK_ASSETS + ["BIL", "TLT", "IEF", "SHY", "HYG", "KRE", "XLF"]))
    for window in [3, 5, 10, 21]:
        for asset in all_assets:
            d[f"{asset.lower()}_ret_{window}"] = prices[asset].pct_change(window)

        risk_cols = [f"{asset.lower()}_ret_{window}" for asset in RISK_ASSETS]
        cyclical_cols = [f"{asset.lower()}_ret_{window}" for asset in ["XLI", "XLB", "XLE"]]
        d[f"risk_avg_ret_{window}"] = d[risk_cols].mean(axis=1)
        d[f"cyclical_avg_ret_{window}"] = d[cyclical_cols].mean(axis=1)
        d[f"risk_loss_count_{window}"] = sum((d[f"{asset.lower()}_ret_{window}"] < 0.0).astype(int) for asset in RISK_ASSETS)
        d[f"def_positive_count_{window}"] = sum((d[f"{asset.lower()}_ret_{window}"] > 0.0).astype(int) for asset in DEFENSIVE_ASSETS)
        d[f"def_win_count_{window}"] = 0
        for asset in DEFENSIVE_ASSETS:
            d[f"{asset.lower()}_edge_{window}"] = d[f"{asset.lower()}_ret_{window}"] - d[f"risk_avg_ret_{window}"]
            d[f"def_win_count_{window}"] += (d[f"{asset.lower()}_edge_{window}"] > 0.0).astype(int)

    for asset in DEFENSIVE_ASSETS:
        name = asset.lower()
        d[f"{name}_score"] = (
            0.30 * d[f"{name}_edge_5"]
            + 0.25 * d[f"{name}_edge_10"]
            + 0.15 * d[f"{name}_edge_21"]
            + 0.15 * d[f"{name}_ret_5"]
            + 0.10 * d[f"{name}_ret_10"]
            + 0.05 * d[f"{name}_ret_21"]
        )

    score_cols = [f"{asset.lower()}_score" for asset in DEFENSIVE_ASSETS]
    d[score_cols] = d[score_cols].fillna(0.0)
    d["top_defensive_etf"] = d[score_cols].idxmax(axis=1).str.replace("_score", "", regex=False).str.upper()
    d["defensive_breadth_score"] = (
        d["def_positive_count_5"] + d["def_win_count_5"] + d["def_positive_count_10"] + d["def_win_count_10"]
    ) / 16.0
    d["rate_relief_flag"] = (
        (d["tlt_ret_5"] > 0.0)
        & (d["tlt_ret_10"] > -0.005)
        & ((d["ief_ret_10"] - d["shy_ret_10"]) > 0.0)
    )
    d["credit_confirmation"] = (
        ((d["hyg_ret_5"] - d["shy_ret_5"]) >= -0.005)
        & ((d["hyg_ret_10"] - d["shy_ret_10"]) >= -0.010)
        & (d["kre_ret_5"] >= -0.025)
        & (d["xlf_ret_5"] >= -0.020)
    )
    return d.fillna(0.0)


def classify_defensive_regime(day_features, params):
    scores = {asset: float(day_features.get(f"{asset.lower()}_score", 0.0)) for asset in DEFENSIVE_ASSETS}
    top = max(scores, key=scores.get)
    breadth = float(day_features.get("defensive_breadth_score", 0.0))
    risk_weak = (
        float(day_features.get("risk_avg_ret_5", 0.0)) <= params["risk_weak_max"]
        or float(day_features.get("risk_loss_count_5", 0.0)) >= params["risk_losses_min"]
    )
    cyclical_weak = float(day_features.get("cyclical_avg_ret_5", 0.0)) <= params["cyclical_weak_max"]
    rate_relief = bool(day_features.get("rate_relief_flag", False))
    credit_ok = bool(day_features.get("credit_confirmation", False))

    real_estate = (
        scores["XLRE"] > params["xlre_score_min"]
        and top == "XLRE"
        and float(day_features.get("xlre_ret_5", 0.0)) > 0.0
        and float(day_features.get("xlre_ret_10", 0.0)) > 0.0
        and rate_relief
        and credit_ok
    )
    strong_recession = (
        scores["XLP"] > params["xlp_score_min"]
        and scores["XLP"] >= scores["XLV"] + params["xlp_vs_xlv_margin"]
        and float(day_features.get("xlp_ret_5", 0.0)) > 0.0
        and float(day_features.get("xlp_edge_5", 0.0)) > 0.0
        and risk_weak
        and cyclical_weak
    )
    broad_recession = (
        breadth >= params["broad_breadth_min"]
        and scores["XLP"] > 0.0
        and risk_weak
        and cyclical_weak
    )
    utilities = top == "XLU" and scores["XLU"] > params["xlu_score_min"] and rate_relief
    healthcare = top == "XLV" and scores["XLV"] > params["xlv_score_min"] and breadth >= params["quality_breadth_min"] and risk_weak

    if real_estate:
        return "real_estate_rate_relief", "XLRE", params["xlre_cap"]
    if strong_recession:
        return "strong_consumer_recession_defense", "XLP", params["strong_xlp_cap"]
    if broad_recession:
        return "broad_recession_defense", "XLP", params["broad_xlp_cap"]
    if utilities:
        return "utilities_yield_defense", "XLU", params["xlu_cap"]
    if healthcare:
        return "healthcare_quality_defense", "XLV", params["xlv_cap"]
    return "no_defensive_allocation", "", 0.0


def apply_regime_defensive_overlay(weights, day_features, params):
    weights = dict(weights)
    regime, destination, cap = classify_defensive_regime(day_features, params)
    if not destination or cap <= 0:
        return weights, False, regime, destination

    sources = ["SOXL", "TQQQ", "SOXX", "QQQM", "IWM", "XSOE"]
    if regime in ["strong_consumer_recession_defense", "broad_recession_defense"]:
        sources += ["UXI", "ERX", "XLI", "XLB", "XLE", "FEZ"]
    if regime == "real_estate_rate_relief" and params["xlre_can_use_bil"]:
        sources += ["BIL"]

    source_total = sum(weights.get(asset, 0.0) for asset in sources)
    if source_total <= 0:
        return weights, False, regime, destination

    move = min(source_total * params["source_take"], cap)
    if move <= 1e-12:
        return weights, False, regime, destination

    for asset in sources:
        weight = weights.get(asset, 0.0)
        if weight > 0:
            cut = move * weight / source_total
            weights[asset] = weight - cut
            weights[destination] = weights.get(destination, 0.0) + cut
    return normalize(weights), True, regime, destination


def run_backtest(rebalance, prices, cfg, defensive_params, europe_params, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + r19a.r18a.EXTRA_ASSETS + r19a.COMMODITY_ASSETS + r14b.PEERS + r20a.EUROPE_ASSETS + DEFENSIVE_ASSETS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    defensive_features = build_defensive_regime_features(prices)
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
    commodity_params = r20b.commodity_params()

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
        defensive_days = 0
        xlv_days = 0
        xlp_days = 0
        xlu_days = 0
        xlre_days = 0
        regimes = {}
        period_returns = []
        for j, hold_date in enumerate(hold_dates):
            day_weights = base_weights.copy()
            day_weights, defense_on, defense_regime, defense_destination = apply_regime_defensive_overlay(
                day_weights, defensive_features.loc[hold_date], defensive_params
            )
            day_weights, tlt_on, tlt_score = r19a.r18c.r17d.apply_tlt_overlay(day_weights, bond_features.loc[hold_date], row, tlt_params)
            day_weights, gld_on, gld_score = r19a.r18a.apply_gld_overlay(day_weights, gold_features.loc[hold_date], row, gld_params)
            day_weights, commodity_on, commodity_score = r19a.apply_commodity_overlay(day_weights, commodity_features.loc[hold_date], row, commodity_params)
            day_weights, europe_on, europe_score, block_reason = r20c.apply_guarded_europe_overlay(
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
            "soxx_overlay_on": soxx_on,
            "soxl_overlay_on": soxl_on,
            "iwm_overlay_on": iwm_on,
            "daily_defensive_regime_days": defensive_days,
            "daily_xlv_days": xlv_days,
            "daily_xlp_days": xlp_days,
            "daily_xlu_days": xlu_days,
            "daily_xlre_days": xlre_days,
            "daily_defensive_regime_counts": str(regimes),
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"last_w_{asset}"] = previous_daily_weights.get(asset, 0.0)
        logs.append(log)
        old_rebalance_weights = base_weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(sum(turnovers) / len(turnovers)) if turnovers else float("nan")


def params(source_take, xlp_margin, xlp_cap, xlre_cap, use_bil):
    return {
        "source_take": source_take,
        "risk_weak_max": 0.0,
        "risk_losses_min": 3,
        "cyclical_weak_max": 0.0,
        "broad_breadth_min": 0.75,
        "quality_breadth_min": 0.50,
        "xlp_score_min": 0.0,
        "xlp_vs_xlv_margin": xlp_margin,
        "strong_xlp_cap": xlp_cap,
        "broad_xlp_cap": min(0.05, xlp_cap),
        "xlre_score_min": 0.0,
        "xlre_cap": xlre_cap,
        "xlre_can_use_bil": use_bil,
        "xlu_score_min": 0.0,
        "xlu_cap": 0.05,
        "xlv_score_min": 0.0,
        "xlv_cap": 0.07,
    }


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
        [0.35, 0.50],
        [0.000, 0.005, 0.010],
        [0.05, 0.07, 0.10],
        [0.05, 0.08],
        [False, True],
    )
    for source_take, xlp_margin, xlp_cap, xlre_cap, use_bil in specs:
        defensive_params = params(source_take, xlp_margin, xlp_cap, xlre_cap, use_bil)
        name = f"022B_take{source_take:.2f}_xlp{xlp_cap:.2f}_m{xlp_margin:.3f}_xlre{xlre_cap:.2f}_bil{int(use_bil)}"
        returns, log, avg_turn = run_backtest(
            rebalance,
            prices,
            cfg,
            defensive_params,
            r20c.guarded_europe_params(0.08, 0.25, True),
        )
        common_m = period_metrics(returns, common_start, common_end)
        rows.append({
            "model": name,
            "common_ann_return": common_m["ann"],
            "common_sharpe": common_m["sharpe"],
            "common_max_drawdown": common_m["dd"],
            "delta_ann_vs_020c": common_m["ann"] - base_common["ann"],
            "delta_sharpe_vs_020c": common_m["sharpe"] - base_common["sharpe"],
            "delta_dd_vs_020c": common_m["dd"] - base_common["dd"],
            "defensive_days": int(log["daily_defensive_regime_days"].sum()),
            "xlv_days": int(log["daily_xlv_days"].sum()),
            "xlp_days": int(log["daily_xlp_days"].sum()),
            "xlu_days": int(log["daily_xlu_days"].sum()),
            "xlre_days": int(log["daily_xlre_days"].sum()),
            "avg_xlv_weight": float(log["last_w_XLV"].mean()),
            "avg_xlp_weight": float(log["last_w_XLP"].mean()),
            "avg_xlu_weight": float(log["last_w_XLU"].mean()),
            "avg_xlre_weight": float(log["last_w_XLRE"].mean()),
            "selection_score": (
                (common_m["ann"] - base_common["ann"])
                + 0.8 * (common_m["sharpe"] - base_common["sharpe"])
                + 3.0 * min(0.0, common_m["dd"] - base_common["dd"])
            ),
            **defensive_params,
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

    print("BASE_020C_COMMON", base_common)
    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "delta_ann_vs_020c", "delta_sharpe_vs_020c", "delta_dd_vs_020c",
        "defensive_days", "xlv_days", "xlp_days", "xlu_days", "xlre_days",
        "avg_xlv_weight", "avg_xlp_weight", "avg_xlu_weight", "avg_xlre_weight",
    ]
    print(results[cols].head(20).to_string(index=False))
    print("\nBEST PERIODS")
    print(period_df[period_df["model"].eq(best_name)].to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv, {OUT_PREFIX}_period_validation.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
