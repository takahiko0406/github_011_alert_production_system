import itertools

import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e
import research_013h_final_soxx_soxl_validation as r13h
import research_014b_iwm_peer_leadership_tna as r14b
import research_015a_xlv_defensive_overlay as r15a
import research_015e_daily_macro_defense_brake as r15e
import research_016a_xlf_credit_cycle as r16a


OUT_PREFIX = "model_c_plus_018A_gld_macro_hedge"
EXTRA_ASSETS = sorted(set(r16a.EXTRA_ASSETS + ["GLD", "TLT", "IEF", "SHY", "UUP"]))
EXEC_ASSETS_EXT = sorted(set(r16a.EXEC_ASSETS_EXT + ["GLD"]))

SOURCE_GROUPS = {
    "growth": ["TQQQ", "SOXL", "SOXX", "QQQM"],
    "cyclicals": ["IWM", "XLE", "XLI", "XLB", "XLF"],
    "all_risk": ["TQQQ", "SOXL", "SOXX", "QQQM", "IWM", "XLE", "XLI", "XLB", "XLF"],
    "defensive_bucket": r15e.DEF_ASSETS,
    "bil": ["BIL"],
}


def download_prices(start_date):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + r14b.PEERS))
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


def build_daily_gold_features(prices):
    d = pd.DataFrame(index=prices.index)
    for window in [3, 5, 10, 21]:
        for asset in ["GLD", "TLT", "IEF", "SHY", "BIL", "QQQM", "SOXX", "IWM", "UUP"]:
            d[f"{asset.lower()}_ret_{window}"] = prices[asset].pct_change(window)
        risk_cols = [f"{asset.lower()}_ret_{window}" for asset in ["QQQM", "SOXX", "IWM"]]
        d[f"risk_avg_ret_{window}"] = d[risk_cols].mean(axis=1)
        d[f"gld_vs_bil_{window}"] = d[f"gld_ret_{window}"] - d[f"bil_ret_{window}"]
        d[f"gld_vs_risk_{window}"] = d[f"gld_ret_{window}"] - d[f"risk_avg_ret_{window}"]
        d[f"gld_vs_tlt_{window}"] = d[f"gld_ret_{window}"] - d[f"tlt_ret_{window}"]
        d[f"gld_vs_uup_{window}"] = d[f"gld_ret_{window}"] - d[f"uup_ret_{window}"]
        d[f"ief_vs_shy_{window}"] = d[f"ief_ret_{window}"] - d[f"shy_ret_{window}"]
    return d.fillna(0.0)


def daily_gld_signal(day_features, macro_row, params):
    risk_off = m.safe_float(macro_row, "risk_off_strength", 0.0)
    crash = m.safe_float(macro_row, "crash_pressure", 0.0)
    growth = m.safe_float(macro_row, "growth_strength", 0.0)
    soxx_strength = m.safe_float(macro_row, "soxx_strength", 0.0)
    breakdown = m.safe_float(macro_row, "breakdown_score", 0.0)

    gld_ret_5d = float(day_features.get("gld_ret_5", 0.0))
    gld_ret_10d = float(day_features.get("gld_ret_10", 0.0))
    gld_vs_bil_5d = float(day_features.get("gld_vs_bil_5", 0.0))
    gld_vs_risk_5d = float(day_features.get("gld_vs_risk_5", 0.0))
    gld_vs_risk_10d = float(day_features.get("gld_vs_risk_10", 0.0))
    gld_vs_tlt_5d = float(day_features.get("gld_vs_tlt_5", 0.0))
    gld_vs_uup_10d = float(day_features.get("gld_vs_uup_10", 0.0))
    rate_easing_10d = float(day_features.get("ief_vs_shy_10", 0.0))
    qqqm_ret_5d = float(day_features.get("qqqm_ret_5", 0.0))
    soxx_ret_5d = float(day_features.get("soxx_ret_5", 0.0))

    growth_state = 0.50 * growth + 0.50 * soxx_strength
    equity_stress = (
        breakdown >= params["breakdown_min"]
        or qqqm_ret_5d <= params["qqqm_5_max"]
        or soxx_ret_5d <= params["soxx_5_max"]
        or gld_vs_risk_5d >= params["gld_vs_risk_5_min"]
    )
    bond_not_dominant = gld_vs_tlt_5d >= params["gld_vs_tlt_5_min"]
    score = (
        0.25 * gld_vs_bil_5d
        + 0.25 * gld_vs_risk_5d
        + 0.15 * gld_vs_risk_10d
        + 0.15 * gld_vs_uup_10d
        + 0.10 * max(0.0, risk_off)
        + 0.10 * max(0.0, -growth_state)
    )
    on = (
        equity_stress
        and gld_ret_5d >= params["gld_5_min"]
        and gld_ret_10d >= params["gld_10_min"]
        and gld_vs_bil_5d >= params["gld_vs_bil_5_min"]
        and gld_vs_uup_10d >= params["gld_vs_uup_10_min"]
        and (params["require_gld_beats_tlt"] is False or bond_not_dominant)
        and params["risk_min"] <= risk_off <= params["risk_max"]
        and crash <= params["crash_max"]
        and score >= params["score_min"]
    )
    return on, score


def apply_gld_overlay(weights, day_features, macro_row, params):
    weights = dict(weights)
    on, score = daily_gld_signal(day_features, macro_row, params)
    if not on:
        return weights, False, score

    source_assets = SOURCE_GROUPS[params["source_group"]]
    source_total = sum(weights.get(asset, 0.0) for asset in source_assets)
    if source_total <= 0:
        return weights, False, score
    move = min(source_total * params["source_take"], params["gld_cap"])
    if move <= 1e-12:
        return weights, False, score

    for asset in source_assets:
        weight = weights.get(asset, 0.0)
        if weight > 0:
            cut = move * weight / source_total
            weights[asset] = weight - cut
            weights["GLD"] = weights.get("GLD", 0.0) + cut
    return normalize(weights), True, score


def run_backtest(rebalance, prices, cfg, params, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + r14b.PEERS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    defensive_features = r15e.build_daily_defensive_features(prices)
    gold_features = build_daily_gold_features(prices)
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
        gld_days = 0
        last_gld_score = 0.0
        period_returns = []
        for j, hold_date in enumerate(hold_dates):
            day_weights = base_weights.copy()
            day_weights, defense_on, destination, defense_score, cyc = r15e.apply_daily_brake(
                day_weights, defensive_features.loc[hold_date], row, r16a.PARAMS_015H
            )
            day_weights, gld_on, gld_score = apply_gld_overlay(day_weights, gold_features.loc[hold_date], row, params)
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
            if gld_on:
                gld_days += 1
                last_gld_score = gld_score

        portfolio.loc[hold_dates] = period_returns
        log = {
            "date": date,
            "turnover": rebalance_turnover,
            "soxx_overlay_on": soxx_on,
            "soxl_overlay_on": soxl_on,
            "iwm_overlay_on": iwm_on,
            "daily_defense_active_days": defense_days,
            "daily_gld_active_days": gld_days,
            "daily_gld_last_score": last_gld_score,
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

    baseline_returns, _, _ = r15e.run_backtest(rebalance, prices, cfg, r16a.PARAMS_015H)
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    early_start, early_end = pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")
    recent_start, recent_end = pd.Timestamp("2025-01-01"), common_end
    rate_stress_start, rate_stress_end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31")
    base_common = period_metrics(baseline_returns, common_start, common_end)
    base_early = period_metrics(baseline_returns, early_start, early_end)
    base_recent = period_metrics(baseline_returns, recent_start, recent_end)
    base_rate_stress = period_metrics(baseline_returns, rate_stress_start, rate_stress_end)

    rows = []
    logs_by_name = {}
    returns_by_name = {}
    for cap, source_take, gld5, gld10, gld_vs_risk, gld_vs_uup, beats_tlt, source_group in itertools.product(
        [0.05, 0.08, 0.10],
        [0.25],
        [0.000],
        [-0.005],
        [0.005, 0.010],
        [-0.005],
        [False, True],
        ["all_risk", "cyclicals"],
    ):
        params = {
            "gld_cap": cap,
            "source_take": source_take,
            "gld_5_min": gld5,
            "gld_10_min": gld10,
            "gld_vs_bil_5_min": 0.000,
            "gld_vs_risk_5_min": gld_vs_risk,
            "gld_vs_uup_10_min": gld_vs_uup,
            "gld_vs_tlt_5_min": -0.005,
            "require_gld_beats_tlt": beats_tlt,
            "risk_min": 0.00,
            "risk_max": 1.75,
            "crash_max": 2.0,
            "breakdown_min": 1.0,
            "qqqm_5_max": -0.012,
            "soxx_5_max": -0.025,
            "score_min": 0.003,
            "source_group": source_group,
        }
        name = (
            f"018A_cap{cap:.2f}_take{source_take:.2f}_g{gld5:.3f}-{gld10:.3f}_"
            f"edge{gld_vs_risk:.3f}_uup{gld_vs_uup:.3f}_bt{int(beats_tlt)}_{source_group}"
        )
        returns, log, avg_turn = run_backtest(rebalance, prices, cfg, params)
        common_m = period_metrics(returns, common_start, common_end)
        early_m = period_metrics(returns, early_start, early_end)
        recent_m = period_metrics(returns, recent_start, recent_end)
        rate_stress_m = period_metrics(returns, rate_stress_start, rate_stress_end)
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
            "rate_stress_sharpe": rate_stress_m["sharpe"],
            "delta_early_sharpe_vs_015h": early_m["sharpe"] - base_early["sharpe"],
            "delta_recent_sharpe_vs_015h": recent_m["sharpe"] - base_recent["sharpe"],
            "delta_rate_stress_sharpe_vs_015h": rate_stress_m["sharpe"] - base_rate_stress["sharpe"],
            "selection_score": (
                (common_m["ann"] - base_common["ann"]) * 1.0
                + (common_m["sharpe"] - base_common["sharpe"]) * 0.8
                + min(0.0, common_m["dd"] - base_common["dd"]) * 3.0
                + (rate_stress_m["sharpe"] - base_rate_stress["sharpe"]) * 0.15
            ),
            "gld_rebalance_count": int((log["daily_gld_active_days"] > 0).sum()),
            "gld_active_days": int(log["daily_gld_active_days"].sum()),
            "gld_last_avg_weight": float(log["last_w_GLD"].mean()),
            "bil_last_avg_weight": float(log["last_w_BIL"].mean()),
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
        "delta_common_ann_vs_015h", "delta_common_sharpe_vs_015h", "delta_common_dd_vs_015h",
        "early_sharpe", "recent_sharpe", "rate_stress_sharpe",
        "gld_rebalance_count", "gld_active_days", "gld_last_avg_weight", "bil_last_avg_weight",
        "source_group", "gld_cap", "source_take", "gld_5_min", "gld_10_min",
        "gld_vs_risk_5_min", "gld_vs_uup_10_min", "require_gld_beats_tlt",
    ]
    print("BASE_015H_COMMON", base_common)
    print(results[cols].head(50).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
