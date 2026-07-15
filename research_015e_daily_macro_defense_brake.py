import itertools

import numpy as np
import pandas as pd

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e
import research_013h_final_soxx_soxl_validation as r13h
import research_014b_iwm_peer_leadership_tna as r14b
import research_015a_xlv_defensive_overlay as r15a
import research_015b_fast_defensive_breadth as r15b


OUT_PREFIX = "model_c_plus_015E_daily_macro_defense_brake"
DEF_ASSETS = ["XLV", "XLU", "XLP", "XLRE"]
RISK_ASSETS = ["QQQM", "SOXX", "IWM", "XSOE"]
EXEC_ASSETS_EXT = m.EXEC_ASSETS + ["SOXX", "SOXL", "IWM", "TNA"] + DEF_ASSETS


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {"ann": m.annualized_return(r), "sharpe": m.sharpe_ratio(r), "dd": m.max_drawdown(r)}


def normalize(weights):
    total = sum(weights.values())
    return {asset: weight / total for asset, weight in weights.items()} if total > 0 else weights


def build_daily_defensive_features(prices):
    d = pd.DataFrame(index=prices.index)
    for window in [3, 5, 10, 21]:
        for asset in DEF_ASSETS + RISK_ASSETS:
            d[f"{asset.lower()}_ret_{window}"] = prices[asset].pct_change(window)
        d[f"def_avg_ret_{window}"] = d[[f"{a.lower()}_ret_{window}" for a in DEF_ASSETS]].mean(axis=1)
        d[f"risk_avg_ret_{window}"] = d[[f"{a.lower()}_ret_{window}" for a in RISK_ASSETS]].mean(axis=1)
        d[f"def_edge_{window}"] = d[f"def_avg_ret_{window}"] - d[f"risk_avg_ret_{window}"]
        d[f"def_win_count_{window}"] = sum((d[f"{a.lower()}_ret_{window}"] > 0.0).astype(int) for a in DEF_ASSETS)
        d[f"risk_loss_count_{window}"] = sum((d[f"{a.lower()}_ret_{window}"] < 0.0).astype(int) for a in RISK_ASSETS)
        def_cols = [f"{a.lower()}_ret_{window}" for a in DEF_ASSETS]
        d[f"best_def_asset_{window}"] = d[def_cols].fillna(-999.0).idxmax(axis=1).str.replace("_ret_" + str(window), "", regex=False).str.upper()
    return d.fillna(0.0)


def daily_defense_signal(day_features, macro_row, params):
    risk_off = m.safe_float(macro_row, "risk_off_strength", 0.0)
    crash = m.safe_float(macro_row, "crash_pressure", 0.0)
    credit = m.safe_float(macro_row, "credit_strength", m.safe_float(macro_row, "hyg_strength", 0.0))
    growth = m.safe_float(macro_row, "growth_strength", 0.0)
    soxx_strength = m.safe_float(macro_row, "soxx_strength", 0.0)
    industrial = m.safe_float(macro_row, "industrial_strength", 0.0)
    materials = m.safe_float(macro_row, "materials_strength", 0.0)
    copper = m.safe_float(macro_row, "copper_strength", 0.0)
    breakdown = m.safe_float(macro_row, "breakdown_score", 0.0)

    def_edge_3 = float(day_features.get("def_edge_3", 0.0))
    def_edge_5 = float(day_features.get("def_edge_5", 0.0))
    def_edge_10 = float(day_features.get("def_edge_10", 0.0))
    def_wins_3 = float(day_features.get("def_win_count_3", 0.0))
    def_wins_5 = float(day_features.get("def_win_count_5", 0.0))
    risk_losses_3 = float(day_features.get("risk_loss_count_3", 0.0))
    risk_losses_5 = float(day_features.get("risk_loss_count_5", 0.0))
    qqqm_3 = float(day_features.get("qqqm_ret_3", 0.0))
    qqqm_5 = float(day_features.get("qqqm_ret_5", 0.0))
    soxx_3 = float(day_features.get("soxx_ret_3", 0.0))
    soxx_5 = float(day_features.get("soxx_ret_5", 0.0))

    growth_breakdown = (
        breakdown >= params["breakdown_min"]
        or soxx_3 <= params["soxx_3_max"]
        or soxx_5 <= params["soxx_5_max"]
        or qqqm_3 <= params["qqqm_3_max"]
        or qqqm_5 <= params["qqqm_5_max"]
    )
    defensive_breadth = (
        def_edge_3 >= params["def_edge_3_min"]
        and def_edge_5 >= params["def_edge_5_min"]
        and def_wins_3 >= params["def_wins_3_min"]
        and def_wins_5 >= params["def_wins_5_min"]
    )
    cyclical_support = 0.40 * industrial + 0.30 * materials + 0.20 * copper + 0.10 * credit
    growth_state = 0.50 * growth + 0.50 * soxx_strength
    score = (
        0.30 * def_edge_3
        + 0.25 * def_edge_5
        + 0.10 * def_edge_10
        + 0.15 * max(0.0, risk_off)
        + 0.10 * max(0.0, -growth_state)
        + 0.10 * max(0.0, -cyclical_support)
    )
    on = (
        defensive_breadth
        and growth_breakdown
        and risk_losses_3 >= params["risk_losses_3_min"]
        and risk_losses_5 >= params["risk_losses_5_min"]
        and params["risk_min"] <= risk_off <= params["risk_max"]
        and crash <= params["crash_max"]
        and score >= params["score_min"]
    )
    return on, score, cyclical_support


def apply_daily_brake(weights, day_features, macro_row, params):
    weights = dict(weights)
    on, score, cyc = daily_defense_signal(day_features, macro_row, params)
    if not on:
        return weights, False, "", score, cyc

    destination = "BIL"
    if params["destination"] == "best_defensive":
        destination = day_features.get(f"best_def_asset_{params['leader_window']}", "XLV")
        if destination not in DEF_ASSETS:
            destination = "XLV"

    sources = ["SOXL", "TQQQ", "SOXX", "QQQM", "IWM", "XSOE"]
    if cyc <= params["cyclical_weak_max"]:
        sources += ["UXI", "ERX", "XLI", "XLB", "XLE"]

    moved = 0.0
    for source in sources:
        if moved >= params["def_cap"] - 1e-12:
            break
        source_w = weights.get(source, 0.0)
        if source_w <= 0:
            continue
        take = min(source_w * params["source_take"], params["def_cap"] - moved)
        weights[source] -= take
        weights[destination] = weights.get(destination, 0.0) + take
        moved += take
    return normalize(weights), moved > 1e-12, destination, score, cyc


def run_backtest(rebalance, prices, cfg, params, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + r15b.EXTRA_ASSETS + r14b.PEERS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    daily_features = build_daily_defensive_features(prices)
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
        active_days = 0
        last_destination = ""
        last_score = 0.0
        last_cyc = 0.0
        period_returns = []
        daily_turnover_cost = 0.0
        for j, hold_date in enumerate(hold_dates):
            day_features = daily_features.loc[hold_date]
            day_weights, on, destination, score, cyc = apply_daily_brake(base_weights, day_features, row, params)
            daily_turnover = sum(abs(day_weights.get(asset, 0.0) - previous_daily_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
            if j == 0:
                daily_turnover += rebalance_turnover
            ret = 0.0
            for asset, weight in day_weights.items():
                if abs(weight) > 1e-12:
                    ret += weight * asset_returns.at[hold_date, asset]
            ret -= daily_turnover * tx_cost
            period_returns.append(ret)
            daily_turnover_cost += daily_turnover * tx_cost
            previous_daily_weights = day_weights
            if on:
                active_days += 1
                last_destination = destination
                last_score = score
                last_cyc = cyc

        portfolio.loc[hold_dates] = period_returns
        log = {
            "date": date,
            "turnover": rebalance_turnover,
            "daily_turnover_cost": daily_turnover_cost,
            "soxx_overlay_on": soxx_on,
            "soxl_overlay_on": soxl_on,
            "soxl_ladder_fraction": soxl_frac,
            "iwm_overlay_on": iwm_on,
            "tna_overlay_on": tna_on,
            "daily_defense_active_days": active_days,
            "daily_defense_last_destination": last_destination,
            "daily_defense_last_score": last_score,
            "cyclical_support": last_cyc,
            "iwm_leadership_score": iwm_score,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"base_w_{asset}"] = base_weights.get(asset, 0.0)
            log[f"last_w_{asset}"] = previous_daily_weights.get(asset, 0.0)
        logs.append(log)
        old_rebalance_weights = base_weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(np.mean(turnovers)) if turnovers else np.nan


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = r15b.download_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, _, _ = r14b.run_backtest(rebalance, prices, cfg, r15a.PARAMS_014D)
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
    for cap, source_take, edge3, edge5, wins3, risk_min, destination in itertools.product(
        [0.05, 0.10, 0.15],
        [0.35, 0.60],
        [0.010, 0.020, 0.030],
        [0.010, 0.020],
        [2, 3],
        [0.00, 0.20],
        ["BIL", "best_defensive"],
    ):
        params = {
            "def_cap": cap,
            "source_take": source_take,
            "def_edge_3_min": edge3,
            "def_edge_5_min": edge5,
            "def_wins_3_min": wins3,
            "def_wins_5_min": 2,
            "risk_losses_3_min": 2,
            "risk_losses_5_min": 1,
            "risk_min": risk_min,
            "risk_max": 1.50,
            "crash_max": 1.75,
            "breakdown_min": 1.0,
            "soxx_3_max": -0.025,
            "soxx_5_max": -0.035,
            "qqqm_3_max": -0.012,
            "qqqm_5_max": -0.018,
            "score_min": 0.00,
            "leader_window": 3,
            "cyclical_weak_max": -0.10,
            "destination": destination,
        }
        name = (
            f"015E_cap{cap:.2f}_take{source_take:.2f}_e{edge3:.3f}-{edge5:.3f}_"
            f"w{wins3}_risk{risk_min:.2f}_{destination}"
        )
        returns, log, avg_turn = run_backtest(rebalance, prices, cfg, params)
        common_m = period_metrics(returns, common_start, common_end)
        early_m = period_metrics(returns, early_start, early_end)
        recent_m = period_metrics(returns, recent_start, recent_end)
        row = m.summary(name, returns, avg_turn, len(log), m.DEFAULT_TRANSACTION_COST)
        row.update({
            "common_ann_return": common_m["ann"],
            "common_sharpe": common_m["sharpe"],
            "common_max_drawdown": common_m["dd"],
            "early_sharpe": early_m["sharpe"],
            "recent_sharpe": recent_m["sharpe"],
            "delta_common_ann_vs_014d": common_m["ann"] - base_common["ann"],
            "delta_common_sharpe_vs_014d": common_m["sharpe"] - base_common["sharpe"],
            "delta_common_dd_vs_014d": common_m["dd"] - base_common["dd"],
            "delta_early_sharpe_vs_014d": early_m["sharpe"] - base_early["sharpe"],
            "delta_recent_sharpe_vs_014d": recent_m["sharpe"] - base_recent["sharpe"],
            "selection_score": (
                (common_m["ann"] - base_common["ann"]) * 1.2
                + (common_m["sharpe"] - base_common["sharpe"]) * 0.7
                + (common_m["dd"] - base_common["dd"]) * 1.5
                + (early_m["sharpe"] - base_early["sharpe"]) * 0.25
            ),
            "defense_rebalance_count": int((log["daily_defense_active_days"] > 0).sum()),
            "defense_active_days": int(log["daily_defense_active_days"].sum()),
            "avg_daily_turnover_cost": float(log["daily_turnover_cost"].mean()),
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
        "delta_common_ann_vs_014d", "delta_common_sharpe_vs_014d", "delta_common_dd_vs_014d",
        "early_sharpe", "recent_sharpe", "defense_rebalance_count", "defense_active_days",
        "avg_daily_turnover_cost", "destination",
    ]
    print("BASE_014D_COMMON", base_common)
    print(results[cols].head(30).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
