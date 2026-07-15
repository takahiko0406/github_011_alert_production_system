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
import research_015e_daily_macro_defense_brake as r15e


OUT_PREFIX = "model_c_plus_016A_xlf_credit_cycle"
EXTRA_ASSETS = sorted(set(r15b.EXTRA_ASSETS + ["XLF"]))
EXEC_ASSETS_EXT = m.EXEC_ASSETS + ["SOXX", "SOXL", "IWM", "TNA"] + r15e.DEF_ASSETS + ["XLF"]
PEERS = ["QQQM", "IWM", "XLI", "XLB", "XLE"]
PARAMS_015H = {
    "def_cap": 0.07,
    "source_take": 0.50,
    "def_edge_3_min": 0.010,
    "def_edge_5_min": 0.010,
    "def_wins_3_min": 2,
    "def_wins_5_min": 2,
    "risk_losses_3_min": 2,
    "risk_losses_5_min": 1,
    "risk_min": 0.00,
    "risk_max": 1.50,
    "crash_max": 1.75,
    "breakdown_min": 1.0,
    "soxx_3_max": -0.025,
    "soxx_5_max": -0.035,
    "qqqm_3_max": -0.012,
    "qqqm_5_max": -0.018,
    "score_min": 0.00,
    "leader_window": 3,
    "cyclical_weak_max": -0.30,
    "destination": "best_defensive",
}


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {"ann": m.annualized_return(r), "sharpe": m.sharpe_ratio(r), "dd": m.max_drawdown(r)}


def add_xlf_credit_features(rebalance, prices):
    d = rebalance.copy()
    for window in [21, 63]:
        d[f"xlf_ret_{window}"] = [
            prices["XLF"].pct_change(window).reindex([dt]).iloc[0] if dt in prices.index else np.nan
            for dt in d["aligned_date"]
        ]
        for peer in PEERS:
            d[f"{peer.lower()}_ret_{window}"] = [
                prices[peer].pct_change(window).reindex([dt]).iloc[0] if dt in prices.index else np.nan
                for dt in d["aligned_date"]
            ]
        peer_cols = [f"{peer.lower()}_ret_{window}" for peer in PEERS]
        d[f"xlf_peer_edge_{window}"] = d[f"xlf_ret_{window}"] - d[peer_cols].mean(axis=1)
        d[f"xlf_peer_win_count_{window}"] = sum(
            (d[f"xlf_ret_{window}"] > d[f"{peer.lower()}_ret_{window}"]).astype(int)
            for peer in PEERS
        )

    d["xlf_ma63_ratio"] = [
        prices["XLF"].reindex([dt]).iloc[0] / prices["XLF"].rolling(63).mean().reindex([dt]).iloc[0] - 1.0
        if dt in prices.index else np.nan
        for dt in d["aligned_date"]
    ]
    d["rates_easing_proxy"] = [
        prices["IEF"].pct_change(63).reindex([dt]).iloc[0] - prices["SHY"].pct_change(63).reindex([dt]).iloc[0]
        if dt in prices.index else np.nan
        for dt in d["aligned_date"]
    ]
    return d.fillna({
        "xlf_ret_21": 0.0,
        "xlf_ret_63": 0.0,
        "xlf_peer_edge_21": 0.0,
        "xlf_peer_edge_63": 0.0,
        "xlf_peer_win_count_21": 0,
        "xlf_peer_win_count_63": 0,
        "xlf_ma63_ratio": 0.0,
        "rates_easing_proxy": 0.0,
    })


def normalize(weights):
    total = sum(weights.values())
    return {asset: weight / total for asset, weight in weights.items()} if total > 0 else weights


def apply_xlf_credit_overlay(weights, row, params):
    weights = dict(weights)
    credit = m.safe_float(row, "credit_strength", m.safe_float(row, "hyg_strength", 0.0))
    hyg = m.safe_float(row, "hyg_strength", 0.0)
    risk = m.safe_float(row, "risk_off_strength", 0.0)
    crash = m.safe_float(row, "crash_pressure", 0.0)
    edge21 = m.safe_float(row, "xlf_peer_edge_21", 0.0)
    edge63 = m.safe_float(row, "xlf_peer_edge_63", 0.0)
    wins21 = m.safe_float(row, "xlf_peer_win_count_21", 0.0)
    wins63 = m.safe_float(row, "xlf_peer_win_count_63", 0.0)
    ma63 = m.safe_float(row, "xlf_ma63_ratio", 0.0)
    xlf21 = m.safe_float(row, "xlf_ret_21", 0.0)
    xlf63 = m.safe_float(row, "xlf_ret_63", 0.0)
    rates_easing = m.safe_float(row, "rates_easing_proxy", 0.0)

    credit_cycle_score = (
        0.30 * edge63
        + 0.20 * edge21
        + 0.20 * credit
        + 0.15 * hyg
        + 0.10 * ma63
        + 0.05 * rates_easing
    )
    xlf_on = (
        wins63 >= params["min_wins_63"]
        and wins21 >= params["min_wins_21"]
        and edge63 >= params["edge_63_min"]
        and edge21 >= params["edge_21_min"]
        and xlf21 >= params["xlf_21_min"]
        and xlf63 >= params["xlf_63_min"]
        and ma63 >= params["ma63_min"]
        and credit >= params["credit_min"]
        and hyg >= params["hyg_min"]
        and risk <= params["risk_max"]
        and crash <= params["crash_max"]
        and credit_cycle_score >= params["score_min"]
    )
    if not xlf_on:
        return weights, False, credit_cycle_score

    target = params["xlf_target"]
    needed = max(0.0, target - weights.get("XLF", 0.0))
    for source in params["sources"]:
        if needed <= 1e-12:
            break
        source_w = weights.get(source, 0.0)
        if source_w <= 0:
            continue
        take = min(source_w * params["source_take"], needed)
        weights[source] -= take
        weights["XLF"] = weights.get("XLF", 0.0) + take
        needed -= take
    return normalize(weights), True, credit_cycle_score


def run_backtest(rebalance, prices, cfg, params, tx_cost=m.DEFAULT_TRANSACTION_COST):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + r14b.PEERS + PEERS))
    asset_returns = prices[assets].pct_change().fillna(0.0)
    daily_features = r15e.build_daily_defensive_features(prices)
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
        base_weights, xlf_on, xlf_score = apply_xlf_credit_overlay(base_weights, row, params)

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
        period_returns = []
        for j, hold_date in enumerate(hold_dates):
            day_features = daily_features.loc[hold_date]
            day_weights, defense_on, destination, defense_score, cyc = r15e.apply_daily_brake(
                base_weights, day_features, row, PARAMS_015H
            )
            for asset in EXEC_ASSETS_EXT:
                day_weights.setdefault(asset, 0.0)
            daily_turnover = sum(abs(day_weights.get(asset, 0.0) - previous_daily_weights.get(asset, 0.0)) for asset in EXEC_ASSETS_EXT)
            if j == 0:
                daily_turnover += rebalance_turnover
            ret = 0.0
            for asset, weight in day_weights.items():
                if abs(weight) > 1e-12:
                    ret += weight * asset_returns.at[hold_date, asset]
            ret -= daily_turnover * tx_cost
            period_returns.append(ret)
            previous_daily_weights = day_weights
            if defense_on:
                active_days += 1
                last_destination = destination

        portfolio.loc[hold_dates] = period_returns
        log = {
            "date": date,
            "turnover": rebalance_turnover,
            "soxx_overlay_on": soxx_on,
            "soxl_overlay_on": soxl_on,
            "soxl_ladder_fraction": soxl_frac,
            "iwm_overlay_on": iwm_on,
            "tna_overlay_on": tna_on,
            "xlf_overlay_on": xlf_on,
            "xlf_credit_cycle_score": xlf_score,
            "daily_defense_active_days": active_days,
            "daily_defense_last_destination": last_destination,
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
    rebalance = add_xlf_credit_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, baseline_log, baseline_turn = r15e.run_backtest(rebalance, prices, cfg, PARAMS_015H)
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    early_start, early_end = pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")
    recent_start, recent_end = pd.Timestamp("2025-01-01"), common_end
    base_common = period_metrics(baseline_returns, common_start, common_end)
    base_early = period_metrics(baseline_returns, early_start, early_end)
    base_recent = period_metrics(baseline_returns, recent_start, recent_end)

    source_sets = {
        "cash_only": ["BIL"],
        "cyc_cash": ["BIL", "XLI", "XLB", "XLE"],
        "broad_cycle": ["BIL", "XLI", "XLB", "XLE", "IWM", "QQQM"],
    }

    rows = []
    logs_by_name = {}
    returns_by_name = {}
    for target, take, wins63, wins21, edge63, edge21, credit_min, hyg_min, risk_max, source_key in itertools.product(
        [0.05, 0.10],
        [0.50],
        [2, 3],
        [2],
        [0.00, 0.02],
        [-0.01],
        [0.00, 0.25],
        [0.00],
        [0.75, 1.25],
        ["cyc_cash", "broad_cycle"],
    ):
        params = {
            "xlf_target": target,
            "source_take": take,
            "min_wins_63": wins63,
            "min_wins_21": wins21,
            "edge_63_min": edge63,
            "edge_21_min": edge21,
            "xlf_21_min": -0.02,
            "xlf_63_min": -0.03,
            "ma63_min": -0.03,
            "credit_min": credit_min,
            "hyg_min": hyg_min,
            "risk_max": risk_max,
            "crash_max": 1.75,
            "score_min": -0.02,
            "sources": source_sets[source_key],
            "source_key": source_key,
        }
        name = (
            f"016A_xlf{target:.2f}_take{take:.2f}_w{wins63}-{wins21}_"
            f"e{edge63:.2f}-{edge21:.2f}_cred{credit_min:.2f}_hyg{hyg_min:.2f}_"
            f"risk{risk_max:.2f}_{source_key}"
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
            "delta_common_ann_vs_015h": common_m["ann"] - base_common["ann"],
            "delta_common_sharpe_vs_015h": common_m["sharpe"] - base_common["sharpe"],
            "delta_common_dd_vs_015h": common_m["dd"] - base_common["dd"],
            "early_sharpe": early_m["sharpe"],
            "recent_sharpe": recent_m["sharpe"],
            "delta_early_sharpe_vs_015h": early_m["sharpe"] - base_early["sharpe"],
            "delta_recent_sharpe_vs_015h": recent_m["sharpe"] - base_recent["sharpe"],
            "selection_score": (
                (common_m["ann"] - base_common["ann"]) * 1.2
                + (common_m["sharpe"] - base_common["sharpe"]) * 0.8
                + min(0.0, common_m["dd"] - base_common["dd"]) * 2.0
                + (early_m["sharpe"] - base_early["sharpe"]) * 0.2
            ),
            "xlf_overlay_rebalances": int(log["xlf_overlay_on"].sum()),
            "xlf_base_avg_weight": float(log["base_w_XLF"].mean()),
            "xlf_base_max_weight": float(log["base_w_XLF"].max()),
            "daily_defense_active_days": int(log["daily_defense_active_days"].sum()),
            **{k: v for k, v in params.items() if k != "sources"},
        })
        rows.append(row)
        logs_by_name[name] = log
        returns_by_name[name] = returns

    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "common_sharpe"], ascending=[False, False]
    )
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    best_name = results.iloc[0]["model"]
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])
    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "delta_common_ann_vs_015h", "delta_common_sharpe_vs_015h", "delta_common_dd_vs_015h",
        "early_sharpe", "recent_sharpe", "xlf_overlay_rebalances",
        "xlf_base_avg_weight", "xlf_base_max_weight", "source_key",
    ]
    print("BASE_015H_COMMON", base_common)
    print(results[cols].head(40).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
