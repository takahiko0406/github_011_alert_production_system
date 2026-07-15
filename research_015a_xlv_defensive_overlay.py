import itertools

import numpy as np
import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e
import research_013h_final_soxx_soxl_validation as r13h
import research_014b_iwm_peer_leadership_tna as r14b


OUT_PREFIX = "model_c_plus_015A_xlv_defensive_overlay"
EXTRA_ASSETS = ["SOXX", "SOXL", "IWM", "TNA", "XLV", "XLF", "SHY", "IEF"]
EXEC_ASSETS_EXT = m.EXEC_ASSETS + ["SOXX", "SOXL", "IWM", "TNA", "XLV"]
DEFENSIVE_PEERS = ["QQQM", "XLI", "XLB", "XLE", "IWM"]

PARAMS_014D = {
    "iwm_target": 0.30,
    "source_take_max": 0.75,
    "min_wins_63": 3,
    "min_wins_21": 3,
    "edge_63_min": 0.0,
    "edge_21_min": 0.0,
    "ma63_min": 0.0,
    "credit_min": 0.0,
    "rates_min": 0.0,
    "risk_max": 0.60,
    "crash_max": 1.25,
    "leadership_min": 0.0,
    "iwm_to_tna": 0.0,
    "tna_cap": 0.10,
    "tna_min_wins_21": 3,
    "tna_edge_63_min": 0.0,
    "tna_edge_21_min": 0.0,
    "tna_credit_min": 0.0,
    "tna_rates_min": 0.0,
    "tna_risk_max": 0.60,
    "tna_leadership_min": 0.0,
}


def download_prices(start_date):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + DEFENSIVE_PEERS))
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


def add_xlv_defensive_features(rebalance, prices):
    d = rebalance.copy()
    for window in [21, 63, 126]:
        d[f"xlv_ret_{window}"] = [
            prices["XLV"].pct_change(window).reindex([dt]).iloc[0] if dt in prices.index else np.nan
            for dt in d["aligned_date"]
        ]
        for peer in DEFENSIVE_PEERS:
            d[f"{peer.lower()}_ret_{window}"] = [
                prices[peer].pct_change(window).reindex([dt]).iloc[0] if dt in prices.index else np.nan
                for dt in d["aligned_date"]
            ]
        peer_cols = [f"{peer.lower()}_ret_{window}" for peer in DEFENSIVE_PEERS]
        d[f"xlv_peer_edge_{window}"] = d[f"xlv_ret_{window}"] - d[peer_cols].mean(axis=1)
        d[f"xlv_peer_win_count_{window}"] = sum(
            (d[f"xlv_ret_{window}"] > d[f"{peer.lower()}_ret_{window}"]).astype(int)
            for peer in DEFENSIVE_PEERS
        )

    d["xlv_ma63_ratio"] = [
        prices["XLV"].reindex([dt]).iloc[0] / prices["XLV"].rolling(63).mean().reindex([dt]).iloc[0] - 1.0
        if dt in prices.index else np.nan
        for dt in d["aligned_date"]
    ]
    d["xlv_vs_bil_63"] = [
        prices["XLV"].pct_change(63).reindex([dt]).iloc[0] - prices["BIL"].pct_change(63).reindex([dt]).iloc[0]
        if dt in prices.index else np.nan
        for dt in d["aligned_date"]
    ]
    d["xlv_vs_qqqm_63"] = d["xlv_ret_63"] - d["qqqm_ret_63"]
    return d.fillna({
        "xlv_ret_21": 0.0,
        "xlv_ret_63": 0.0,
        "xlv_ret_126": 0.0,
        "xlv_peer_edge_21": 0.0,
        "xlv_peer_edge_63": 0.0,
        "xlv_peer_edge_126": 0.0,
        "xlv_peer_win_count_21": 0,
        "xlv_peer_win_count_63": 0,
        "xlv_peer_win_count_126": 0,
        "xlv_ma63_ratio": 0.0,
        "xlv_vs_bil_63": 0.0,
        "xlv_vs_qqqm_63": 0.0,
    })


def normalize(weights):
    total = sum(weights.values())
    return {asset: weight / total for asset, weight in weights.items()} if total > 0 else weights


def apply_xlv_overlay(weights, row, params):
    weights = dict(weights)
    risk = m.safe_float(row, "risk_off_strength", 0.0)
    crash = m.safe_float(row, "crash_pressure", 0.0)
    growth = m.safe_float(row, "growth_strength", 0.0)
    defensive = m.safe_float(row, "defensive_strength", 0.0)
    edge_21 = m.safe_float(row, "xlv_peer_edge_21", 0.0)
    edge_63 = m.safe_float(row, "xlv_peer_edge_63", 0.0)
    edge_126 = m.safe_float(row, "xlv_peer_edge_126", 0.0)
    wins_21 = m.safe_float(row, "xlv_peer_win_count_21", 0.0)
    wins_63 = m.safe_float(row, "xlv_peer_win_count_63", 0.0)
    ma63 = m.safe_float(row, "xlv_ma63_ratio", 0.0)
    vs_bil = m.safe_float(row, "xlv_vs_bil_63", 0.0)
    vs_qqqm = m.safe_float(row, "xlv_vs_qqqm_63", 0.0)

    leadership_score = (
        0.40 * edge_63
        + 0.20 * edge_21
        + 0.15 * edge_126
        + 0.15 * (defensive - growth)
        + 0.10 * vs_bil
    )
    mild_defense = params["risk_min"] <= risk <= params["risk_max"]
    strong_defense = defensive - growth >= params["defensive_gap_min"] or vs_qqqm >= params["vs_qqqm_min"]
    xlv_on = (
        mild_defense
        and crash <= params["crash_max"]
        and wins_63 >= params["min_wins_63"]
        and wins_21 >= params["min_wins_21"]
        and edge_63 >= params["edge_63_min"]
        and edge_21 >= params["edge_21_min"]
        and ma63 >= params["ma63_min"]
        and vs_bil >= params["vs_bil_min"]
        and strong_defense
        and leadership_score >= params["leadership_min"]
    )
    if not xlv_on:
        return weights, False, leadership_score

    target = params["xlv_target"]
    available_sources = params["sources"]
    current_xlv = weights.get("XLV", 0.0)
    needed = max(0.0, target - current_xlv)
    for source in available_sources:
        if needed <= 1e-12:
            break
        source_w = weights.get(source, 0.0)
        if source_w <= 0:
            continue
        take = min(source_w * params["source_take_max"], needed)
        weights[source] -= take
        weights["XLV"] = weights.get("XLV", 0.0) + take
        needed -= take

    return normalize(weights), True, leadership_score


def run_backtest(rebalance, prices, cfg, params):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + DEFENSIVE_PEERS))
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
        weights, soxx_on, soxl_on, soxl_frac = r13e.apply_overlay(weights, row, r13h.PARAMS_013F)
        weights, iwm_on, tna_on, iwm_score = r14b.apply_iwm_tna(weights, row, PARAMS_014D)
        weights, xlv_on, xlv_score = apply_xlv_overlay(weights, row, params)

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
            "iwm_overlay_on": iwm_on,
            "tna_overlay_on": tna_on,
            "iwm_leadership_score": iwm_score,
            "xlv_overlay_on": xlv_on,
            "xlv_leadership_score": xlv_score,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"exec_w_{asset}"] = weights.get(asset, 0.0)
        logs.append(log)
        old_weights = weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(np.mean(turnovers)) if turnovers else np.nan


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    prices = download_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    rebalance = add_xlv_defensive_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, baseline_log, baseline_turn = r14b.run_backtest(rebalance, prices, cfg, PARAMS_014D)
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
    source_sets = {
        "cash_cyc": ["BIL", "XLI", "XLB", "XLE"],
        "cash_only": ["BIL"],
        "cyc_only": ["XLI", "XLB", "XLE"],
        "weak_equity": ["BIL", "XLI", "XLB", "XLE", "QQQM"],
    }

    for target, take_max, risk_min, risk_max, wins63, wins21, edge63, edge21, vs_bil, vs_qqqm, def_gap, source_key in itertools.product(
        [0.10, 0.15],
        [0.50],
        [0.25],
        [0.75, 1.00],
        [2, 3],
        [2],
        [-0.02, 0.00],
        [-0.02],
        [-0.01, 0.00],
        [-0.03],
        [-0.25],
        ["cash_cyc", "cyc_only"],
    ):
        if risk_min > risk_max:
            continue
        params = {
            "xlv_target": target,
            "source_take_max": take_max,
            "risk_min": risk_min,
            "risk_max": risk_max,
            "crash_max": 1.25,
            "min_wins_63": wins63,
            "min_wins_21": wins21,
            "edge_63_min": edge63,
            "edge_21_min": edge21,
            "ma63_min": -0.03,
            "vs_bil_min": vs_bil,
            "vs_qqqm_min": vs_qqqm,
            "defensive_gap_min": def_gap,
            "leadership_min": -0.01,
            "sources": source_sets[source_key],
            "source_key": source_key,
        }
        name = (
            f"015A_xlv{target:.2f}_take{take_max:.2f}_risk{risk_min:.2f}-{risk_max:.2f}_"
            f"w{wins63}-{wins21}_e{edge63:.2f}-{edge21:.2f}_"
            f"bil{vs_bil:.2f}_qqq{vs_qqqm:.2f}_dg{def_gap:.2f}_{source_key}"
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
            "beats_014d_return": common_m["ann"] > base_common["ann"],
            "beats_014d_sharpe": common_m["sharpe"] > base_common["sharpe"],
            "no_worse_dd": common_m["dd"] >= base_common["dd"] - 1e-9,
            "selection_score": (
                (common_m["ann"] - base_common["ann"]) * 1.5
                + (common_m["sharpe"] - base_common["sharpe"]) * 0.7
                + (early_m["sharpe"] - base_early["sharpe"]) * 0.35
                + min(0.0, common_m["dd"] - base_common["dd"]) * 1.25
            ),
            "xlv_overlay_rebalances": int(log["xlv_overlay_on"].sum()),
            "xlv_avg_weight": float(log["exec_w_XLV"].mean()),
            "xlv_max_weight": float(log["exec_w_XLV"].max()),
            "iwm_overlay_rebalances": int(log["iwm_overlay_on"].sum()),
            "soxx_overlay_rebalances": int(log["soxx_overlay_on"].sum()),
            "soxl_overlay_rebalances": int(log["soxl_overlay_on"].sum()),
            **{k: v for k, v in params.items() if k != "sources"},
        })
        rows.append(row)
        logs_by_name[name] = log
        returns_by_name[name] = returns

    results = pd.DataFrame(rows).sort_values(
        ["beats_014d_return", "beats_014d_sharpe", "no_worse_dd", "selection_score", "common_sharpe"],
        ascending=[False, False, False, False, False],
    )
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    best_name = results.iloc[0]["model"]
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])

    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "delta_common_ann_vs_014d", "delta_common_sharpe_vs_014d", "delta_common_dd_vs_014d",
        "early_sharpe", "delta_early_sharpe_vs_014d", "recent_sharpe", "delta_recent_sharpe_vs_014d",
        "xlv_overlay_rebalances", "xlv_avg_weight", "xlv_max_weight", "source_key",
    ]
    print("BASE_014D_COMMON", base_common)
    print("BASE_014D_EARLY", base_early)
    print("BASE_014D_RECENT", base_recent)
    print(results[cols].head(30).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
