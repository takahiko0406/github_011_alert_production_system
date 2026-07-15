import itertools

import numpy as np
import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e
import research_013h_final_soxx_soxl_validation as r13h


OUT_PREFIX = "model_c_plus_014B_iwm_peer_leadership_tna"
EXTRA_ASSETS = ["SOXX", "SOXL", "IWM", "TNA", "SHY", "IEF"]
EXEC_ASSETS_EXT = m.EXEC_ASSETS + ["SOXX", "SOXL", "IWM", "TNA"]
PEERS = ["QQQM", "XLI", "XLF", "XLB"]


def download_prices(start_date):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + PEERS))
    prices = yf.download(assets, start=start_date, auto_adjust=True, progress=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"]
    prices = prices.sort_index().dropna(how="all").ffill()
    missing = [asset for asset in assets if asset not in prices.columns]
    if missing:
        raise ValueError(f"Missing prices: {missing}")
    return prices


def add_iwm_peer_features(rebalance, prices):
    d = rebalance.copy()
    for window in [21, 63]:
        d[f"iwm_ret_{window}"] = [
            prices["IWM"].pct_change(window).reindex([dt]).iloc[0] if dt in prices.index else np.nan
            for dt in d["aligned_date"]
        ]
        for peer in PEERS:
            d[f"{peer.lower()}_ret_{window}"] = [
                prices[peer].pct_change(window).reindex([dt]).iloc[0] if dt in prices.index else np.nan
                for dt in d["aligned_date"]
            ]
        peer_cols = [f"{peer.lower()}_ret_{window}" for peer in PEERS]
        d[f"iwm_peer_edge_{window}"] = d[f"iwm_ret_{window}"] - d[peer_cols].mean(axis=1)
        d[f"iwm_peer_win_count_{window}"] = sum(
            (d[f"iwm_ret_{window}"] > d[f"{peer.lower()}_ret_{window}"]).astype(int)
            for peer in PEERS
        )

    d["iwm_ma63_ratio"] = [
        prices["IWM"].reindex([dt]).iloc[0] / prices["IWM"].rolling(63).mean().reindex([dt]).iloc[0] - 1.0
        if dt in prices.index else np.nan
        for dt in d["aligned_date"]
    ]
    d["rates_ok_proxy"] = [
        prices["IEF"].pct_change(63).reindex([dt]).iloc[0] - prices["SHY"].pct_change(63).reindex([dt]).iloc[0]
        if dt in prices.index else np.nan
        for dt in d["aligned_date"]
    ]
    return d.fillna({
        "iwm_ret_21": 0.0,
        "iwm_ret_63": 0.0,
        "iwm_peer_edge_21": 0.0,
        "iwm_peer_edge_63": 0.0,
        "iwm_peer_win_count_21": 0,
        "iwm_peer_win_count_63": 0,
        "iwm_ma63_ratio": 0.0,
        "rates_ok_proxy": 0.0,
    })


def normalize(weights):
    total = sum(weights.values())
    return {asset: weight / total for asset, weight in weights.items()} if total > 0 else weights


def apply_iwm_tna(weights, row, params):
    weights = dict(weights)
    credit = m.safe_float(row, "credit_strength", m.safe_float(row, "hyg_strength", 0.0))
    risk = m.safe_float(row, "risk_off_strength", 0.0)
    crash = m.safe_float(row, "crash_pressure", 0.0)
    edge_21 = m.safe_float(row, "iwm_peer_edge_21", 0.0)
    edge_63 = m.safe_float(row, "iwm_peer_edge_63", 0.0)
    wins_21 = m.safe_float(row, "iwm_peer_win_count_21", 0.0)
    wins_63 = m.safe_float(row, "iwm_peer_win_count_63", 0.0)
    ma63 = m.safe_float(row, "iwm_ma63_ratio", 0.0)
    rates_ok = m.safe_float(row, "rates_ok_proxy", 0.0)

    leadership_score = 0.45 * edge_63 + 0.25 * edge_21 + 0.15 * credit + 0.15 * rates_ok
    iwm_on = (
        wins_63 >= params["min_wins_63"]
        and wins_21 >= params["min_wins_21"]
        and edge_63 >= params["edge_63_min"]
        and edge_21 >= params["edge_21_min"]
        and ma63 >= params["ma63_min"]
        and credit >= params["credit_min"]
        and rates_ok >= params["rates_min"]
        and risk <= params["risk_max"]
        and crash <= params["crash_max"]
        and leadership_score >= params["leadership_min"]
    )
    if not iwm_on:
        return weights, False, False, leadership_score

    target = params["iwm_target"]
    available_sources = ["BIL", "XLI", "XLF", "XLB", "QQQM"]
    current_iwm = weights.get("IWM", 0.0)
    needed = max(0.0, target - current_iwm)
    for source in available_sources:
        if needed <= 1e-12:
            break
        source_w = weights.get(source, 0.0)
        if source_w <= 0:
            continue
        take = min(source_w * params["source_take_max"], needed)
        weights[source] -= take
        weights["IWM"] = weights.get("IWM", 0.0) + take
        needed -= take

    tna_on = False
    tna_on_gate = (
        wins_63 >= 4
        and wins_21 >= params["tna_min_wins_21"]
        and edge_63 >= params["tna_edge_63_min"]
        and edge_21 >= params["tna_edge_21_min"]
        and credit >= params["tna_credit_min"]
        and rates_ok >= params["tna_rates_min"]
        and risk <= params["tna_risk_max"]
        and leadership_score >= params["tna_leadership_min"]
    )
    if tna_on_gate:
        iwm_w = weights.get("IWM", 0.0)
        move = min(iwm_w * params["iwm_to_tna"], max(0.0, params["tna_cap"] - weights.get("TNA", 0.0)))
        if move > 0:
            weights["IWM"] -= move
            weights["TNA"] = weights.get("TNA", 0.0) + move
            tna_on = True

    return normalize(weights), True, tna_on, leadership_score


def run_backtest(rebalance, prices, cfg, params):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + PEERS))
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
        weights, iwm_on, tna_on, leadership_score = apply_iwm_tna(weights, row, params)

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
            "iwm_overlay_on": iwm_on,
            "tna_overlay_on": tna_on,
            "iwm_leadership_score": leadership_score,
            **details,
        }
        for asset in EXEC_ASSETS_EXT:
            log[f"exec_w_{asset}"] = weights.get(asset, 0.0)
        logs.append(log)
        old_weights = weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(np.mean(turnovers)) if turnovers else np.nan


def period_metrics(returns, start, end):
    r = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return {"ann": m.annualized_return(r), "sharpe": m.sharpe_ratio(r), "dd": m.max_drawdown(r)}


def main():
    cfg, cfg_msg = m.load_best_cfg()
    print(cfg_msg)
    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    prices = download_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = add_iwm_peer_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    base_returns, _, _ = r13e.run_backtest(rebalance, prices, cfg, r13h.PARAMS_013F)
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
    for iwm_target, take_max, edge63, edge21, wins63, wins21, rates_min, tna_frac, tna_cap in itertools.product(
        [0.05, 0.10],
        [0.25],
        [0.00, 0.03],
        [0.00],
        [3, 4],
        [3, 4],
        [0.00],
        [0.00, 0.25],
        [0.10],
    ):
        params = {
            "iwm_target": iwm_target,
            "source_take_max": take_max,
            "min_wins_63": wins63,
            "min_wins_21": wins21,
            "edge_63_min": edge63,
            "edge_21_min": edge21,
            "ma63_min": 0.0,
            "credit_min": 0.0,
            "rates_min": rates_min,
            "risk_max": 0.75,
            "crash_max": 1.25,
            "leadership_min": 0.0,
            "iwm_to_tna": tna_frac,
            "tna_cap": tna_cap,
            "tna_min_wins_21": 4,
            "tna_edge_63_min": max(0.03, edge63),
            "tna_edge_21_min": max(0.02, edge21),
            "tna_credit_min": 0.0,
            "tna_rates_min": max(0.0, rates_min),
            "tna_risk_max": 0.60,
            "tna_leadership_min": 0.02,
        }
        name = (
            f"014B_target{iwm_target:.2f}_take{take_max:.2f}_e{edge63:.2f}-{edge21:.2f}_"
            f"w{wins63}-{wins21}_rate{rates_min:.2f}_tna{tna_frac:.2f}_cap{tna_cap:.2f}"
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
            "delta_common_ann_vs_013f": common_m["ann"] - base_common["ann"],
            "delta_common_sharpe_vs_013f": common_m["sharpe"] - base_common["sharpe"],
            "delta_common_dd_vs_013f": common_m["dd"] - base_common["dd"],
            "delta_early_sharpe_vs_013f": early_m["sharpe"] - base_early["sharpe"],
            "delta_recent_sharpe_vs_013f": recent_m["sharpe"] - base_recent["sharpe"],
            "beats_013f_return": common_m["ann"] > base_common["ann"],
            "beats_013f_sharpe": common_m["sharpe"] > base_common["sharpe"],
            "no_worse_dd": common_m["dd"] >= base_common["dd"] - 1e-9,
            "selection_score": (
                (common_m["ann"] - base_common["ann"]) * 1.5
                + (common_m["sharpe"] - base_common["sharpe"]) * 0.7
                + (early_m["sharpe"] - base_early["sharpe"]) * 0.4
                + min(0.0, common_m["dd"] - base_common["dd"]) * 1.25
            ),
            "iwm_overlay_rebalances": int(log["iwm_overlay_on"].sum()),
            "tna_overlay_rebalances": int(log["tna_overlay_on"].sum()),
            "iwm_avg_weight": float(log["exec_w_IWM"].mean()),
            "tna_avg_weight": float(log["exec_w_TNA"].mean()),
            "tna_max_weight": float(log["exec_w_TNA"].max()),
            **params,
        })
        rows.append(row)
        logs_by_name[name] = log
        returns_by_name[name] = returns

    results = pd.DataFrame(rows).sort_values(
        ["beats_013f_return", "beats_013f_sharpe", "no_worse_dd", "selection_score", "common_sharpe"],
        ascending=[False, False, False, False, False],
    )
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)
    best_name = results.iloc[0]["model"]
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])
    cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "delta_common_ann_vs_013f", "delta_common_sharpe_vs_013f", "delta_common_dd_vs_013f",
        "early_sharpe", "delta_early_sharpe_vs_013f", "recent_sharpe", "delta_recent_sharpe_vs_013f",
        "iwm_overlay_rebalances", "tna_overlay_rebalances", "iwm_avg_weight", "tna_avg_weight", "tna_max_weight",
    ]
    print("BASE_013F_COMMON", base_common)
    print("BASE_013F_EARLY", base_early)
    print("BASE_013F_RECENT", base_recent)
    print(results[cols].head(30).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
