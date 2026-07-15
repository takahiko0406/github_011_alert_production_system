import itertools

import numpy as np
import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m
import research_013c_soxx_soxl_maturity as r13c
import research_013e_soxl_ladder as r13e
import research_013h_final_soxx_soxl_validation as r13h
import research_014b_iwm_peer_leadership_tna as r14b
import research_015a_xlv_defensive_overlay as r15a


OUT_PREFIX = "model_c_plus_015B_fast_defensive_breadth"
DEF_ASSETS = ["XLV", "XLU", "XLP", "XLRE"]
RISK_ASSETS = ["QQQM", "SOXX", "IWM", "XSOE"]
EXTRA_ASSETS = ["SOXX", "SOXL", "IWM", "TNA", "XLF", "SHY", "IEF"] + DEF_ASSETS
EXEC_ASSETS_EXT = m.EXEC_ASSETS + ["SOXX", "SOXL", "IWM", "TNA"] + DEF_ASSETS


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


def add_fast_defensive_features(rebalance, prices):
    d = rebalance.copy()
    for window in [5, 10, 21]:
        for asset in DEF_ASSETS + RISK_ASSETS:
            d[f"{asset.lower()}_ret_{window}"] = [
                prices[asset].pct_change(window).reindex([dt]).iloc[0] if dt in prices.index else np.nan
                for dt in d["aligned_date"]
            ]
        d[f"def_avg_ret_{window}"] = d[[f"{a.lower()}_ret_{window}" for a in DEF_ASSETS]].mean(axis=1)
        d[f"risk_avg_ret_{window}"] = d[[f"{a.lower()}_ret_{window}" for a in RISK_ASSETS]].mean(axis=1)
        d[f"def_edge_{window}"] = d[f"def_avg_ret_{window}"] - d[f"risk_avg_ret_{window}"]
        d[f"def_win_count_{window}"] = sum((d[f"{a.lower()}_ret_{window}"] > 0.0).astype(int) for a in DEF_ASSETS)
        d[f"risk_loss_count_{window}"] = sum((d[f"{a.lower()}_ret_{window}"] < 0.0).astype(int) for a in RISK_ASSETS)
        d[f"best_def_asset_{window}"] = d[[f"{a.lower()}_ret_{window}" for a in DEF_ASSETS]].idxmax(axis=1).str.replace("_ret_" + str(window), "", regex=False).str.upper()
    return d.fillna(0.0)


def normalize(weights):
    total = sum(weights.values())
    return {asset: weight / total for asset, weight in weights.items()} if total > 0 else weights


def apply_fast_defense(weights, row, params):
    weights = dict(weights)
    risk = m.safe_float(row, "risk_off_strength", 0.0)
    crash = m.safe_float(row, "crash_pressure", 0.0)
    def_edge_5 = m.safe_float(row, "def_edge_5", 0.0)
    def_edge_10 = m.safe_float(row, "def_edge_10", 0.0)
    def_edge_21 = m.safe_float(row, "def_edge_21", 0.0)
    def_wins_5 = m.safe_float(row, "def_win_count_5", 0.0)
    risk_losses_5 = m.safe_float(row, "risk_loss_count_5", 0.0)
    risk_losses_10 = m.safe_float(row, "risk_loss_count_10", 0.0)
    qqqm_5 = m.safe_float(row, "qqqm_ret_5", 0.0)
    soxx_5 = m.safe_float(row, "soxx_ret_5", 0.0)

    defense_score = 0.45 * def_edge_5 + 0.25 * def_edge_10 + 0.15 * def_edge_21 + 0.15 * risk
    defense_on = (
        risk >= params["risk_min"]
        and risk <= params["risk_max"]
        and crash <= params["crash_max"]
        and def_edge_5 >= params["edge_5_min"]
        and def_edge_10 >= params["edge_10_min"]
        and def_wins_5 >= params["def_wins_5_min"]
        and risk_losses_5 >= params["risk_losses_5_min"]
        and risk_losses_10 >= params["risk_losses_10_min"]
        and (qqqm_5 <= params["qqqm_5_max"] or soxx_5 <= params["soxx_5_max"])
        and defense_score >= params["score_min"]
    )
    if not defense_on:
        return weights, False, "", defense_score

    target_asset = row.get("best_def_asset_5", "XLV")
    if target_asset not in DEF_ASSETS:
        target_asset = "XLV"
    target_weight = params["def_target"]
    needed = max(0.0, target_weight - weights.get(target_asset, 0.0))
    for source in params["sources"]:
        if needed <= 1e-12:
            break
        source_w = weights.get(source, 0.0)
        if source_w <= 0:
            continue
        take = min(source_w * params["source_take_max"], needed)
        weights[source] -= take
        weights[target_asset] = weights.get(target_asset, 0.0) + take
        needed -= take
    return normalize(weights), True, target_asset, defense_score


def run_backtest(rebalance, prices, cfg, params):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS + r14b.PEERS))
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
        weights, iwm_on, tna_on, iwm_score = r14b.apply_iwm_tna(weights, row, r15a.PARAMS_014D)
        weights, defense_on, defense_asset, defense_score = apply_fast_defense(weights, row, params)

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
            "defense_overlay_on": defense_on,
            "defense_asset": defense_asset,
            "defense_score": defense_score,
            "iwm_leadership_score": iwm_score,
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
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    prices = download_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = r13c.add_soxx_market_filters(rebalance, prices)
    rebalance = r14b.add_iwm_peer_features(rebalance, prices)
    rebalance = add_fast_defensive_features(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    baseline_returns, baseline_log, baseline_turn = r14b.run_backtest(rebalance, prices, cfg, r15a.PARAMS_014D)
    common = current_returns.index.intersection(baseline_returns.index)
    common_start, common_end = common.min(), common.max()
    early_start, early_end = pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")
    recent_start, recent_end = pd.Timestamp("2025-01-01"), common_end
    base_common = period_metrics(baseline_returns, common_start, common_end)
    base_early = period_metrics(baseline_returns, early_start, early_end)
    base_recent = period_metrics(baseline_returns, recent_start, recent_end)

    source_sets = {
        "growth_risk": ["SOXL", "TQQQ", "SOXX", "QQQM", "IWM"],
        "all_risk": ["SOXL", "TQQQ", "SOXX", "QQQM", "IWM", "XSOE"],
        "cyc_growth": ["SOXL", "TQQQ", "SOXX", "QQQM", "XLI", "XLB", "XLE"],
    }
    rows = []
    logs_by_name = {}
    returns_by_name = {}
    for target, take_max, risk_min, risk_max, edge5, edge10, wins5, losses5, losses10, source_key in itertools.product(
        [0.05, 0.10],
        [0.50],
        [0.20],
        [1.25],
        [0.02, 0.03],
        [0.00],
        [3],
        [2, 3],
        [1],
        ["growth_risk", "cyc_growth"],
    ):
        params = {
            "def_target": target,
            "source_take_max": take_max,
            "risk_min": risk_min,
            "risk_max": risk_max,
            "crash_max": 1.50,
            "edge_5_min": edge5,
            "edge_10_min": edge10,
            "def_wins_5_min": wins5,
            "risk_losses_5_min": losses5,
            "risk_losses_10_min": losses10,
            "qqqm_5_max": 0.00,
            "soxx_5_max": 0.00,
            "score_min": 0.00,
            "sources": source_sets[source_key],
            "source_key": source_key,
        }
        name = (
            f"015B_def{target:.2f}_take{take_max:.2f}_risk{risk_min:.2f}-{risk_max:.2f}_"
            f"e{edge5:.2f}-{edge10:.2f}_w{wins5}_l{losses5}-{losses10}_{source_key}"
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
            "defense_overlay_rebalances": int(log["defense_overlay_on"].sum()),
            "xlv_avg_weight": float(log["exec_w_XLV"].mean()),
            "xlu_avg_weight": float(log["exec_w_XLU"].mean()),
            "xlp_avg_weight": float(log["exec_w_XLP"].mean()),
            "xlre_avg_weight": float(log["exec_w_XLRE"].mean()),
            "def_max_weight": float(log[[f"exec_w_{a}" for a in DEF_ASSETS]].sum(axis=1).max()),
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
        "early_sharpe", "recent_sharpe", "defense_overlay_rebalances",
        "xlv_avg_weight", "xlu_avg_weight", "xlp_avg_weight", "xlre_avg_weight", "def_max_weight",
    ]
    print("BASE_014D_COMMON", base_common)
    print(results[cols].head(30).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
