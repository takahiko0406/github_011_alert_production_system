import itertools

import numpy as np
import pandas as pd
import yfinance as yf

import model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST as m


OUT_PREFIX = "model_c_plus_013C_soxx_soxl_maturity"
EXTRA_ASSETS = ["SOXX", "SOXL"]
EXEC_ASSETS_EXT = m.EXEC_ASSETS + EXTRA_ASSETS


def normalize(weights):
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {asset: weight / total for asset, weight in weights.items()}


def download_extended_prices(start_date):
    assets = sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS))
    prices = yf.download(assets, start=start_date, auto_adjust=True, progress=False)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"]
    prices = prices.sort_index().dropna(how="all").ffill()
    missing = [asset for asset in assets if asset not in prices.columns]
    if missing:
        raise ValueError(f"Missing prices: {missing}")
    return prices


def add_soxx_market_filters(rebalance, prices):
    d = rebalance.copy()
    soxx = prices["SOXX"]
    qqqm = prices["QQQM"]
    d["soxx_63d_return"] = [soxx.pct_change(63).reindex([dt]).iloc[0] if dt in soxx.index else np.nan for dt in d["aligned_date"]]
    d["qqqm_63d_return"] = [qqqm.pct_change(63).reindex([dt]).iloc[0] if dt in qqqm.index else np.nan for dt in d["aligned_date"]]
    d["soxx_ma63_ratio"] = [
        soxx.reindex([dt]).iloc[0] / soxx.rolling(63).mean().reindex([dt]).iloc[0] - 1.0
        if dt in soxx.index else np.nan
        for dt in d["aligned_date"]
    ]
    return d.fillna({"soxx_63d_return": 0.0, "qqqm_63d_return": 0.0, "soxx_ma63_ratio": 0.0})


def overlay_allowed(row, params):
    growth = m.safe_float(row, "growth_strength", 0.0)
    soxx = m.safe_float(row, "soxx_strength", 0.0)
    risk = m.safe_float(row, "risk_off_strength", 0.0)
    crash = m.safe_float(row, "crash_pressure", 0.0)
    breakdown = m.safe_float(row, "breakdown_score", 0.0)
    leadership = soxx - growth

    if soxx < params["soxx_min"]:
        return False
    if growth < params["growth_min"]:
        return False
    if risk > params["risk_max"] or crash > params["crash_max"] or breakdown > params["breakdown_max"]:
        return False
    if leadership < params["lead_min"]:
        return False
    if params["trend_filter"]:
        if m.safe_float(row, "soxx_63d_return", 0.0) <= m.safe_float(row, "qqqm_63d_return", 0.0):
            return False
        if m.safe_float(row, "soxx_ma63_ratio", 0.0) <= params["ma63_min"]:
            return False
    return True


def apply_soxx_overlay(weights, row, params):
    weights = dict(weights)
    if not overlay_allowed(row, params):
        return weights, False

    qqqm_move = weights.get("QQQM", 0.0) * params["qqqm_to_soxx"]
    if qqqm_move > 0:
        weights["QQQM"] -= qqqm_move
        weights["SOXX"] = weights.get("SOXX", 0.0) + qqqm_move

    tqqq_move = weights.get("TQQQ", 0.0) * params["tqqq_to_soxl"]
    if tqqq_move > 0:
        room = max(0.0, params["soxl_cap"] - weights.get("SOXL", 0.0))
        move = min(tqqq_move, room)
        weights["TQQQ"] -= move
        weights["SOXL"] = weights.get("SOXL", 0.0) + move

    return normalize(weights), True


def run_overlay_backtest(rebalance, prices, cfg, params):
    asset_returns = prices[sorted(set(m.PRICE_ASSETS + EXTRA_ASSETS))].pct_change().fillna(0.0)
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

        weights, overlay_on = apply_soxx_overlay(weights, row, params)
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

        log = {"date": date, "turnover": turnover, "soxx_overlay_on": overlay_on, **details}
        for asset in EXEC_ASSETS_EXT:
            log[f"exec_w_{asset}"] = weights.get(asset, 0.0)
        logs.append(log)
        old_weights = weights.copy()

    return portfolio.dropna(), pd.DataFrame(logs), float(np.mean(turnovers)) if turnovers else np.nan


def main():
    cfg, cfg_message = m.load_best_cfg()
    print(cfg_message)

    rebalance_raw = m.load_rebalance_log()
    start = (rebalance_raw["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    prices = download_extended_prices(start)
    rebalance = m.align_rebalance_dates(rebalance_raw, prices[m.PRICE_ASSETS])
    rebalance = m.add_transition_features(rebalance)
    rebalance = add_soxx_market_filters(rebalance, prices)
    current_returns = m.load_current_returns_optional()

    base_returns, base_log, base_turnover = m.run_backtest(rebalance, prices[m.PRICE_ASSETS], cfg, m.DEFAULT_TRANSACTION_COST)
    common = current_returns.index.intersection(base_returns.index)
    base_common = base_returns.loc[common]
    base_ann = m.annualized_return(base_common)
    base_sharpe = m.sharpe_ratio(base_common)
    base_dd = m.max_drawdown(base_common)

    param_grid = []
    for lead_min, qfrac, tfrac, cap, trend_filter in itertools.product(
        [0.20, 0.25, 0.30, 0.35],
        [0.00, 0.30, 0.60, 0.75],
        [0.00, 0.25, 0.50, 0.65],
        [0.15, 0.20, 0.25, 0.30],
        [False, True],
    ):
        if qfrac == 0.0 and tfrac == 0.0:
            continue
        param_grid.append({
            "soxx_min": 0.0,
            "growth_min": 0.0,
            "risk_max": 0.75,
            "crash_max": 1.25,
            "breakdown_max": 2.0,
            "lead_min": lead_min,
            "qqqm_to_soxx": qfrac,
            "tqqq_to_soxl": tfrac,
            "soxl_cap": cap,
            "trend_filter": trend_filter,
            "ma63_min": 0.0,
        })

    rows = []
    logs_by_name = {}
    returns_by_name = {}

    baseline_row = m.summary("011_LIGHT_BASELINE", base_returns, base_turnover, len(base_log), m.DEFAULT_TRANSACTION_COST)
    baseline_row.update({
        "common_ann_return": base_ann,
        "common_sharpe": base_sharpe,
        "common_max_drawdown": base_dd,
        "dominance_score": 0.0,
        "beats_return": False,
        "beats_sharpe": False,
        "no_worse_dd": True,
        "soxx_overlay_rebalances": 0,
        "soxx_avg_weight": 0.0,
        "soxl_avg_weight": 0.0,
        "soxl_max_weight": 0.0,
        "trend_filter": False,
        "qqqm_to_soxx": 0.0,
        "tqqq_to_soxl": 0.0,
        "soxl_cap": 0.0,
        "lead_min": np.nan,
    })
    rows.append(baseline_row)
    logs_by_name["011_LIGHT_BASELINE"] = base_log
    returns_by_name["011_LIGHT_BASELINE"] = base_returns

    for params in param_grid:
        name = (
            f"013C_lead{params['lead_min']:.2f}_q{params['qqqm_to_soxx']:.2f}_"
            f"t{params['tqqq_to_soxl']:.2f}_cap{params['soxl_cap']:.2f}_"
            f"trend{int(params['trend_filter'])}"
        )
        returns, log, avg_turnover = run_overlay_backtest(rebalance, prices, cfg, params)
        common_idx = current_returns.index.intersection(returns.index)
        common_returns = returns.loc[common_idx]
        common_ann = m.annualized_return(common_returns)
        common_sharpe = m.sharpe_ratio(common_returns)
        common_dd = m.max_drawdown(common_returns)
        dd_delta = common_dd - base_dd

        row = m.summary(name, returns, avg_turnover, len(log), m.DEFAULT_TRANSACTION_COST)
        row.update({
            "common_ann_return": common_ann,
            "common_sharpe": common_sharpe,
            "common_max_drawdown": common_dd,
            "beats_return": common_ann > base_ann,
            "beats_sharpe": common_sharpe > base_sharpe,
            "no_worse_dd": common_dd >= base_dd - 1e-9,
            "dd_delta_vs_base": dd_delta,
            "dominance_score": (
                (common_ann - base_ann) * 2.0
                + (common_sharpe - base_sharpe) * 0.45
                + min(0.0, dd_delta) * 1.25
            ),
            "soxx_overlay_rebalances": int(log["soxx_overlay_on"].sum()),
            "soxx_avg_weight": float(log["exec_w_SOXX"].mean()),
            "soxl_avg_weight": float(log["exec_w_SOXL"].mean()),
            "soxl_max_weight": float(log["exec_w_SOXL"].max()),
            **params,
        })
        rows.append(row)
        logs_by_name[name] = log
        returns_by_name[name] = returns

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["no_worse_dd", "beats_return", "beats_sharpe", "dominance_score", "common_sharpe"],
        ascending=[False, False, False, False, False],
    )
    results.to_csv(f"{OUT_PREFIX}_results.csv", index=False)

    best_name = results.iloc[0]["model"]
    logs_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_rebalance_log.csv", index=False)
    returns_by_name[best_name].to_csv(f"{OUT_PREFIX}_best_daily_returns.csv", header=["portfolio_return"])

    subperiods = {
        "full_common": (common.min(), common.max()),
        "early_2023_2024": (pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")),
        "recent_2025_2026": (pd.Timestamp("2025-01-01"), common.max()),
    }
    sub_rows = []
    best_returns = returns_by_name[best_name]
    for label, (start_dt, end_dt) in subperiods.items():
        for model_name, model_returns in [("baseline", base_returns), ("best_013C", best_returns)]:
            r = model_returns.loc[(model_returns.index >= start_dt) & (model_returns.index <= end_dt)]
            if len(r) > 20:
                row = m.summary(f"{model_name}_{label}", r)
                sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(f"{OUT_PREFIX}_best_subperiods.csv", index=False)

    display_cols = [
        "model", "common_ann_return", "common_sharpe", "common_max_drawdown",
        "no_worse_dd", "dominance_score", "soxx_overlay_rebalances",
        "soxx_avg_weight", "soxl_avg_weight", "soxl_max_weight",
        "lead_min", "qqqm_to_soxx", "tqqq_to_soxl", "soxl_cap", "trend_filter",
    ]
    print("BASE_COMMON", base_ann, base_sharpe, base_dd)
    print(results[display_cols].head(30).to_string(index=False))
    print("\nSUBPERIODS")
    print(pd.DataFrame(sub_rows).to_string(index=False))
    print(f"\nSaved {OUT_PREFIX}_results.csv")
    print("Best:", best_name)


if __name__ == "__main__":
    main()
