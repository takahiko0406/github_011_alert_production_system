"""Append-only completed-prediction monitor for Dashboard Page 5.

Only forecasts that were actually emitted by a production run are admitted.
Historical bootstrap reads committed dashboard snapshots from git; it never
re-fits a model or regenerates a past forecast. Outcomes remain pending until
ten later trading sessions are observable.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


PREDICTION_HISTORY = "production_prediction_history.csv"
PREDICTION_ERRORS = "production_prediction_errors.csv"
PERFORMANCE_HISTORY = "production_model_performance_history.csv"
REPORT = "production_page5_report.md"

LATEST = "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv"
RANKING = "model_c_plus_034_live_dashboard_ranking.csv"
EXPANDED = "model_c_plus_expanded_execution_candidate_latest_recommendation.csv"
SCORES = "model_c_plus_full_universe_expected_returns_trading_scores.csv"

PREDICTION_ASSETS = [
    "QQQM", "SOXX", "IWM", "FEZ", "XLE", "XLB", "XLI", "XLF",
    "XLV", "XLP", "XLU", "XLRE", "TLT", "IEF", "GLD", "XSOE",
]
FOCUS_ASSETS = ["XSOE", "QQQM", "SOXX", "XLE", "XLB", "XLI"]
LIVE_MODEL_ASSETS = {"QQQM", "XLE", "XSOE", "XLI", "XLB"}
OVERLAY_ASSETS = {"SOXX", "IWM", "FEZ", "XLV", "XLP", "XLU", "XLRE", "TLT", "GLD"}
LEVERAGE_SOURCE = {"TQQQ": "QQQM", "SOXL": "SOXX", "TNA": "IWM", "ERX": "XLE", "UXI": "XLI"}
EXECUTION_ASSETS = [
    "BIL", "ERX", "FEZ", "GLD", "IEF", "IWM", "QQQM", "SOXL", "SOXX", "TLT",
    "TNA", "TQQQ", "UXI", "XLB", "XLE", "XLF", "XLI", "XLP", "XLRE", "XLU", "XLV", "XSOE",
]
MARKET_TICKERS = sorted(set(PREDICTION_ASSETS + EXECUTION_ASSETS + ["SPY", "^VIX"]))
PERIODS_PER_YEAR = 25.2


PREDICTION_COLUMNS = [
    "source_commit", "captured_at_utc", "record_origin", "signal_date", "execution_date", "maturity_date",
    "outcome_status", "ETF", "allocation_weights", "final_portfolio_weight", "expected_portfolio_return",
    "expected_etf_return", "strength", "model_score", "authority", "source_model", "macro_regime",
    "growth_strength", "usd_strength", "vix_level", "risk_off_strength", "realized_etf_return",
    "realized_portfolio_return", "benchmark", "benchmark_return", "prediction_error", "absolute_error",
    "squared_error", "direction_correct", "prediction_rank", "realized_rank", "realized_etf_alpha",
    "realized_portfolio_alpha",
]


def finite(value, default=np.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def authority(asset: str, ranking_authority: str = "") -> str:
    if asset in LIVE_MODEL_ASSETS:
        return "LIVE_EXECUTION_MODEL"
    if asset in OVERLAY_ASSETS:
        return "LIVE_OVERLAY_OR_GATED_CANDIDATE"
    return "RESEARCH_CANDIDATE" if not ranking_authority else str(ranking_authority).upper()


def git(*args: str, root: Path, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=check, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout


def csv_at_commit(root: Path, commit: str, filename: str) -> pd.DataFrame:
    text = git("show", f"{commit}:{filename}", root=root, check=False)
    return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()


def actual_production_snapshots(root: Path) -> list[dict]:
    snapshots: dict[pd.Timestamp, dict] = {}
    ledger_path = root / PREDICTION_HISTORY
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path, parse_dates=["signal_date"])
        required = {"source_commit", "captured_at_utc", "record_origin", "signal_date", "ETF", "allocation_weights", "expected_etf_return"}
        if required.issubset(existing.columns) and set(existing.record_origin).issubset({"ACTUAL_COMMITTED_PRODUCTION_SNAPSHOT", "CURRENT_PRODUCTION_RUN"}):
            for signal_date, group in existing.groupby("signal_date"):
                first = group.iloc[0]
                weight_map = json.loads(first.allocation_weights)
                forecasts = group.set_index("ETF").expected_etf_return.to_dict()
                if set(forecasts) != set(PREDICTION_ASSETS):
                    continue
                snapshots[pd.Timestamp(signal_date)] = {
                    "source_commit": str(first.source_commit), "captured_at_utc": str(first.captured_at_utc),
                    "record_origin": str(first.record_origin), "signal_date": pd.Timestamp(signal_date),
                    "weights": {asset: float(weight_map.get(asset, 0.0)) for asset in EXECUTION_ASSETS},
                    "allocation_weights": json.dumps(weight_map, sort_keys=True),
                    "expected_portfolio_return": finite(first.expected_portfolio_return), "forecasts": forecasts,
                    "strength": group.set_index("ETF").strength.to_dict(), "model_score": group.set_index("ETF").model_score.to_dict(),
                    "authority": group.set_index("ETF").authority.to_dict(), "source_model": str(first.source_model),
                    "macro_regime": str(first.macro_regime), "growth_strength": finite(first.growth_strength),
                    "usd_strength": finite(first.usd_strength), "vix_level": finite(first.vix_level),
                    "risk_off_strength": finite(first.risk_off_strength),
                }
    log = git("log", "--format=%H|%cI", "--", LATEST, root=root)
    commits = [line.split("|", 1) for line in reversed(log.splitlines()) if "|" in line]
    for commit, captured in commits:
        latest = csv_at_commit(root, commit, LATEST)
        ranking = csv_at_commit(root, commit, RANKING)
        expanded = csv_at_commit(root, commit, EXPANDED)
        if latest.empty or ranking.empty or expanded.empty:
            continue
        snapshot = snapshot_from_frames(commit, captured, "ACTUAL_COMMITTED_PRODUCTION_SNAPSHOT", latest, ranking, expanded)
        snapshots[snapshot["signal_date"]] = snapshot

    latest = pd.read_csv(root / LATEST)
    scores = pd.read_csv(root / SCORES)
    scores = scores[scores["signal_date"].astype(str).str[:10].eq(str(latest.iloc[-1]["signal_date"])[:10])]
    ranking = pd.DataFrame({
        "ETF": scores["asset"],
        "Live Score": scores["tradable_score_0_100"],
        "Authority": scores["asset"].map(lambda asset: "RESEARCH" if asset in {"XLF", "IEF"} else "LIVE"),
    })
    expanded = pd.read_csv(root / EXPANDED)
    head = git("rev-parse", "HEAD", root=root).strip()
    capture_time = str(latest.iloc[-1].get("latest_data_date", latest.iloc[-1].get("signal_date")))[:10] + "T21:00:00Z"
    current = snapshot_from_frames(head, capture_time, "CURRENT_PRODUCTION_RUN", latest, ranking, expanded)
    snapshots[current["signal_date"]] = current
    return [snapshots[key] for key in sorted(snapshots)]


def snapshot_from_frames(commit: str, captured: str, origin: str, latest_df: pd.DataFrame, ranking_df: pd.DataFrame, expanded_df: pd.DataFrame) -> dict:
    latest = latest_df.iloc[-1]
    expanded = expanded_df.iloc[-1]
    signal_date = pd.Timestamp(str(latest["signal_date"])[:10])
    weights = {asset: finite(latest.get(f"exec_w_{asset}"), 0.0) for asset in EXECUTION_ASSETS}
    if abs(sum(weights.values()) - 1.0) > 1e-8 or any(value < -1e-12 for value in weights.values()):
        raise RuntimeError(f"Invalid committed production weights at {signal_date.date()}")
    rank = ranking_df.drop_duplicates("ETF", keep="first").set_index("ETF")
    forecasts = {asset: finite(expanded.get(f"adj_pred_{asset}")) for asset in PREDICTION_ASSETS}
    expected_portfolio = sum(weight * finite(forecasts.get(LEVERAGE_SOURCE.get(asset, asset)), 0.0) for asset, weight in weights.items())
    growth, usd = finite(expanded.get("growth_strength")), finite(expanded.get("usd_3m_strength"))
    vix, risk_off = finite(expanded.get("vix_level")), finite(expanded.get("risk_off_strength"))
    regime = " | ".join([
        "Strong USD" if usd >= 0 else "Weak USD" if np.isfinite(usd) else "USD Unavailable",
        "High VIX" if vix >= 20 else "Low VIX" if np.isfinite(vix) else "VIX Unavailable",
        "Strong Growth" if growth >= 0 else "Weak Growth" if np.isfinite(growth) else "Growth Unavailable",
        "High Risk-Off" if risk_off > 0 else "Low Risk-Off" if np.isfinite(risk_off) else "Risk-Off Unavailable",
    ])
    return {
        "source_commit": commit, "captured_at_utc": captured, "record_origin": origin,
        "signal_date": signal_date, "weights": weights, "allocation_weights": json.dumps({k: v for k, v in weights.items() if v > 1e-12}, sort_keys=True),
        "expected_portfolio_return": expected_portfolio, "forecasts": forecasts,
        "strength": {asset: finite(rank.at[asset, "Live Score"]) if asset in rank.index else np.nan for asset in PREDICTION_ASSETS},
        "model_score": {asset: finite(expanded.get(f"raw_pred_{asset}")) for asset in PREDICTION_ASSETS},
        "authority": {asset: authority(asset, rank.at[asset, "Authority"] if asset in rank.index else "") for asset in PREDICTION_ASSETS},
        "source_model": str(latest.get("source_model", "UNKNOWN")), "macro_regime": regime,
        "growth_strength": growth, "usd_strength": usd, "vix_level": vix, "risk_off_strength": risk_off,
    }


def download_prices(start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        MARKET_TICKERS, start=start.strftime("%Y-%m-%d"), auto_adjust=True, progress=False,
        group_by="column", threads=False, timeout=60,
    )
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError("Page 5 market data download is empty or malformed")
    close, opening = data["Close"].copy(), data["Open"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    opening.index = pd.to_datetime(opening.index).tz_localize(None).normalize()
    missing = [ticker for ticker in MARKET_TICKERS if ticker not in close or close[ticker].dropna().empty]
    if missing:
        raise RuntimeError(f"Page 5 market data missing: {missing}")
    return close.sort_index(), opening.sort_index()


def prediction_ledger(root: Path) -> pd.DataFrame:
    snapshots = actual_production_snapshots(root)
    close, opening = download_prices(min(item["signal_date"] for item in snapshots) - pd.Timedelta(days=5))
    calendar = close["SPY"].dropna().index
    rows: list[dict] = []
    for snapshot in snapshots:
        signal = snapshot["signal_date"]
        future = calendar[calendar > signal]
        execution = future[0] if len(future) else pd.NaT
        maturity = future[9] if len(future) >= 10 else pd.NaT
        complete = pd.notna(maturity) and maturity <= calendar.max()
        predicted_rank = pd.Series(snapshot["forecasts"]).rank(ascending=False, method="average")
        realized = {}
        if complete:
            for asset in PREDICTION_ASSETS:
                if signal not in close.index or maturity not in close.index or pd.isna(close.at[signal, asset]) or pd.isna(close.at[maturity, asset]):
                    complete = False
                    break
                realized[asset] = float(close.at[maturity, asset] / close.at[signal, asset] - 1.0)
        realized_rank = pd.Series(realized).rank(ascending=False, method="average") if complete else pd.Series(dtype=float)
        portfolio_return = benchmark_return = np.nan
        if complete:
            portfolio_return = 0.0
            for asset, weight in snapshot["weights"].items():
                if weight <= 0:
                    continue
                if execution not in opening.index or maturity not in close.index or pd.isna(opening.at[execution, asset]) or pd.isna(close.at[maturity, asset]):
                    complete = False
                    break
                portfolio_return += weight * float(close.at[maturity, asset] / opening.at[execution, asset] - 1.0)
            if complete:
                benchmark_return = float(close.at[maturity, "SPY"] / opening.at[execution, "SPY"] - 1.0)
        for asset in PREDICTION_ASSETS:
            prediction = snapshot["forecasts"][asset]
            actual = realized.get(asset, np.nan) if complete else np.nan
            error = prediction - actual if complete else np.nan
            rows.append({
                "source_commit": snapshot["source_commit"], "captured_at_utc": snapshot["captured_at_utc"], "record_origin": snapshot["record_origin"],
                "signal_date": signal, "execution_date": execution, "maturity_date": maturity, "outcome_status": "COMPLETED" if complete else "PENDING",
                "ETF": asset, "allocation_weights": snapshot["allocation_weights"], "final_portfolio_weight": snapshot["weights"].get(asset, 0.0),
                "expected_portfolio_return": snapshot["expected_portfolio_return"], "expected_etf_return": prediction,
                "strength": snapshot["strength"][asset], "model_score": snapshot["model_score"][asset], "authority": snapshot["authority"][asset],
                "source_model": snapshot["source_model"], "macro_regime": snapshot["macro_regime"], "growth_strength": snapshot["growth_strength"],
                "usd_strength": snapshot["usd_strength"], "vix_level": snapshot["vix_level"], "risk_off_strength": snapshot["risk_off_strength"],
                "realized_etf_return": actual, "realized_portfolio_return": portfolio_return if complete else np.nan,
                "benchmark": "SPY", "benchmark_return": benchmark_return if complete else np.nan,
                "prediction_error": error, "absolute_error": abs(error) if complete else np.nan, "squared_error": error * error if complete else np.nan,
                "direction_correct": bool(np.sign(prediction) == np.sign(actual)) if complete else pd.NA,
                "prediction_rank": predicted_rank[asset], "realized_rank": realized_rank.get(asset, np.nan),
                "realized_etf_alpha": actual - benchmark_return if complete else np.nan,
                "realized_portfolio_alpha": portfolio_return - benchmark_return if complete else np.nan,
            })
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(["signal_date", "ETF"]).reset_index(drop=True)


def calibration_error(frame: pd.DataFrame) -> float:
    usable = frame[["expected_etf_return", "realized_etf_return"]].dropna().copy()
    if len(usable) < 10:
        return np.nan
    bins = min(5, usable.expected_etf_return.nunique())
    usable["bucket"] = pd.qcut(usable.expected_etf_return.rank(method="first"), bins, labels=False)
    grouped = usable.groupby("bucket", observed=True).agg(predicted=("expected_etf_return", "mean"), realized=("realized_etf_return", "mean"))
    return float((grouped.predicted - grouped.realized).abs().mean())


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else np.nan


def trend_label(current: float, prior: float, threshold: float, absolute_improvement: bool = False) -> str:
    if not np.isfinite(current) or not np.isfinite(prior):
        return "→ Stable"
    delta = abs(prior) - abs(current) if absolute_improvement else current - prior
    return "↑ Improving" if delta > threshold else "↓ Deteriorating" if delta < -threshold else "→ Stable"


def allocation_metrics(ledger: pd.DataFrame) -> pd.DataFrame:
    signals = ledger.drop_duplicates("signal_date").sort_values("signal_date").copy()
    parsed = [json.loads(value) for value in signals.allocation_weights]
    turnover, persistence, hhi = [], [], []
    prior = None
    episodes: list[int] = []
    active = {asset: 0 for asset in EXECUTION_ASSETS}
    holding = []
    for weights in parsed:
        turnover.append(0.0 if prior is None else 0.5 * sum(abs(weights.get(a, 0.0) - prior.get(a, 0.0)) for a in EXECUTION_ASSETS))
        persistence.append(1.0 if prior is None else sum(min(weights.get(a, 0.0), prior.get(a, 0.0)) for a in EXECUTION_ASSETS))
        hhi.append(sum(float(value) ** 2 for value in weights.values()))
        for asset in EXECUTION_ASSETS:
            if weights.get(asset, 0.0) > 1e-10:
                active[asset] += 1
            elif active[asset]:
                episodes.append(active[asset]); active[asset] = 0
        current_episodes = episodes + [value for value in active.values() if value]
        holding.append(float(np.mean(current_episodes)) if current_episodes else 0.0)
        prior = weights
    signals["allocation_turnover"] = turnover
    signals["allocation_persistence"] = persistence
    signals["concentration_hhi"] = hhi
    signals["average_holding_period_sessions"] = holding
    return signals.set_index("signal_date")


def performance_history(ledger: pd.DataFrame) -> pd.DataFrame:
    completed = ledger[ledger.outcome_status.eq("COMPLETED")].copy()
    alloc = allocation_metrics(ledger)
    rows = []
    completed_dates = sorted(completed.signal_date.unique())
    for index, signal in enumerate(completed_dates):
        current = completed[completed.signal_date.eq(signal)]
        dates20 = completed_dates[max(0, index - 19): index + 1]
        dates60 = completed_dates[max(0, index - 59): index + 1]
        e20, e60 = completed[completed.signal_date.isin(dates20)], completed[completed.signal_date.isin(dates60)]
        by_signal = completed.drop_duplicates("signal_date").set_index("signal_date")
        r = by_signal.loc[dates20, "realized_portfolio_return"]
        b = by_signal.loc[dates20, "benchmark_return"]
        vol = r.std(ddof=1); downside = np.sqrt(np.mean(np.square(np.minimum(r, 0.0))))
        ic20 = e20.groupby("signal_date").apply(lambda x: x.prediction_rank.corr(x.realized_rank, method="spearman"), include_groups=False)
        ic60 = e60.groupby("signal_date").apply(lambda x: x.prediction_rank.corr(x.realized_rank, method="spearman"), include_groups=False)
        predicted_top = current.loc[current.prediction_rank.idxmin(), "ETF"]
        realized_top = current.loc[current.realized_rank.idxmin(), "ETF"]
        top3 = set(current.nsmallest(3, "realized_rank").ETF)
        hit_series = []
        hit3_series = []
        for date in dates20:
            group = completed[completed.signal_date.eq(date)]
            top = group.loc[group.prediction_rank.idxmin(), "ETF"]
            hit_series.append(top == group.loc[group.realized_rank.idxmin(), "ETF"])
            hit3_series.append(top in set(group.nsmallest(3, "realized_rank").ETF))
        allocation = alloc.loc[signal]
        rows.append({
            "signal_date": signal, "execution_date": current.execution_date.iloc[0], "maturity_date": current.maturity_date.iloc[0],
            "monitor_status": "COMPLETED", "completed_signal_count": index + 1, "completed_prediction_count": len(e20),
            "source_model": current.source_model.iloc[0], "expected_portfolio_return": current.expected_portfolio_return.iloc[0],
            "realized_portfolio_return": current.realized_portfolio_return.iloc[0], "benchmark_return": current.benchmark_return.iloc[0],
            "realized_alpha": current.realized_portfolio_alpha.iloc[0],
            "rank_ic": current.prediction_rank.corr(current.realized_rank, method="spearman"),
            "rolling_rank_ic_20": ic20.mean(), "rolling_rank_ic_60": ic60.mean(),
            "rolling_directional_accuracy_20": e20.direction_correct.astype(bool).mean(),
            "prediction_mae_20": e20.absolute_error.mean(), "prediction_rmse_20": np.sqrt(e20.squared_error.mean()),
            "mean_prediction_error_20": e20.prediction_error.mean(), "calibration_error_20": calibration_error(e20),
            "rolling_portfolio_return_20": (1 + r).prod() - 1,
            "rolling_sharpe_20": r.mean() / vol * np.sqrt(PERIODS_PER_YEAR) if vol > 0 else np.nan,
            "rolling_sortino_20": r.mean() / downside * np.sqrt(PERIODS_PER_YEAR) if downside > 0 else np.nan,
            "rolling_maximum_drawdown_20": max_drawdown(r), "rolling_alpha_20": (1 + r).prod() - (1 + b).prod(),
            "win_rate_20": (r > 0).mean(), "top_etf_hit": predicted_top == realized_top, "top_3_etf_hit": predicted_top in top3,
            "top_etf_hit_rate_20": np.mean(hit_series), "top_3_etf_hit_rate_20": np.mean(hit3_series),
            "allocation_turnover": allocation.allocation_turnover, "allocation_persistence": allocation.allocation_persistence,
            "average_holding_period_sessions": allocation.average_holding_period_sessions,
            "concentration_hhi": allocation.concentration_hhi, "growth_strength": current.growth_strength.iloc[0],
            "usd_strength": current.usd_strength.iloc[0], "vix_level": current.vix_level.iloc[0], "risk_off_strength": current.risk_off_strength.iloc[0],
        })
    performance = pd.DataFrame(rows)
    if performance.empty:
        latest = ledger.sort_values("signal_date").iloc[-1]
        empty = {column: np.nan for column in [
            "expected_portfolio_return", "realized_portfolio_return", "benchmark_return", "realized_alpha", "rank_ic",
            "rolling_rank_ic_20", "rolling_rank_ic_60", "rolling_directional_accuracy_20", "prediction_mae_20", "prediction_rmse_20",
            "mean_prediction_error_20", "calibration_error_20", "rolling_portfolio_return_20", "rolling_sharpe_20", "rolling_sortino_20",
            "rolling_maximum_drawdown_20", "rolling_alpha_20", "win_rate_20", "top_etf_hit", "top_3_etf_hit",
            "top_etf_hit_rate_20", "top_3_etf_hit_rate_20",
        ]}
        allocation = alloc.loc[latest.signal_date]
        performance = pd.DataFrame([{**empty, "signal_date": latest.signal_date, "execution_date": latest.execution_date,
            "maturity_date": latest.maturity_date, "monitor_status": "INSUFFICIENT_COMPLETED_HISTORY", "completed_signal_count": 0,
            "completed_prediction_count": 0, "source_model": latest.source_model,
            "allocation_turnover": allocation.allocation_turnover, "allocation_persistence": allocation.allocation_persistence,
            "average_holding_period_sessions": allocation.average_holding_period_sessions, "concentration_hhi": allocation.concentration_hhi,
            "growth_strength": latest.growth_strength, "usd_strength": latest.usd_strength, "vix_level": latest.vix_level,
            "risk_off_strength": latest.risk_off_strength,
        }])
    present = set(pd.to_datetime(performance.signal_date))
    pending_rows = []
    for signal in alloc.index:
        if signal in present:
            continue
        source = ledger[ledger.signal_date.eq(signal)].iloc[0]
        prior_rows = performance[pd.to_datetime(performance.signal_date) < signal]
        base = (prior_rows.iloc[-1] if not prior_rows.empty else performance.iloc[0]).to_dict()
        allocation = alloc.loc[signal]
        base.update({
            "signal_date": signal, "execution_date": source.execution_date, "maturity_date": source.maturity_date,
            "monitor_status": "PENDING", "completed_signal_count": int(ledger[(ledger.outcome_status.eq("COMPLETED")) & (ledger.signal_date <= signal)].signal_date.nunique()),
            "source_model": source.source_model, "expected_portfolio_return": source.expected_portfolio_return,
            "realized_portfolio_return": np.nan, "benchmark_return": np.nan, "realized_alpha": np.nan,
            "rank_ic": np.nan, "top_etf_hit": np.nan, "top_3_etf_hit": np.nan,
            "allocation_turnover": allocation.allocation_turnover, "allocation_persistence": allocation.allocation_persistence,
            "average_holding_period_sessions": allocation.average_holding_period_sessions, "concentration_hhi": allocation.concentration_hhi,
            "growth_strength": source.growth_strength, "usd_strength": source.usd_strength,
            "vix_level": source.vix_level, "risk_off_strength": source.risk_off_strength,
        })
        pending_rows.append(base)
    if pending_rows:
        performance = pd.concat([performance, pd.DataFrame(pending_rows)], ignore_index=True).sort_values("signal_date").reset_index(drop=True)
    for index in performance.index:
        current = performance.loc[index]
        prior = performance.loc[index - 20] if index >= 20 else current
        trends = [
            trend_label(current.rolling_rank_ic_20, prior.rolling_rank_ic_20, 0.05),
            trend_label(current.rolling_directional_accuracy_20, prior.rolling_directional_accuracy_20, 0.03),
            trend_label(current.mean_prediction_error_20, prior.mean_prediction_error_20, 0.005, True),
        ]
        deteriorating = sum("Deteriorating" in value for value in trends)
        if current.completed_signal_count < 20:
            health = "WATCH"
        elif current.rolling_rank_ic_20 >= 0.10 and current.rolling_directional_accuracy_20 >= 0.55 and abs(current.mean_prediction_error_20) <= 0.02 and deteriorating == 0:
            health = "HEALTHY"
        elif current.rolling_rank_ic_20 < 0 or current.rolling_directional_accuracy_20 < 0.48 or deteriorating >= 2:
            health = "DETERIORATING"
        else:
            health = "WATCH"
        performance.loc[index, ["rank_ic_trend", "directional_accuracy_trend", "prediction_bias_trend", "overall_health"]] = [*trends, health]
    return performance


def etf_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    completed = ledger[ledger.outcome_status.eq("COMPLETED") & ledger.ETF.isin(FOCUS_ASSETS)]
    if completed.empty:
        return pd.DataFrame({"ETF": FOCUS_ASSETS, "completed_predictions": 0, "average_predicted_return": np.nan,
            "average_realized_return": np.nan, "mean_error": np.nan, "directional_accuracy": np.nan,
            "overpredicted": False, "underpredicted": False})
    result = completed.groupby("ETF", sort=False).agg(completed_predictions=("ETF", "size"), average_predicted_return=("expected_etf_return", "mean"),
        average_realized_return=("realized_etf_return", "mean"), mean_error=("prediction_error", "mean"), directional_accuracy=("direction_correct", "mean")).reindex(FOCUS_ASSETS).reset_index()
    result["overpredicted"] = result.mean_error > 0.005; result["underpredicted"] = result.mean_error < -0.005
    return result


def regime_summary(performance: pd.DataFrame) -> pd.DataFrame:
    completed = performance[performance.monitor_status.eq("COMPLETED")]
    masks = {
        "Strong USD": completed.usd_strength >= 0, "Weak USD": completed.usd_strength < 0,
        "High VIX": completed.vix_level >= 20, "Low VIX": completed.vix_level < 20,
        "Strong Growth": completed.growth_strength >= 0, "Weak Growth": completed.growth_strength < 0,
        "High Risk-Off": completed.risk_off_strength > 0, "Low Risk-Off": completed.risk_off_strength <= 0,
    }
    return pd.DataFrame([{"regime": name, "completed_signals": int(mask.sum()),
        "average_portfolio_return": completed.loc[mask, "realized_portfolio_return"].mean(),
        "average_alpha": completed.loc[mask, "realized_alpha"].mean(), "win_rate": (completed.loc[mask, "realized_portfolio_return"] > 0).mean() if mask.any() else np.nan,
        "average_rank_ic": completed.loc[mask, "rank_ic"].mean()} for name, mask in masks.items()])


def pct(value, digits=1) -> str:
    return "DATA_UNAVAILABLE" if not np.isfinite(finite(value)) else f"{100 * float(value):+.{digits}f}%"


def num(value, digits=3) -> str:
    return "DATA_UNAVAILABLE" if not np.isfinite(finite(value)) else f"{float(value):.{digits}f}"


def completed_metric(latest: pd.Series, value, formatter, minimum=20) -> str:
    """Fail closed when an institutional metric lacks enough matured signals."""
    if int(latest.completed_signal_count) < minimum:
        return "INSUFFICIENT_DATA"
    return formatter(value)


def completed_trend(latest: pd.Series, value, minimum=20) -> str:
    if int(latest.completed_signal_count) < minimum:
        return "INSUFFICIENT_DATA"
    return str(value)


def automatic_summary(latest: pd.Series) -> str:
    if latest.completed_signal_count < 20:
        return f"WATCH: {int(latest.completed_signal_count)} production signals have completed their 10-session outcome; 20 are required before institutional health classification. No backfilled replay observations are used."
    return (f"{latest.overall_health}: Rank IC is {str(latest.rank_ic_trend).lower()}, directional accuracy is "
        f"{str(latest.directional_accuracy_trend).lower()}, and prediction bias is {str(latest.prediction_bias_trend).lower()}. "
        f"The latest 20-signal Rank IC is {latest.rolling_rank_ic_20:.3f}, directional accuracy is {latest.rolling_directional_accuracy_20:.1%}, "
        f"and rolling alpha versus SPY is {latest.rolling_alpha_20:+.1%}.")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns); lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |")
    return "\n".join(lines)


def load_snapshot(root: Path) -> dict:
    ledger = pd.read_csv(root / PREDICTION_HISTORY, parse_dates=["signal_date", "execution_date", "maturity_date"])
    performance = pd.read_csv(root / PERFORMANCE_HISTORY, parse_dates=["signal_date", "execution_date", "maturity_date"])
    latest = performance.iloc[-1]
    completed = ledger[ledger.outcome_status.eq("COMPLETED")]
    return {"ledger": ledger, "performance": performance, "latest": latest, "latest_signal": ledger.signal_date.max(), "etfs": etf_summary(ledger),
        "regimes": regime_summary(performance), "recent": completed.sort_values(["maturity_date", "ETF"]).tail(20),
        "summary": automatic_summary(latest), "pending_signals": ledger[ledger.outcome_status.eq("PENDING")].signal_date.nunique()}


def write_report(root: Path) -> None:
    snapshot = load_snapshot(root); latest = snapshot["latest"]
    recent = snapshot["recent"][["signal_date", "ETF", "expected_etf_return", "realized_etf_return", "prediction_error"]] if not snapshot["recent"].empty else pd.DataFrame(columns=["signal_date", "ETF", "expected_etf_return", "realized_etf_return", "prediction_error"])
    report = f"""# 034 PRODUCTION MODEL PERFORMANCE

This institutional monitor uses only forecasts captured from actual production runs. It is not a backtest, redesign, or validation framework. Historical replay artifacts are explicitly excluded.

## Overall Health

**{latest.overall_health}**

{snapshot['summary']}

- Completed signals: {int(latest.completed_signal_count)}
- Pending signals: {snapshot['pending_signals']}
- Latest captured signal: {str(snapshot['latest_signal'])[:10]}

## Prediction Quality

| Metric | Latest |
|---|---:|
| Rolling Rank IC (20) | {completed_metric(latest, latest.rolling_rank_ic_20, num)} |
| Rolling Rank IC (60) | {completed_metric(latest, latest.rolling_rank_ic_60, num)} |
| Directional Accuracy | {completed_metric(latest, latest.rolling_directional_accuracy_20, pct)} |
| MAE | {completed_metric(latest, latest.prediction_mae_20, pct)} |
| RMSE | {completed_metric(latest, latest.prediction_rmse_20, pct)} |
| Mean Prediction Error | {completed_metric(latest, latest.mean_prediction_error_20, pct)} |
| Calibration Error | {completed_metric(latest, latest.calibration_error_20, pct)} |

## Portfolio and Allocation Quality

| Metric | Latest |
|---|---:|
| Rolling Portfolio Return | {completed_metric(latest, latest.rolling_portfolio_return_20, pct)} |
| Rolling Sharpe | {completed_metric(latest, latest.rolling_sharpe_20, num)} |
| Rolling Sortino | {completed_metric(latest, latest.rolling_sortino_20, num)} |
| Rolling Maximum Drawdown | {completed_metric(latest, latest.rolling_maximum_drawdown_20, pct)} |
| Rolling Alpha vs SPY | {completed_metric(latest, latest.rolling_alpha_20, pct)} |
| Win Rate | {completed_metric(latest, latest.win_rate_20, pct)} |
| Top ETF Hit Rate | {completed_metric(latest, latest.top_etf_hit_rate_20, pct)} |
| Top 3 ETF Hit Rate | {completed_metric(latest, latest.top_3_etf_hit_rate_20, pct)} |
| Allocation Turnover | {num(latest.allocation_turnover)} |
| Allocation Persistence | {pct(latest.allocation_persistence)} |
| Average Holding Period | {num(latest.average_holding_period_sessions,1)} sessions |
| Concentration (HHI) | {num(latest.concentration_hhi)} |

## Drift Detection

- Rank IC: **{completed_trend(latest, latest.rank_ic_trend)}**
- Directional Accuracy: **{completed_trend(latest, latest.directional_accuracy_trend)}**
- Prediction Bias: **{completed_trend(latest, latest.prediction_bias_trend)}**

## ETF Error Analysis

{markdown_table(snapshot['etfs'])}

## Recent Completed Predictions

{markdown_table(recent)}

## Regime Performance

{markdown_table(snapshot['regimes'])}

## Institutional Controls

- A prediction is captured before its outcome is known and keyed by signal date, ETF, and source commit.
- Outcomes mature only after ten later market sessions are available.
- ETF outcomes use the production target convention (signal close to maturity close).
- Portfolio and SPY outcomes use next-session adjusted open through maturity close.
- Health remains WATCH until at least 20 production signals have completed.
- No historical replay, current fitted model, or validation output is used to backfill a past production prediction.
"""
    (root / REPORT).write_text(report, encoding="utf-8")


def dashboard_fragment(root: Path) -> str:
    s = load_snapshot(root); latest = s["latest"]
    def badge(value: str, kind="neutral"): return f'<span class="badge {kind}">{html.escape(str(value))}</span>'
    def table(headers, rows):
        return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{html.escape(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'
    health_kind = {"HEALTHY":"ok","WATCH":"warn","DETERIORATING":"bad"}[latest.overall_health]
    insufficient = lambda value, formatter: completed_metric(latest, value, formatter)
    trend = lambda value: completed_trend(latest, value)
    quality = [["Rank IC 20 / 60", f"{insufficient(latest.rolling_rank_ic_20, num)} / {insufficient(latest.rolling_rank_ic_60, num)}", badge(trend(latest.rank_ic_trend))],
        ["Directional accuracy", insufficient(latest.rolling_directional_accuracy_20, pct), badge(trend(latest.directional_accuracy_trend))], ["MAE / RMSE", f"{insufficient(latest.prediction_mae_20, pct)} / {insufficient(latest.prediction_rmse_20, pct)}", "Completed only"],
        ["Mean error", insufficient(latest.mean_prediction_error_20, pct), badge(trend(latest.prediction_bias_trend))], ["Calibration error", insufficient(latest.calibration_error_20, pct), "Bucket gap"]]
    portfolio = [["Rolling return", insufficient(latest.rolling_portfolio_return_20, pct), "20 completed signals"], ["Sharpe / Sortino", f"{insufficient(latest.rolling_sharpe_20, num)} / {insufficient(latest.rolling_sortino_20, num)}", "10-session outcomes"],
        ["Maximum drawdown", insufficient(latest.rolling_maximum_drawdown_20, pct), "Completed only"], ["Alpha vs SPY", insufficient(latest.rolling_alpha_20, pct), "Next-open matched"], ["Win rate", insufficient(latest.win_rate_20, pct), "Completed only"]]
    allocation = [["Top / top-3 hit", f"{insufficient(latest.top_etf_hit_rate_20, pct)} / {insufficient(latest.top_3_etf_hit_rate_20, pct)}", "Completed only"], ["Turnover", num(latest.allocation_turnover), "Half-L1"],
        ["Persistence", pct(latest.allocation_persistence), "Weight overlap"], ["Holding period", f"{num(latest.average_holding_period_sessions,1)} sessions", "Actual snapshots"], ["HHI", num(latest.concentration_hhi), "Weight concentration"]]
    etfs = [[r.ETF, str(int(r.completed_predictions)), pct(r.average_predicted_return), pct(r.average_realized_return), pct(r.mean_error), pct(r.directional_accuracy), "YES" if r.overpredicted else "NO", "YES" if r.underpredicted else "NO"] for r in s["etfs"].itertuples()]
    recent = [[str(r.signal_date)[:10], r.ETF, pct(r.expected_etf_return), pct(r.realized_etf_return), pct(r.prediction_error)] for r in s["recent"].itertuples()]
    regimes = [[r.regime, str(int(r.completed_signals)), pct(r.average_portfolio_return), pct(r.average_alpha), pct(r.win_rate), num(r.average_rank_ic)] for r in s["regimes"].itertuples()]
    if not recent: recent = [["—","No completed predictions yet","—","—","—"]]
    return f'''<section id="p5" class="page"><h2>034 PRODUCTION MODEL PERFORMANCE</h2><p>Actual captured production forecasts; completed outcomes only. Not a backtest, redesign, or validation framework.</p><div class="card"><div class="kpis"><div class="kpi"><small>Overall health</small><strong>{badge(latest.overall_health,health_kind)}</strong></div><div class="kpi"><small>Completed signals</small><strong>{int(latest.completed_signal_count)}</strong></div><div class="kpi"><small>Pending signals</small><strong>{s['pending_signals']}</strong></div><div class="kpi"><small>Rank IC (20)</small><strong>{insufficient(latest.rolling_rank_ic_20, num)}</strong></div><div class="kpi"><small>Latest signal</small><strong>{str(s['latest_signal'])[:10]}</strong></div></div><p><b>Institutional summary:</b> {html.escape(s['summary'])}</p></div><div class="grid" style="margin-top:14px"><div class="card span4"><h3>Prediction Quality</h3>{table(['Metric','Latest','Trend / scope'],quality)}</div><div class="card span4"><h3>Portfolio Quality</h3>{table(['Metric','Latest','Scope'],portfolio)}</div><div class="card span4"><h3>Allocation Behaviour</h3>{table(['Metric','Latest','Definition'],allocation)}</div><div class="card span12"><h3>ETF Error Analysis</h3>{table(['ETF','N','Avg predicted','Avg realized','Mean error','Directional','Over?','Under?'],etfs)}</div><div class="card span12"><h3>Recent Performance — last 20 completed predictions</h3>{table(['Signal','ETF','Prediction','Realized','Error'],recent)}</div><div class="card span12"><h3>Regime Performance</h3>{table(['Regime','Signals','Avg return','Avg alpha','Win rate','Rank IC'],regimes)}</div></div></section>'''


def telegram_page(root: Path) -> str:
    s = load_snapshot(root); latest = s["latest"]
    lines = ["034 PRODUCTION MODEL PERFORMANCE", f"Latest captured signal: {str(s['latest_signal'])[:10]}", "Completed production outcomes only — not a backtest.", "",
        f"OVERALL HEALTH: {latest.overall_health}", s["summary"], "", f"Completed / pending signals: {int(latest.completed_signal_count)} / {s['pending_signals']}", "",
        "PREDICTION QUALITY", f"Rank IC 20 / 60: {completed_metric(latest, latest.rolling_rank_ic_20, num)} / {completed_metric(latest, latest.rolling_rank_ic_60, num)}", f"Directional accuracy: {completed_metric(latest, latest.rolling_directional_accuracy_20, pct)}",
        f"MAE / RMSE: {completed_metric(latest, latest.prediction_mae_20, pct)} / {completed_metric(latest, latest.prediction_rmse_20, pct)}", f"Bias / calibration: {completed_metric(latest, latest.mean_prediction_error_20, pct)} / {completed_metric(latest, latest.calibration_error_20, pct)}",
        f"Rank IC: {completed_trend(latest, latest.rank_ic_trend)}", f"Direction: {completed_trend(latest, latest.directional_accuracy_trend)}", f"Bias: {completed_trend(latest, latest.prediction_bias_trend)}", "", "PORTFOLIO QUALITY",
        f"Return / alpha: {completed_metric(latest, latest.rolling_portfolio_return_20, pct)} / {completed_metric(latest, latest.rolling_alpha_20, pct)}", f"Sharpe / Sortino: {completed_metric(latest, latest.rolling_sharpe_20, num)} / {completed_metric(latest, latest.rolling_sortino_20, num)}",
        f"Drawdown / win rate: {completed_metric(latest, latest.rolling_maximum_drawdown_20, pct)} / {completed_metric(latest, latest.win_rate_20, pct)}", "", "ALLOCATION BEHAVIOUR",
        f"Top / top-3 hit: {completed_metric(latest, latest.top_etf_hit_rate_20, pct)} / {completed_metric(latest, latest.top_3_etf_hit_rate_20, pct)}", f"Turnover / persistence: {num(latest.allocation_turnover)} / {pct(latest.allocation_persistence)}",
        f"Holding period / HHI: {num(latest.average_holding_period_sessions,1)} sessions / {num(latest.concentration_hhi)}", "", "FOCUS ETF ERRORS"]
    for row in s["etfs"].itertuples():
        lines.append(f"{row.ETF}: n={int(row.completed_predictions)} | pred {pct(row.average_predicted_return)} | real {pct(row.average_realized_return)} | err {pct(row.mean_error)} | dir {pct(row.directional_accuracy)}")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent); args = parser.parse_args(); root = args.root.resolve()
    ledger = prediction_ledger(root)
    errors = ledger[ledger.outcome_status.eq("COMPLETED")][["signal_date","execution_date","maturity_date","ETF","authority","source_model","expected_etf_return","realized_etf_return","prediction_error","absolute_error","squared_error","direction_correct","prediction_rank","realized_rank","realized_etf_alpha","final_portfolio_weight","macro_regime"]].copy()
    performance = performance_history(ledger)
    ledger.to_csv(root / PREDICTION_HISTORY, index=False, float_format="%.17g"); errors.to_csv(root / PREDICTION_ERRORS, index=False, float_format="%.17g"); performance.to_csv(root / PERFORMANCE_HISTORY, index=False, float_format="%.17g")
    write_report(root)
    latest = performance.iloc[-1]
    print(f"Saved forward production monitor: captured={ledger.signal_date.nunique()}, completed={int(latest.completed_signal_count)}, health={latest.overall_health}")


if __name__ == "__main__":
    main()
