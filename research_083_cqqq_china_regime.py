#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

PRICE_ASSETS = {
    "CQQQ": "CQQQ",
    "XSOE": "XSOE",
    "QQQM": "QQQM",
    "SOXX": "SOXX",
    "MCHI": "MCHI",
    "KWEB": "KWEB",
    "FXI": "FXI",
    "SPY": "SPY",
}

FX_CANDIDATES = {
    "USDCNY": ["CNY=X"],
    "USDCNH": ["CNH=X", "USDCNH=X"],
    "USDKRW": ["KRW=X"],
}

START_DATE = "2014-01-01"
OUTPUT_LATEST = "research_083_cqqq_china_regime_latest.csv"
OUTPUT_HISTORY = "research_083_cqqq_china_regime_history.csv"
OUTPUT_REPORT = "research_083_cqqq_china_regime_report.txt"


def download_close(tickers: Iterable[str], start: str, attempts: int = 3) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )
            if raw.empty:
                raise RuntimeError("Yahoo Finance returned no rows")
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" not in raw.columns.get_level_values(0):
                    raise RuntimeError("Downloaded data does not contain Close prices")
                close = raw["Close"].copy()
            else:
                if "Close" not in raw.columns:
                    raise RuntimeError("Downloaded data does not contain Close prices")
                close = raw[["Close"]].copy()
                close.columns = tickers[:1]
            if isinstance(close, pd.Series):
                close = close.to_frame()
            close.index = pd.to_datetime(close.index)
            return close.sort_index()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(4 * attempt)
    raise RuntimeError(f"Price download failed after {attempts} attempts: {last_error}")


def choose_available_fx(close: pd.DataFrame) -> dict[str, str | None]:
    selected: dict[str, str | None] = {}
    for name, candidates in FX_CANDIDATES.items():
        chosen = None
        for ticker in candidates:
            if ticker in close.columns and close[ticker].dropna().shape[0] >= 100:
                chosen = ticker
                break
        selected[name] = chosen
    return selected


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    minimum = max(60, window // 3)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std()
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-3.0, 3.0)


def safe_return(prices: pd.Series, days: int) -> pd.Series:
    return prices.pct_change(days, fill_method=None)


def trailing_drawdown(prices: pd.Series, days: int) -> pd.Series:
    rolling_peak = prices.rolling(days, min_periods=max(5, days // 3)).max()
    return prices / rolling_peak - 1.0


def annualized_volatility(prices: pd.Series, days: int = 63) -> pd.Series:
    return prices.pct_change(fill_method=None).rolling(days).std() * math.sqrt(252)


def label_from_score(score: float | None) -> str:
    if score is None or not math.isfinite(score):
        return "UNAVAILABLE"
    if score >= 0.75:
        return "STRONG"
    if score >= 0.20:
        return "IMPROVING"
    if score <= -0.75:
        return "WEAK"
    if score <= -0.20:
        return "DETERIORATING"
    return "NEUTRAL"


def build_features(close: pd.DataFrame, fx_map: dict[str, str | None]) -> pd.DataFrame:
    required = ["CQQQ", "XSOE", "QQQM", "SOXX", "SPY"]
    missing = [t for t in required if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise ValueError(f"Missing required price histories: {missing}")

    px = close.ffill()
    out = pd.DataFrame(index=px.index)

    out["cqqq_ret_5d"] = safe_return(px["CQQQ"], 5)
    out["cqqq_ret_21d"] = safe_return(px["CQQQ"], 21)
    out["cqqq_ret_63d"] = safe_return(px["CQQQ"], 63)
    out["cqqq_ret_126d"] = safe_return(px["CQQQ"], 126)
    out["cqqq_drawdown_63d"] = trailing_drawdown(px["CQQQ"], 63)
    out["cqqq_vol_63d"] = annualized_volatility(px["CQQQ"], 63)

    out["cqqq_rel_xsoe_21d"] = safe_return(px["CQQQ"] / px["XSOE"], 21)
    out["cqqq_rel_qqqm_21d"] = safe_return(px["CQQQ"] / px["QQQM"], 21)
    out["cqqq_rel_soxx_21d"] = safe_return(px["CQQQ"] / px["SOXX"], 21)
    out["cqqq_rel_spy_63d"] = safe_return(px["CQQQ"] / px["SPY"], 63)

    china_candidates = [t for t in ["CQQQ", "MCHI", "KWEB", "FXI"] if t in px.columns]
    breadth_inputs = {t: safe_return(px[t], 21) > 0 for t in china_candidates}
    out["china_proxy_breadth_21d"] = pd.DataFrame(breadth_inputs).mean(axis=1)

    for logical_name, ticker in fx_map.items():
        col = logical_name.lower()
        if ticker is None:
            out[f"{col}_21d"] = np.nan
            out[f"{col}_63d"] = np.nan
        else:
            out[f"{col}_21d"] = safe_return(px[ticker], 21)
            out[f"{col}_63d"] = safe_return(px[ticker], 63)

    out["z_cqqq_momentum"] = rolling_zscore(out["cqqq_ret_63d"])
    out["z_cqqq_rel_xsoe"] = rolling_zscore(out["cqqq_rel_xsoe_21d"])
    out["z_cqqq_rel_qqqm"] = rolling_zscore(out["cqqq_rel_qqqm_21d"])
    out["z_china_breadth"] = rolling_zscore(out["china_proxy_breadth_21d"])
    out["z_usdcny_headwind"] = -rolling_zscore(out["usdcny_63d"])
    out["z_usdcnh_headwind"] = -rolling_zscore(out["usdcnh_63d"])
    out["z_krw_confirmation"] = -rolling_zscore(out["usdkrw_63d"])

    fx_component = out[
        ["z_usdcny_headwind", "z_usdcnh_headwind", "z_krw_confirmation"]
    ].mean(axis=1, skipna=True)

    out["china_tech_regime_score"] = (
        0.30 * out["z_cqqq_momentum"]
        + 0.20 * out["z_cqqq_rel_xsoe"]
        + 0.15 * out["z_cqqq_rel_qqqm"]
        + 0.15 * out["z_china_breadth"]
        + 0.20 * fx_component
    )

    confirmation_frame = pd.DataFrame({
        "positive_21d": out["cqqq_ret_21d"] > 0,
        "beats_xsoe": out["cqqq_rel_xsoe_21d"] > 0,
        "beats_qqqm": out["cqqq_rel_qqqm_21d"] > 0,
        "breadth_majority": out["china_proxy_breadth_21d"] >= 0.50,
        "yuan_support": out[["usdcny_21d", "usdcnh_21d"]].mean(axis=1, skipna=True) < 0,
        "krw_support": out["usdkrw_21d"] < 0,
    })
    out["confirmation_count"] = confirmation_frame.sum(axis=1)
    out["available_confirmation_count"] = confirmation_frame.notna().sum(axis=1)
    return out


def report_value(name: str, value: float | None, percent: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return f"{name}: N/A"
    return f"{name}: {value * 100:.2f}%" if percent else f"{name}: {value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--start", default=START_DATE)
    args = parser.parse_args()
    root = args.root.resolve()

    tickers = list(PRICE_ASSETS.values()) + [
        ticker for candidates in FX_CANDIDATES.values() for ticker in candidates
    ]
    close = download_close(tickers, args.start)
    fx_map = choose_available_fx(close)
    close = close.rename(columns={ticker: logical for logical, ticker in PRICE_ASSETS.items()})

    features = build_features(close, fx_map)
    usable = features.dropna(subset=["china_tech_regime_score"])
    if usable.empty:
        raise ValueError("No usable CQQQ regime rows were produced")

    latest_date = usable.index[-1]
    latest = usable.iloc[-1]
    score = float(latest["china_tech_regime_score"])
    regime = label_from_score(score)

    latest_row = {
        "data_date": latest_date.date().isoformat(),
        "module": "RESEARCH_083_CQQQ_CHINA_TECH_REGIME",
        "production_authority": "RESEARCH_ONLY",
        "regime_label": regime,
        "china_tech_regime_score": score,
        "cqqq_ret_5d": latest["cqqq_ret_5d"],
        "cqqq_ret_21d": latest["cqqq_ret_21d"],
        "cqqq_ret_63d": latest["cqqq_ret_63d"],
        "cqqq_rel_xsoe_21d": latest["cqqq_rel_xsoe_21d"],
        "cqqq_rel_qqqm_21d": latest["cqqq_rel_qqqm_21d"],
        "cqqq_rel_soxx_21d": latest["cqqq_rel_soxx_21d"],
        "china_proxy_breadth_21d": latest["china_proxy_breadth_21d"],
        "usdcny_21d": latest["usdcny_21d"],
        "usdcnh_21d": latest["usdcnh_21d"],
        "usdkrw_21d": latest["usdkrw_21d"],
        "confirmation_count": int(latest["confirmation_count"]),
        "available_confirmation_count": int(latest["available_confirmation_count"]),
        "selected_usdcny_ticker": fx_map["USDCNY"] or "UNAVAILABLE",
        "selected_usdcnh_ticker": fx_map["USDCNH"] or "UNAVAILABLE",
        "selected_usdkrw_ticker": fx_map["USDKRW"] or "UNAVAILABLE",
    }
    pd.DataFrame([latest_row]).to_csv(root / OUTPUT_LATEST, index=False)

    history_cols = [
        "cqqq_ret_5d", "cqqq_ret_21d", "cqqq_ret_63d",
        "cqqq_rel_xsoe_21d", "cqqq_rel_qqqm_21d", "cqqq_rel_soxx_21d",
        "china_proxy_breadth_21d", "usdcny_21d", "usdcnh_21d",
        "usdkrw_21d", "china_tech_regime_score",
        "confirmation_count", "available_confirmation_count",
    ]
    history = usable[history_cols].copy()
    history.index.name = "date"
    history.to_csv(root / OUTPUT_HISTORY)

    lines = [
        "RESEARCH 083 — CQQQ / CHINA TECHNOLOGY REGIME",
        f"Data date: {latest_date.date().isoformat()}",
        "Authority: RESEARCH ONLY — production model unchanged",
        "",
        f"Regime: {regime}",
        report_value("Composite score", score),
        report_value("CQQQ 5D", latest["cqqq_ret_5d"], percent=True),
        report_value("CQQQ 21D", latest["cqqq_ret_21d"], percent=True),
        report_value("CQQQ 63D", latest["cqqq_ret_63d"], percent=True),
        report_value("CQQQ vs XSOE 21D", latest["cqqq_rel_xsoe_21d"], percent=True),
        report_value("CQQQ vs QQQM 21D", latest["cqqq_rel_qqqm_21d"], percent=True),
        report_value("CQQQ vs SOXX 21D", latest["cqqq_rel_soxx_21d"], percent=True),
        report_value("China proxy breadth 21D", latest["china_proxy_breadth_21d"], percent=True),
        "",
        "CURRENCY CONFIRMATION",
        report_value("USD/CNY 21D", latest["usdcny_21d"], percent=True),
        report_value("USD/CNH 21D", latest["usdcnh_21d"], percent=True),
        report_value("USD/KRW 21D", latest["usdkrw_21d"], percent=True),
        f"Confirmations: {int(latest['confirmation_count'])}/{int(latest['available_confirmation_count'])}",
        "",
        "INTERPRETATION RULES",
        "- Falling USD/CNY or USD/CNH supports the yuan and China risk assets.",
        "- Falling USD/KRW supports the Korean won and Asian export cycle.",
        "- CQQQ strength versus XSOE separates China-tech leadership from broad EM.",
        "- CQQQ strength versus QQQM tests whether China tech is leading US growth.",
        "- This module reports evidence only and cannot alter verified 034 allocation.",
    ]
    (root / OUTPUT_REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {OUTPUT_LATEST}")
    print(f"Saved {OUTPUT_HISTORY}")
    print(f"Saved {OUTPUT_REPORT}")
    print("RESEARCH ONLY: no production model, allocation, gate, or dashboard authority changed.")


if __name__ == "__main__":
    main()
