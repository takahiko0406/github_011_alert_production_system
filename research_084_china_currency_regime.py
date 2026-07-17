#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = "2014-01-01"
FX_CANDIDATES = {
    "USDCNY": ["CNY=X"],
    "USDCNH": ["CNH=X", "USDCNH=X"],
}
CONFIRMATION = {
    "CQQQ": "CQQQ",
    "XSOE": "XSOE",
    "MCHI": "MCHI",
    "FXI": "FXI",
    "COPPER": "HG=F",
    "DXY": "DX-Y.NYB",
}
OUT_LATEST = "research_084_china_currency_regime_latest.csv"
OUT_HISTORY = "research_084_china_currency_regime_history.csv"
OUT_REPORT = "research_084_china_currency_regime_report.txt"


def download_close(tickers: list[str], start: str, attempts: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                tickers=list(dict.fromkeys(tickers)),
                start=start,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )
            if raw.empty:
                raise RuntimeError("Yahoo Finance returned no rows")
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:
                close = raw[["Close"]].copy()
                close.columns = tickers[:1]
            if isinstance(close, pd.Series):
                close = close.to_frame()
            close.index = pd.to_datetime(close.index)
            return close.sort_index()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(4 * attempt)
    raise RuntimeError(f"Price download failed: {last_error}")


def choose_series(close: pd.DataFrame) -> dict[str, str | None]:
    chosen: dict[str, str | None] = {}
    for name, candidates in FX_CANDIDATES.items():
        chosen[name] = next(
            (
                ticker
                for ticker in candidates
                if ticker in close.columns and close[ticker].dropna().shape[0] >= 100
            ),
            None,
        )
    return chosen


def ret(series: pd.Series, days: int) -> pd.Series:
    return series.pct_change(days, fill_method=None)


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    minimum = max(60, window // 3)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std()
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-3.0, 3.0)


def ann_vol(series: pd.Series, days: int = 63) -> pd.Series:
    return series.pct_change(fill_method=None).rolling(days).std() * math.sqrt(252)


def ma_gap(series: pd.Series, fast: int = 21, slow: int = 63) -> pd.Series:
    return series.rolling(fast).mean() / series.rolling(slow).mean() - 1.0


def regime_label(score: float) -> str:
    if score >= 0.75:
        return "STRONG YUAN / SUPPORTIVE"
    if score >= 0.20:
        return "MODERATELY SUPPORTIVE"
    if score <= -0.75:
        return "YUAN STRESS / DEFENSIVE"
    if score <= -0.20:
        return "MODERATELY NEGATIVE"
    return "NEUTRAL"


def support_label(score: float) -> str:
    if score >= 0.50:
        return "POSITIVE"
    if score <= -0.50:
        return "NEGATIVE"
    return "NEUTRAL"


def pressure_label(score: float) -> str:
    if score >= 1.00:
        return "HIGH"
    if score >= 0.35:
        return "ELEVATED"
    if score <= -0.50:
        return "LOW"
    return "NORMAL"


def fmt(name: str, value: float, percent: bool = False) -> str:
    if pd.isna(value):
        return f"{name}: N/A"
    return f"{name}: {value * 100:.2f}%" if percent else f"{name}: {value:.2f}"


def build_features(close: pd.DataFrame, fx_map: dict[str, str | None]) -> pd.DataFrame:
    if fx_map["USDCNY"] is None:
        raise ValueError("USD/CNY data is unavailable")

    px = close.ffill()
    out = pd.DataFrame(index=px.index)

    for logical, ticker in fx_map.items():
        name = logical.lower()
        if ticker is None:
            out[name] = np.nan
            for horizon in (5, 21, 63, 126):
                out[f"{name}_{horizon}d"] = np.nan
            out[f"{name}_vol_63d"] = np.nan
            out[f"{name}_ma_gap"] = np.nan
            out[f"{name}_z_63d"] = np.nan
            continue

        out[name] = px[ticker]
        for horizon in (5, 21, 63, 126):
            out[f"{name}_{horizon}d"] = ret(px[ticker], horizon)
        out[f"{name}_vol_63d"] = ann_vol(px[ticker])
        out[f"{name}_ma_gap"] = ma_gap(px[ticker])
        out[f"{name}_z_63d"] = zscore(out[f"{name}_63d"])

    if fx_map["USDCNH"] is not None:
        out["cnh_cny_spread"] = out["usdcnh"] / out["usdcny"] - 1.0
        out["cnh_cny_spread_z"] = zscore(out["cnh_cny_spread"])
    else:
        out["cnh_cny_spread"] = np.nan
        out["cnh_cny_spread_z"] = np.nan

    for logical, ticker in CONFIRMATION.items():
        if ticker in px.columns:
            out[f"{logical.lower()}_21d"] = ret(px[ticker], 21)
        else:
            out[f"{logical.lower()}_21d"] = np.nan

    trend_component = pd.concat(
        [
            -out["usdcny_z_63d"],
            -out["usdcnh_z_63d"],
            -zscore(out["usdcny_ma_gap"]),
            -zscore(out["usdcnh_ma_gap"]),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    spread_component = -out["cnh_cny_spread_z"]
    dxy_component = -zscore(out["dxy_21d"])

    # Combine only components that actually have data.
    # CNH and the CNH-CNY spread are optional because Yahoo may return
    # only sparse CNH history. Missing components are excluded and the
    # remaining weights are renormalized row by row.
    component_frame = pd.DataFrame(
        {
            "trend": trend_component,
            "spread": spread_component,
            "dxy": dxy_component,
        }
    )
    component_weights = pd.Series(
        {
            "trend": 0.55,
            "spread": 0.25,
            "dxy": 0.20,
        }
    )

    weighted_values = component_frame.mul(component_weights, axis=1)
    available_weights = component_frame.notna().mul(component_weights, axis=1).sum(axis=1)

    out["yuan_support_score"] = (
        weighted_values.sum(axis=1, min_count=1)
        / available_weights.replace(0.0, np.nan)
    )
    out["capital_flow_pressure_score"] = -out["yuan_support_score"]

    china_confirmation = pd.concat(
        [
            zscore(out["cqqq_21d"]),
            zscore(out["mchi_21d"]),
            zscore(out["fxi_21d"]),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)

    out["china_technology_support_score"] = (
        0.65 * out["yuan_support_score"] + 0.35 * china_confirmation
    )
    out["em_support_score"] = (
        0.70 * out["yuan_support_score"] + 0.30 * zscore(out["xsoe_21d"])
    )
    out["commodity_support_score"] = (
        0.55 * out["yuan_support_score"] + 0.45 * zscore(out["copper_21d"])
    )

    checks = pd.DataFrame(
        {
            "cny_strengthening": out["usdcny_21d"] < 0,
            "cnh_strengthening": out["usdcnh_21d"] < 0,
            "offshore_spread_orderly": out["cnh_cny_spread_z"].abs() < 1.0,
            "cqqq_positive": out["cqqq_21d"] > 0,
            "mchi_positive": out["mchi_21d"] > 0,
            "xsoe_positive": out["xsoe_21d"] > 0,
            "copper_positive": out["copper_21d"] > 0,
        }
    )
    out["confirmation_count"] = checks.sum(axis=1)
    out["available_confirmation_count"] = checks.notna().sum(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--start", default=START_DATE)
    args = parser.parse_args()
    root = args.root.resolve()

    tickers = [t for values in FX_CANDIDATES.values() for t in values]
    tickers += list(CONFIRMATION.values())

    close = download_close(tickers, args.start)
    fx_map = choose_series(close)
    features = build_features(close, fx_map)
    usable = features.dropna(subset=["yuan_support_score"])
    if usable.empty:
        raise ValueError("No usable China currency regime rows were produced")

    latest_date = usable.index[-1]
    latest = usable.iloc[-1]
    score = float(latest["yuan_support_score"])
    pressure = float(latest["capital_flow_pressure_score"])

    latest_row = {
        "data_date": latest_date.date().isoformat(),
        "module": "RESEARCH_084_CHINA_CURRENCY_REGIME",
        "production_authority": "RESEARCH_ONLY",
        "regime_label": regime_label(score),
        "yuan_support_score": score,
        "capital_flow_pressure_label": pressure_label(pressure),
        "capital_flow_pressure_score": pressure,
        "china_technology_support": support_label(float(latest["china_technology_support_score"])),
        "china_technology_support_score": latest["china_technology_support_score"],
        "em_support": support_label(float(latest["em_support_score"])),
        "em_support_score": latest["em_support_score"],
        "commodity_support": support_label(float(latest["commodity_support_score"])),
        "commodity_support_score": latest["commodity_support_score"],
        "usdcny_5d": latest["usdcny_5d"],
        "usdcny_21d": latest["usdcny_21d"],
        "usdcny_63d": latest["usdcny_63d"],
        "usdcny_126d": latest["usdcny_126d"],
        "usdcnh_5d": latest["usdcnh_5d"],
        "usdcnh_21d": latest["usdcnh_21d"],
        "usdcnh_63d": latest["usdcnh_63d"],
        "usdcnh_126d": latest["usdcnh_126d"],
        "cnh_cny_spread": latest["cnh_cny_spread"],
        "cnh_cny_spread_z": latest["cnh_cny_spread_z"],
        "confirmation_count": int(latest["confirmation_count"]),
        "available_confirmation_count": int(latest["available_confirmation_count"]),
        "selected_usdcny_ticker": fx_map["USDCNY"] or "UNAVAILABLE",
        "selected_usdcnh_ticker": fx_map["USDCNH"] or "UNAVAILABLE",
    }
    pd.DataFrame([latest_row]).to_csv(root / OUT_LATEST, index=False)

    history_cols = [
        "usdcny_5d", "usdcny_21d", "usdcny_63d", "usdcny_126d",
        "usdcny_vol_63d", "usdcny_ma_gap", "usdcny_z_63d",
        "usdcnh_5d", "usdcnh_21d", "usdcnh_63d", "usdcnh_126d",
        "usdcnh_vol_63d", "usdcnh_ma_gap", "usdcnh_z_63d",
        "cnh_cny_spread", "cnh_cny_spread_z",
        "yuan_support_score", "capital_flow_pressure_score",
        "china_technology_support_score", "em_support_score",
        "commodity_support_score", "confirmation_count",
        "available_confirmation_count",
    ]
    history = usable[history_cols].copy()
    history.index.name = "date"
    history.to_csv(root / OUT_HISTORY)

    lines = [
        "RESEARCH 084 — CHINA CURRENCY REGIME",
        f"Data date: {latest_date.date().isoformat()}",
        "Authority: RESEARCH ONLY — production model unchanged",
        "",
        f"Overall regime: {regime_label(score)}",
        fmt("Yuan support score", score),
        f"Capital-flow pressure: {pressure_label(pressure)} ({pressure:.2f})",
        "",
        "USD/CNY",
        fmt("5D", latest["usdcny_5d"], True),
        fmt("21D", latest["usdcny_21d"], True),
        fmt("63D", latest["usdcny_63d"], True),
        fmt("126D", latest["usdcny_126d"], True),
        fmt("63D annualized volatility", latest["usdcny_vol_63d"], True),
        fmt("21D/63D moving-average gap", latest["usdcny_ma_gap"], True),
        "",
        "USD/CNH",
        fmt("5D", latest["usdcnh_5d"], True),
        fmt("21D", latest["usdcnh_21d"], True),
        fmt("63D", latest["usdcnh_63d"], True),
        fmt("126D", latest["usdcnh_126d"], True),
        fmt("63D annualized volatility", latest["usdcnh_vol_63d"], True),
        fmt("21D/63D moving-average gap", latest["usdcnh_ma_gap"], True),
        "",
        "ONSHORE / OFFSHORE SPREAD",
        fmt("CNH-CNY spread", latest["cnh_cny_spread"], True),
        fmt("Spread z-score", latest["cnh_cny_spread_z"]),
        "",
        "MARKET SUPPORT",
        f"China technology: {support_label(float(latest['china_technology_support_score']))} ({latest['china_technology_support_score']:.2f})",
        f"Emerging markets: {support_label(float(latest['em_support_score']))} ({latest['em_support_score']:.2f})",
        f"Commodities: {support_label(float(latest['commodity_support_score']))} ({latest['commodity_support_score']:.2f})",
        f"Confirmations: {int(latest['confirmation_count'])}/{int(latest['available_confirmation_count'])}",
        "",
        "INTERPRETATION RULES",
        "- USD/CNY rising means the onshore yuan is weakening.",
        "- USD/CNH rising means the offshore yuan is weakening.",
        "- A widening positive CNH-CNY spread indicates offshore stress.",
        "- Yuan strength usually supports China equities and broad EM.",
        "- Yuan weakness can tighten financial conditions and pressure commodities.",
        "- This module cannot alter verified 034 production allocation.",
    ]
    (root / OUT_REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {OUT_LATEST}")
    print(f"Saved {OUT_HISTORY}")
    print(f"Saved {OUT_REPORT}")
    print("RESEARCH ONLY: production allocation and model logic unchanged.")


if __name__ == "__main__":
    main()
