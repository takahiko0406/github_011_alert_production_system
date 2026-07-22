import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

print("Current working directory:", os.getcwd())

# ============================================================
# MODEL C+ UPGRADE:
# - Baseline universe: QQQM, XLE, XSOE, BIL with TQQQ overlay
# - Upgraded universe: QQQM, XLE, XSOE, XLI, XLB, BIL with TQQQ overlay
# - Adds industrial/materials regime logic:
#       XLB = copper/materials/early industrial cycle
#       XLI = industrial/reshoring/capex cycle
# - Includes old-vs-new comparison in one run
# - Adds a V2 overlay test:
#       V1 = original XLI/XLB overlay
#       V2 = stricter industrial regime classifier to reduce false signals
# - Adds divergence/crash detector risk layer:
#       detects tech-only rallies and scales down TQQQ before full risk-off
# ============================================================

# ============================================================
# 1. SETTINGS
# ============================================================
BASELINE_SECTOR_ETFS = ["QQQM", "XLE", "XSOE"]
UPGRADED_SECTOR_ETFS = ["QQQM", "XLE", "XSOE", "XLI", "XLB"]
SOXX_012A_SECTOR_ETFS = ["QQQM", "SOXX", "XLE", "XSOE", "XLI", "XLB"]
EXPANDED_012_SECTOR_ETFS = ["QQQM", "SOXX", "XLE", "XLI", "XLB", "XLF", "XLV", "IWM", "XSOE"]
FULL_EXPECTED_RETURN_ETFS = [
    "QQQM", "SOXX", "IWM", "FEZ",
    "XLE", "XLB", "XLI", "XLF", "XLV", "XLP", "XLU", "XLRE",
    "TLT", "IEF", "GLD", "XSOE",
]
run_soxx_012a_test = os.getenv("RUN_SOXX_012A_TEST", "0").strip().lower() in ("1", "true", "yes")
run_soxx_012b_test = os.getenv("RUN_SOXX_012B_TEST", "0").strip().lower() in ("1", "true", "yes")
run_soxx_012c_test = os.getenv("RUN_SOXX_012C_TEST", "0").strip().lower() in ("1", "true", "yes")
run_soxx_012d_test = os.getenv("RUN_SOXX_012D_TEST", "0").strip().lower() in ("1", "true", "yes")
run_soxx_012e_test = os.getenv("RUN_SOXX_012E_TEST", "0").strip().lower() in ("1", "true", "yes")
run_expanded_012_test = os.getenv("RUN_EXPANDED_012_TEST", "0").strip().lower() in ("1", "true", "yes")
run_full_expected_return_study = os.getenv("RUN_FULL_EXPECTED_RETURN_STUDY", "1").strip().lower() in ("1", "true", "yes")

cash_etf = "BIL"
spy_etf = "SPY"
feature_etfs = ["ITA", "SOXX", "HYG"]
execution_extra = ["TQQQ", "ERX", "UXI"]
if (
    run_soxx_012a_test
    or run_soxx_012b_test
    or run_soxx_012c_test
    or run_soxx_012d_test
    or run_soxx_012e_test
    or run_expanded_012_test
    or run_full_expected_return_study
):
    execution_extra.append("SOXL")

# Download everything needed by both models.
all_assets = sorted(set(
    BASELINE_SECTOR_ETFS
    + UPGRADED_SECTOR_ETFS
    + (SOXX_012A_SECTOR_ETFS if (run_soxx_012a_test or run_soxx_012b_test or run_soxx_012c_test or run_soxx_012d_test or run_soxx_012e_test) else [])
    + (EXPANDED_012_SECTOR_ETFS if run_expanded_012_test else [])
    + (FULL_EXPECTED_RETURN_ETFS if run_full_expected_return_study else [])
    + [cash_etf, spy_etf]
    + feature_etfs
    + execution_extra
))

start_date = "2010-01-01"
end_date = None

forward_return_days = 10
rebalance_step = 10
train_window = 252 * 3
transaction_cost = 0.001
risk_free_rate_annual = 0.0

rf_params = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

overlay_scale = 0.002
risk_off_cash_threshold = 0.01
zscore_window = 252

# ============================================================
# CONDITIONAL DIVERGENCE + SOXX BREAKDOWN DEFENSE SETTINGS
# ============================================================
# Key upgrade vs pure divergence defense:
#   Divergence alone is NOT a sell signal. Your grid search proved that the
#   best profile was no defense because tech-only rallies can keep running.
#
#   This version only reduces TQQQ when BOTH are true:
#     1) Tech/real-economy divergence is high
#     2) SOXX/QQQM short-term momentum starts breaking
#
# That makes it an early crash detector, not an anti-momentum rule.
use_conditional_breakdown_defense = False

conditional_breakdown_rules = {
    "watch": {
        "divergence_min": 2.75,
        "breakdown_score_min": 1.0,
        "risk_off_min": -0.25,
        "tqqq_multiplier": 0.90,
        "cash_buffer": 0.00,
    },
    "warning": {
        "divergence_min": 3.00,
        "breakdown_score_min": 2.0,
        "risk_off_min": 0.00,
        "tqqq_multiplier": 0.65,
        "cash_buffer": 0.10,
    },
    "danger": {
        "divergence_min": 3.25,
        "breakdown_score_min": 3.0,
        "risk_off_min": 0.25,
        "tqqq_multiplier": 0.35,
        "cash_buffer": 0.25,
    },
}

# Component thresholds used to calculate breakdown_score.
# Example: SOXX 5-day return below -3% counts as one breakdown point.
breakdown_component_thresholds = {
    "soxx_5d_max": -0.030,
    "soxx_10d_max": -0.045,
    "soxx_dd_21_max": -0.060,
    "qqqm_5d_max": -0.020,
    "qqqm_10d_max": -0.035,
    "qqqm_dd_21_max": -0.045,
}

# TQQQ overlay only applies when QQQM is the top signal asset.
tiered_tqqq_rule = {
    "moderate": {
        "gap_min": 0.003,
        "top_score_min": 0.010,
        "growth_min": 0.30,
        "soxx_min": 0.30,
        "risk_off_max": 1.00,
        "vix_max": 25.0,
        "replace_fraction": 0.50,
    },
    "strong": {
        "gap_min": 0.004,
        "top_score_min": 0.010,
        "growth_min": 0.20,
        "soxx_min": 0.20,
        "risk_off_max": 0.75,
        "vix_max": 22.0,
        "replace_fraction": 1.00,
    },
}

# ============================================================
# 2. PERFORMANCE FUNCTIONS
# ============================================================
def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    cumulative = (1 + returns).prod()
    years = len(returns) / 252
    if years <= 0:
        return np.nan
    return cumulative ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - rf_annual) / ann_vol


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min()


def compute_turnover(old_weights: dict, new_weights: dict, universe: list) -> float:
    old_vec = np.array([old_weights.get(a, 0.0) for a in universe], dtype=float)
    new_vec = np.array([new_weights.get(a, 0.0) for a in universe], dtype=float)
    return np.abs(new_vec - old_vec).sum()


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std.replace(0, np.nan)
    z = z.clip(-3, 3)
    return z.fillna(0.0)


def performance_summary(name: str, returns: pd.Series, avg_turnover: float) -> dict:
    return {
        "model": name,
        "annual_return": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate_annual),
        "max_drawdown": max_drawdown(returns),
        "avg_turnover": avg_turnover,
        "start": returns.dropna().index.min(),
        "end": returns.dropna().index.max(),
        "days": len(returns.dropna()),
    }


def subperiod_diagnostics(model_name: str, returns: pd.Series) -> pd.DataFrame:
    periods = [
        ("FULL", returns.index.min(), returns.index.max()),
        ("AI_RALLY_2023_2024", pd.Timestamp("2023-10-18"), pd.Timestamp("2024-12-31")),
        ("RECENT_2025_2026", pd.Timestamp("2025-01-01"), returns.index.max()),
    ]

    rows = []
    for label, start, end in periods:
        r = returns.loc[(returns.index >= start) & (returns.index <= end)].dropna()
        if len(r) == 0:
            continue
        rows.append({
            "model": model_name,
            "period": label,
            "annual_return": annualized_return(r),
            "volatility": annualized_volatility(r),
            "sharpe": sharpe_ratio(r, risk_free_rate_annual),
            "max_drawdown": max_drawdown(r),
            "start": r.index.min(),
            "end": r.index.max(),
            "days": len(r),
            "final_equity": (1.0 + r).prod(),
        })
    return pd.DataFrame(rows)


def exposure_diagnostics(model_name: str, rebalance_df: pd.DataFrame) -> pd.DataFrame:
    if rebalance_df.empty:
        return pd.DataFrame()

    rows = []
    trade_count = len(rebalance_df)
    for asset in ["SOXX", "SOXL", "QQQM", "TQQQ", "XLE", "ERX", "XSOE", "XLI", "UXI", "XLB", cash_etf]:
        col = f"exec_w_{asset}"
        if col not in rebalance_df.columns:
            continue
        s = rebalance_df[col].fillna(0.0)
        rows.append({
            "model": model_name,
            "asset": asset,
            "avg_weight": float(s.mean()),
            "max_weight": float(s.max()),
            "active_count": int((s > 1e-8).sum()),
            "active_pct": float((s > 1e-8).sum() / trade_count),
        })

    for asset in ["SOXX", "QQQM", "XLE", "XSOE", "XLI", "XLB"]:
        rows.append({
            "model": model_name,
            "asset": f"top_{asset}",
            "avg_weight": np.nan,
            "max_weight": np.nan,
            "active_count": int((rebalance_df["top_asset"] == asset).sum()),
            "active_pct": float((rebalance_df["top_asset"] == asset).sum() / trade_count),
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. DATA DOWNLOAD
# ============================================================
def download_close_data(tickers, start, end=None):
    requested = list(dict.fromkeys(tickers if isinstance(tickers, (list, tuple, set)) else [tickers]))
    data = yf.download(
        requested,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
        timeout=30,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.sort_index()

    missing = [ticker for ticker in requested if ticker not in data.columns or data[ticker].dropna().empty]
    for ticker in missing:
        try:
            single = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
                timeout=20,
            )
            if isinstance(single.columns, pd.MultiIndex):
                single = single["Close"]
            elif "Close" in single.columns:
                single = single["Close"]
            if isinstance(single, pd.DataFrame):
                single = single.squeeze("columns")
            if isinstance(single, pd.Series) and not single.dropna().empty:
                data[ticker] = single.sort_index()
        except Exception as exc:
            print(f"Warning: failed individual download for {ticker}: {exc}")

    return data.reindex(columns=requested).sort_index()


print("Downloading price data...")
prices = download_close_data(all_assets, start_date, end_date).dropna(how="all")
if prices.empty:
    raise ValueError("No price data downloaded.")

print("Downloading macro proxies...")
macro_tickers = {
    "short_rate": "^IRX",
    "long_rate": "^TNX",
    "oil": "CL=F",
    "usd": "DX-Y.NYB",
    "vix_level": "^VIX",
    "copper": "HG=F",
}
macro_raw = download_close_data(list(macro_tickers.values()), start_date, end_date)
macro_raw = macro_raw.rename(columns={v: k for k, v in macro_tickers.items()})
macro_raw = macro_raw.reindex(prices.index).ffill()

required_cols = all_assets
prices = prices.dropna(subset=required_cols)
macro_raw = macro_raw.reindex(prices.index).ffill()

print(f"Latest available data date: {prices.index[-1].date()}")

# ============================================================
# 4. COMMON SERIES / MACRO FEATURES
# ============================================================
asset_returns = prices[all_assets].pct_change()

spy = prices[spy_etf]
spy_ret_1m = spy.pct_change(21)
spy_ret_3m = spy.pct_change(63)
spy_ret_6m = spy.pct_change(126)
spy_vol_1m = spy.pct_change().rolling(21).std() * np.sqrt(252)
spy_vol_3m = spy.pct_change().rolling(63).std() * np.sqrt(252)

short_rate = macro_raw["short_rate"] / 100.0
long_rate = macro_raw["long_rate"] / 100.0
yield_curve = long_rate - short_rate

oil_1m = macro_raw["oil"].pct_change(21)
oil_3m = macro_raw["oil"].pct_change(63)

usd_1m = macro_raw["usd"].pct_change(21)
usd_3m = macro_raw["usd"].pct_change(63)
usd_6m = macro_raw["usd"].pct_change(126)
usd_level_strength = rolling_zscore(macro_raw["usd"], zscore_window)
usd_1m_strength = rolling_zscore(usd_1m, zscore_window)
usd_3m_strength = rolling_zscore(usd_3m, zscore_window)

vix_level = macro_raw["vix_level"]
vix_1m = macro_raw["vix_level"].pct_change(21)

latest_macro_date = prices.index[-1]
print(
    "Computed macro metrics "
    f"for {latest_macro_date.date()}: "
    f"yield_curve={float(yield_curve.loc[latest_macro_date]):.17g}, "
    f"vix_level={float(vix_level.loc[latest_macro_date]):.17g}"
)

copper_1m = macro_raw["copper"].pct_change(21)
copper_3m = macro_raw["copper"].pct_change(63)
copper_rel_spy_1m = copper_1m - spy_ret_1m
copper_rel_spy_3m = copper_3m - spy_ret_3m

ita_1m = prices["ITA"].pct_change(21)
ita_rel_spy_1m = ita_1m - spy_ret_1m

soxx_1m = prices["SOXX"].pct_change(21)
soxx_3m = prices["SOXX"].pct_change(63)
soxx_rel_spy_1m = soxx_1m - spy_ret_1m

# Short-term breakdown indicators for the conditional divergence trigger.
# These are NOT used to reduce TQQQ unless divergence is already high.
soxx_5d = prices["SOXX"].pct_change(5)
soxx_10d = prices["SOXX"].pct_change(10)
soxx_dd_21 = prices["SOXX"] / prices["SOXX"].rolling(21).max() - 1.0
qqqm_5d = prices["QQQM"].pct_change(5)
qqqm_10d = prices["QQQM"].pct_change(10)
qqqm_dd_21 = prices["QQQM"] / prices["QQQM"].rolling(21).max() - 1.0

hyg_1m = prices["HYG"].pct_change(21)
hyg_3m = prices["HYG"].pct_change(63)
hyg_6m = prices["HYG"].pct_change(126)
hyg_rel_spy_1m = hyg_1m - spy_ret_1m
hyg_rel_spy_3m = hyg_3m - spy_ret_3m

qqqm_rel_spy_1m = prices["QQQM"].pct_change(21) - spy_ret_1m
xli_rel_spy_1m = prices["XLI"].pct_change(21) - spy_ret_1m
xlb_rel_spy_1m = prices["XLB"].pct_change(21) - spy_ret_1m

# Continuous regime strengths
ita_strength = rolling_zscore(ita_rel_spy_1m, zscore_window)
soxx_strength = rolling_zscore(soxx_rel_spy_1m, zscore_window)
qqqm_strength = rolling_zscore(qqqm_rel_spy_1m, zscore_window)
xli_strength = rolling_zscore(xli_rel_spy_1m, zscore_window)
xlb_strength = rolling_zscore(xlb_rel_spy_1m, zscore_window)
oil_strength = rolling_zscore(oil_1m, zscore_window)
vix_strength = rolling_zscore(vix_1m, zscore_window)
copper_strength = rolling_zscore(copper_rel_spy_1m, zscore_window)
copper_3m_strength = rolling_zscore(copper_rel_spy_3m, zscore_window)
hyg_strength = rolling_zscore(hyg_rel_spy_1m, zscore_window)

war_strength = ((ita_strength + oil_strength) / 2.0).fillna(0.0)
growth_strength = ((qqqm_strength + soxx_strength) / 2.0).fillna(0.0)
risk_off_strength = ((-qqqm_strength + vix_strength) / 2.0).fillna(0.0)
credit_strength = ((hyg_strength - vix_strength) / 2.0).fillna(0.0)

# New: industrial/materials score.
# Idea:
#   copper = raw materials demand
#   XLB relative strength = materials confirmation
#   XLI relative strength = industrial/capex confirmation
#   credit strength = risk-on confirmation
industrial_strength = (
    0.40 * copper_strength
    + 0.25 * xlb_strength
    + 0.20 * xli_strength
    + 0.15 * credit_strength
).fillna(0.0).clip(-3, 3)

materials_strength = (
    0.60 * copper_strength
    + 0.25 * copper_3m_strength
    + 0.15 * xlb_strength
).fillna(0.0).clip(-3, 3)

# Divergence detector:
#   Positive = tech/semis are strong, but industrial/copper/materials confirmation is weak.
#   It does not force an exit by itself. It becomes dangerous when risk-off also rises.
tech_real_economy_divergence = (
    0.50 * soxx_strength
    + 0.50 * qqqm_strength
    - 0.35 * industrial_strength
    - 0.35 * materials_strength
    - 0.30 * copper_strength
).fillna(0.0).clip(-5, 5)

crash_pressure = (
    0.60 * tech_real_economy_divergence
    + 0.40 * risk_off_strength
).fillna(0.0).clip(-5, 5)

# ============================================================
# 5. FEATURE BUILDER
# ============================================================
def build_features_by_asset(sector_etfs: list):
    features_by_asset = {}

    for asset in sector_etfs:
        px = prices[asset]

        ret_1m = px.pct_change(21)
        ret_3m = px.pct_change(63)
        ret_6m = px.pct_change(126)
        ret_12m = px.pct_change(252)
        rel_6m_vs_spy = ret_6m - spy_ret_6m
        vol_1m = px.pct_change().rolling(21).std() * np.sqrt(252)
        vol_3m = px.pct_change().rolling(63).std() * np.sqrt(252)

        is_QQQM = 1 if asset == "QQQM" else 0
        is_XLE = 1 if asset == "XLE" else 0
        is_XSOE = 1 if asset == "XSOE" else 0
        is_XLI = 1 if asset == "XLI" else 0
        is_XLB = 1 if asset == "XLB" else 0
        is_SOXX = 1 if asset == "SOXX" else 0
        is_XLF = 1 if asset == "XLF" else 0
        is_XLV = 1 if asset == "XLV" else 0
        is_IWM = 1 if asset == "IWM" else 0
        include_expanded_012_features = any(a in sector_etfs for a in ["SOXX", "XLF", "XLV", "IWM"])

        df = pd.DataFrame({
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "rel_6m_vs_spy": rel_6m_vs_spy,

            "spy_ret_1m": spy_ret_1m,
            "spy_ret_3m": spy_ret_3m,
            "spy_ret_6m": spy_ret_6m,
            "spy_vol_1m": spy_vol_1m,
            "spy_vol_3m": spy_vol_3m,

            "vol_1m": vol_1m,
            "vol_3m": vol_3m,

            "short_rate": short_rate,
            "long_rate": long_rate,
            "yield_curve": yield_curve,

            "oil_1m": oil_1m,
            "oil_3m": oil_3m,

            "usd_1m": usd_1m,
            "usd_3m": usd_3m,
            "usd_6m": usd_6m,
            "usd_level_strength": usd_level_strength,
            "usd_1m_strength": usd_1m_strength,
            "usd_3m_strength": usd_3m_strength,

            "vix_level": vix_level,
            "vix_1m": vix_1m,

            "copper_1m": copper_1m,
            "copper_3m": copper_3m,
            "copper_rel_spy_1m": copper_rel_spy_1m,
            "copper_rel_spy_3m": copper_rel_spy_3m,
            "copper_strength": copper_strength,
            "copper_3m_strength": copper_3m_strength,

            "ita_1m": ita_1m,
            "ita_rel_spy_1m": ita_rel_spy_1m,

            "soxx_1m": soxx_1m,
            "soxx_3m": soxx_3m,
            "soxx_rel_spy_1m": soxx_rel_spy_1m,
            "soxx_5d": soxx_5d,
            "soxx_10d": soxx_10d,
            "soxx_dd_21": soxx_dd_21,
            "qqqm_5d": qqqm_5d,
            "qqqm_10d": qqqm_10d,
            "qqqm_dd_21": qqqm_dd_21,

            "hyg_1m": hyg_1m,
            "hyg_3m": hyg_3m,
            "hyg_6m": hyg_6m,
            "hyg_rel_spy_1m": hyg_rel_spy_1m,
            "hyg_rel_spy_3m": hyg_rel_spy_3m,
            "hyg_strength": hyg_strength,
            "credit_strength": credit_strength,

            "qqqm_rel_spy_1m": qqqm_rel_spy_1m,
            "xli_rel_spy_1m": xli_rel_spy_1m,
            "xlb_rel_spy_1m": xlb_rel_spy_1m,

            "war_strength": war_strength,
            "growth_strength": growth_strength,
            "risk_off_strength": risk_off_strength,
            "industrial_strength": industrial_strength,
            "materials_strength": materials_strength,
            "tech_real_economy_divergence": tech_real_economy_divergence,
            "crash_pressure": crash_pressure,

            "is_QQQM": is_QQQM,
            "is_XLE": is_XLE,
            "is_XSOE": is_XSOE,
            "is_XLI": is_XLI,
            "is_XLB": is_XLB,
            "is_SOXX": is_SOXX,
            "is_XLF": is_XLF,
            "is_XLV": is_XLV,
            "is_IWM": is_IWM,

            # Asset-specific interactions.
            "yield_curve_QQQM": yield_curve * is_QQQM,

            "usd_XSOE": usd_1m * is_XSOE,
            "usd_3m_XSOE": usd_3m * is_XSOE,
            "usd_6m_XSOE": usd_6m * is_XSOE,
            "usd_level_XSOE": usd_level_strength * is_XSOE,
            "usd_1m_strength_XSOE": usd_1m_strength * is_XSOE,
            "usd_3m_strength_XSOE": usd_3m_strength * is_XSOE,

            "copper_XSOE": copper_strength * is_XSOE,
            "copper_XLB": copper_strength * is_XLB,
            "copper_3m_XLB": copper_3m_strength * is_XLB,
            "copper_XLI": copper_strength * is_XLI,

            "industrial_XLI": industrial_strength * is_XLI,
            "materials_XLB": materials_strength * is_XLB,
            "growth_XLI": growth_strength * is_XLI,

            "hyg_QQQM": hyg_strength * is_QQQM,
            "hyg_XSOE": hyg_strength * is_XSOE,
            "hyg_XLI": hyg_strength * is_XLI,
            "credit_QQQM": credit_strength * is_QQQM,
            "credit_XSOE": credit_strength * is_XSOE,
            "credit_XLI": credit_strength * is_XLI,

            "war_XLE": war_strength * is_XLE,
            "growth_QQQM": growth_strength * is_QQQM,
            "risk_off_QQQM": risk_off_strength * is_QQQM,
            "divergence_QQQM": tech_real_economy_divergence * is_QQQM,
            "crash_pressure_QQQM": crash_pressure * is_QQQM,
            "risk_off_XSOE": risk_off_strength * is_XSOE,
            "risk_off_XLI": risk_off_strength * is_XLI,
            "risk_off_XLB": risk_off_strength * is_XLB,
            "growth_SOXX": growth_strength * is_SOXX,
            "soxx_SOXX": soxx_strength * is_SOXX,
            "risk_off_SOXX": risk_off_strength * is_SOXX,
            "crash_pressure_SOXX": crash_pressure * is_SOXX,
            "yield_curve_XLF": yield_curve * is_XLF,
            "credit_XLF": credit_strength * is_XLF,
            "hyg_XLF": hyg_strength * is_XLF,
            "risk_off_XLF": risk_off_strength * is_XLF,
            "risk_off_XLV": risk_off_strength * is_XLV,
            "growth_XLV": growth_strength * is_XLV,
            "credit_IWM": credit_strength * is_IWM,
            "growth_IWM": growth_strength * is_IWM,
            "risk_off_IWM": risk_off_strength * is_IWM,
            "usd_3m_IWM": usd_3m_strength * is_IWM,
        })

        if not include_expanded_012_features:
            df = df.drop(columns=[
                "is_SOXX",
                "is_XLF",
                "is_XLV",
                "is_IWM",
                "growth_SOXX",
                "soxx_SOXX",
                "risk_off_SOXX",
                "crash_pressure_SOXX",
                "yield_curve_XLF",
                "credit_XLF",
                "hyg_XLF",
                "risk_off_XLF",
                "risk_off_XLV",
                "growth_XLV",
                "credit_IWM",
                "growth_IWM",
                "risk_off_IWM",
                "usd_3m_IWM",
            ])

        df["target"] = px.shift(-forward_return_days) / px - 1.0
        features_by_asset[asset] = df

    return features_by_asset

# ============================================================
# 6. TRAINING HELPERS
# ============================================================
def build_train_data(features_by_asset, asset_list, end_loc, train_window):
    start_loc = end_loc - train_window
    if start_loc < 0:
        return None, None

    x_parts = []
    y_parts = []

    for asset in asset_list:
        df = features_by_asset[asset].iloc[start_loc:end_loc].copy().dropna()
        if df.empty:
            continue
        x_parts.append(df.drop(columns=["target"]))
        y_parts.append(df["target"])

    if len(x_parts) == 0:
        return None, None

    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)

    common_idx = x_train.index.intersection(y_train.index)
    x_train = x_train.loc[common_idx]
    y_train = y_train.loc[common_idx]

    if len(x_train) == 0:
        return None, None
    return x_train, y_train


def get_today_features(features_by_asset, asset: str, date: pd.Timestamp):
    row = features_by_asset[asset].loc[[date]].drop(columns=["target"], errors="ignore")
    if row.empty:
        return None
    if row.isna().any(axis=1).iloc[0]:
        return None
    return row

# ============================================================
# 7. OVERLAY / ALLOCATION LOGIC
# ============================================================
def apply_regime_overlay(raw_preds: dict, date: pd.Timestamp, sector_etfs: list, overlay_style: str = "v1"):
    def val(series, default=0.0):
        if date in series.index and pd.notna(series.loc[date]):
            return float(series.loc[date])
        return default

    war = val(war_strength)
    growth = val(growth_strength)
    risk_off = val(risk_off_strength)
    soxx = val(soxx_strength)
    copper = val(copper_strength)
    copper3 = val(copper_3m_strength)
    industrial = val(industrial_strength)
    materials = val(materials_strength)
    yield_curve_now = val(yield_curve)
    usd_regime = val(usd_3m_strength)
    hyg_regime = val(hyg_strength)
    credit_regime = val(credit_strength)
    divergence = val(tech_real_economy_divergence)
    crash = val(crash_pressure)

    # Conditional SOXX/QQQM breakdown data.
    soxx_5d_now = val(soxx_5d)
    soxx_10d_now = val(soxx_10d)
    soxx_dd_21_now = val(soxx_dd_21)
    qqqm_5d_now = val(qqqm_5d)
    qqqm_10d_now = val(qqqm_10d)
    qqqm_dd_21_now = val(qqqm_dd_21)

    breakdown_score = 0.0
    breakdown_score += 1.0 if soxx_5d_now <= breakdown_component_thresholds["soxx_5d_max"] else 0.0
    breakdown_score += 1.0 if soxx_10d_now <= breakdown_component_thresholds["soxx_10d_max"] else 0.0
    breakdown_score += 1.0 if soxx_dd_21_now <= breakdown_component_thresholds["soxx_dd_21_max"] else 0.0
    breakdown_score += 1.0 if qqqm_5d_now <= breakdown_component_thresholds["qqqm_5d_max"] else 0.0
    breakdown_score += 1.0 if qqqm_10d_now <= breakdown_component_thresholds["qqqm_10d_max"] else 0.0
    breakdown_score += 1.0 if qqqm_dd_21_now <= breakdown_component_thresholds["qqqm_dd_21_max"] else 0.0

    adjusted = raw_preds.copy()

    war_pos = max(0.0, war)
    growth_pos = max(0.0, growth)
    risk_off_pos = max(0.0, risk_off)
    soxx_pos = max(0.0, soxx)
    copper_pos = max(0.0, copper)
    copper3_pos = max(0.0, copper3)
    industrial_pos = max(0.0, industrial)
    materials_pos = max(0.0, materials)
    usd_regime_pos = max(0.0, usd_regime)
    hyg_pos = max(0.0, hyg_regime)
    credit_pos = max(0.0, credit_regime)
    scale = overlay_scale

    def add(asset, amount):
        if asset in adjusted:
            adjusted[asset] += amount

    # Existing logic preserved.
    if war_pos > 0:
        add("XLE", scale * war_pos)
        add("QQQM", -scale * 0.5 * war_pos)
        add("XSOE", -scale * 0.3 * war_pos)

    if growth_pos > 0:
        add("QQQM", scale * growth_pos)
        add("XLE", -scale * 0.4 * growth_pos)

    if soxx_pos > 0:
        add("QQQM", scale * 0.8 * soxx_pos)

    if copper_pos > 0:
        add("XSOE", scale * 0.8 * copper_pos)

    if usd_regime_pos > 0:
        add("XSOE", -scale * 0.6 * usd_regime_pos)

    if hyg_pos > 0:
        add("QQQM", scale * 0.35 * hyg_pos)
        add("XSOE", scale * 0.45 * hyg_pos)

    if credit_pos > 0:
        add("QQQM", scale * 0.25 * credit_pos)
        add("XSOE", scale * 0.35 * credit_pos)

    if risk_off_pos > 0:
        add("QQQM", -scale * 1.2 * risk_off_pos)
        add("XSOE", -scale * 1.0 * risk_off_pos)

    if war_pos > 0 and risk_off_pos > 1.0:
        add("XLE", -scale * 0.4 * risk_off_pos)

    # 012 universe expansion overlays.
    # These are intentionally modest so new assets must still win mostly through ML forecasts.
    if "SOXX" in sector_etfs:
        if soxx_pos > 0:
            add("SOXX", scale * 0.90 * soxx_pos)
        if growth_pos > 0:
            add("SOXX", scale * 0.35 * growth_pos)
        if risk_off_pos > 0:
            add("SOXX", -scale * 1.10 * risk_off_pos)
        if crash > 1.0:
            add("SOXX", -scale * 0.40 * max(0.0, crash))

    if "XLF" in sector_etfs:
        if credit_pos > 0:
            add("XLF", scale * 0.55 * credit_pos)
        if hyg_pos > 0:
            add("XLF", scale * 0.30 * hyg_pos)
        if yield_curve_now > 0:
            add("XLF", scale * 0.25 * min(yield_curve_now, 3.0))
        if risk_off_pos > 0:
            add("XLF", -scale * 0.95 * risk_off_pos)

    if "XLV" in sector_etfs:
        if 0.0 < risk_off <= 1.25:
            add("XLV", scale * 0.45 * risk_off_pos)
        if risk_off > 1.75:
            add("XLV", -scale * 0.35 * risk_off_pos)
        if growth > 1.25 and soxx > 1.00 and risk_off < 0.50:
            add("XLV", -scale * 0.20 * growth_pos)

    if "IWM" in sector_etfs:
        if growth_pos > 0 and credit_regime > -0.50 and risk_off < 0.75:
            add("IWM", scale * 0.35 * growth_pos)
        if credit_pos > 0:
            add("IWM", scale * 0.25 * credit_pos)
        if usd_regime > 1.0:
            add("IWM", -scale * 0.30 * usd_regime_pos)
        if risk_off_pos > 0:
            add("IWM", -scale * 1.00 * risk_off_pos)

    # New XLI/XLB overlay logic.
    # V1 = original broad boost; V2 = stricter regime classifier.
    # Keep overlays modest because ML already sees the features.
    if overlay_style == "v1":
        # XLB: early industrial/materials/copper cycle.
        if "XLB" in sector_etfs:
            if copper_pos > 0:
                add("XLB", scale * 0.90 * copper_pos)
            if copper3_pos > 0:
                add("XLB", scale * 0.35 * copper3_pos)
            if materials_pos > 0:
                add("XLB", scale * 0.50 * materials_pos)
            if risk_off_pos > 0:
                add("XLB", -scale * 0.70 * risk_off_pos)

        # XLI: industrial/reshoring/capex cycle; likes industrial acceleration and credit support.
        if "XLI" in sector_etfs:
            if industrial_pos > 0:
                add("XLI", scale * 0.80 * industrial_pos)
            if growth_pos > 0:
                add("XLI", scale * 0.35 * growth_pos)
            if credit_pos > 0:
                add("XLI", scale * 0.25 * credit_pos)
            if risk_off_pos > 0:
                add("XLI", -scale * 0.80 * risk_off_pos)

    elif overlay_style in ("v2", "hybrid"):
        # V2 philosophy:
        # - XLB should be boosted only when materials/copper strength is confirmed.
        # - XLI should be boosted only when industrial strength is positive AND risk-off is not dominant.
        # - Strong USD/risk-off gets a small penalty because it often hurts global cyclicals/materials.
        industrial_regime_on = (industrial > 0.25) and (risk_off < 0.75) and (usd_regime < 1.50)
        materials_regime_on = (materials > 0.25) and (risk_off < 1.00)
        early_cycle_on = (copper3 > 0.50) and (credit_regime > -0.50) and (risk_off < 1.00)

        if "XLB" in sector_etfs:
            if materials_regime_on:
                add("XLB", scale * 0.70 * materials_pos)
            if early_cycle_on:
                add("XLB", scale * 0.30 * copper3_pos)
            if risk_off_pos > 0:
                add("XLB", -scale * 0.85 * risk_off_pos)
            if usd_regime > 1.0:
                add("XLB", -scale * 0.20 * usd_regime_pos)

        if "XLI" in sector_etfs:
            if industrial_regime_on:
                add("XLI", scale * 0.75 * industrial_pos)
                if growth_pos > 0:
                    add("XLI", scale * 0.20 * growth_pos)
                if credit_pos > 0:
                    add("XLI", scale * 0.20 * credit_pos)
            if risk_off_pos > 0:
                add("XLI", -scale * 0.90 * risk_off_pos)
            if usd_regime > 1.5:
                add("XLI", -scale * 0.15 * usd_regime_pos)

        # HYBRID extra: preserve fast-growth tech/TQQQ engine when semis + QQQM leadership are very strong.
        # This prevents XLI/XLB from diluting the original tech engine unless industrial/materials signals are truly active.
        if overlay_style == "hybrid":
            tech_regime_on = (growth > 1.00) and (soxx > 1.00) and (risk_off < 0.50)
            industrial_regime_on_h = (industrial > 0.50) and (copper3 > 0.25) and (risk_off < 0.75) and (usd_regime < 1.25)
            materials_regime_on_h = (materials > 0.50) and (copper3 > 0.50) and (risk_off < 0.90)
            if tech_regime_on:
                add("QQQM", scale * 0.45 * min(growth_pos + soxx_pos, 6.0))
                if "XLI" in sector_etfs and not industrial_regime_on_h:
                    add("XLI", -scale * 0.25 * max(0.0, -industrial))
                if "XLB" in sector_etfs and not materials_regime_on_h:
                    add("XLB", -scale * 0.25 * max(0.0, -materials))

            # NEW: Non-tech conviction boost.
            # Purpose: reduce tech bias when oil/industrial/materials regimes are truly strong.
            oil_regime = war
            if oil_regime > 0.5:
                add("XLE", scale * 1.5 * oil_regime)
            if industrial > 0.5:
                add("XLI", scale * 1.2 * industrial)
            if copper > 0.7:
                add("XLB", scale * 1.0 * copper)
            if oil_regime > 0.5 or industrial > 0.5:
                add("QQQM", -scale * 0.8 * max(oil_regime, industrial))

    else:
        raise ValueError(f"Unknown overlay_style: {overlay_style}")

    overlay_info = {
        "war_strength": war,
        "growth_strength": growth,
        "risk_off_strength": risk_off,
        "soxx_strength": soxx,
        "copper_strength": copper,
        "copper_3m_strength": copper3,
        "industrial_strength": industrial,
        "materials_strength": materials,
        "usd_3m_strength": usd_regime,
        "hyg_strength": hyg_regime,
        "credit_strength": credit_regime,
        "tech_real_economy_divergence": divergence,
        "crash_pressure": crash,
        "soxx_5d": soxx_5d_now,
        "soxx_10d": soxx_10d_now,
        "soxx_dd_21": soxx_dd_21_now,
        "qqqm_5d": qqqm_5d_now,
        "qqqm_10d": qqqm_10d_now,
        "qqqm_dd_21": qqqm_dd_21_now,
        "breakdown_score": breakdown_score,
        "overlay_style": overlay_style,
    }
    return adjusted, overlay_info


def should_go_cash(top_score: float, second_score: float, risk_off_strength_val: float):
    threshold = 0.0
    if risk_off_strength_val > 1.0:
        threshold += risk_off_cash_threshold
    return (top_score < threshold) and (second_score < threshold)


def apply_soxx_admission_filter(adjusted_preds: dict, overlay_info: dict, mode: str = "none") -> dict:
    """Research filter: SOXX must have regime support before it can win the ranking."""
    if mode not in ("strict", "hurdle") or "SOXX" not in adjusted_preds:
        return adjusted_preds

    growth = float(overlay_info.get("growth_strength", 0.0))
    soxx = float(overlay_info.get("soxx_strength", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    crash = float(overlay_info.get("crash_pressure", 0.0))
    breakdown = float(overlay_info.get("breakdown_score", 0.0))

    regime_confirmed = (
        growth >= 0.0
        and soxx >= 0.0
        and risk_off <= 0.75
        and crash <= 0.75
        and breakdown <= 2.0
    )
    if regime_confirmed:
        return adjusted_preds

    filtered = adjusted_preds.copy()
    non_soxx_scores = [v for k, v in filtered.items() if k != "SOXX"]
    if not non_soxx_scores:
        return filtered

    best_non_soxx = max(non_soxx_scores)
    if mode == "strict":
        # Keep SOXX visible, but force it below the best confirmed alternative.
        filtered["SOXX"] = min(filtered["SOXX"], best_non_soxx - 1e-6)
        return filtered

    hurdle = 0.012
    if growth < 0.0:
        hurdle += 0.006
    if soxx < 0.0:
        hurdle += 0.006
    if risk_off > 0.50:
        hurdle += 0.006
    if breakdown > 2.0:
        hurdle += 0.004
    if crash > 0.50:
        hurdle += 0.004

    soxx_edge = filtered["SOXX"] - best_non_soxx
    if soxx_edge <= hurdle:
        filtered["SOXX"] = best_non_soxx - 1e-6
    else:
        filtered["SOXX"] = best_non_soxx + min(soxx_edge - hurdle, 0.020)
    return filtered


def apply_expanded_012_guardrails(signal_weights: dict, top_asset: str, second_asset: str, overlay_info: dict, sector_etfs: list) -> dict:
    """Research-only cap to keep SOXX from dominating during weak tech/risk regimes."""
    if "SOXX" not in sector_etfs:
        return signal_weights

    soxx_weight = signal_weights.get("SOXX", 0.0)
    if soxx_weight <= 0:
        return signal_weights

    growth = float(overlay_info.get("growth_strength", 0.0))
    soxx = float(overlay_info.get("soxx_strength", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    crash = float(overlay_info.get("crash_pressure", 0.0))

    if risk_off > 1.0 or crash > 1.0:
        soxx_cap = 0.25
    elif growth < 0.0 or soxx < 0.0 or risk_off > 0.50:
        soxx_cap = 0.50
    else:
        soxx_cap = 0.70

    if soxx_weight <= soxx_cap:
        return signal_weights

    guarded = signal_weights.copy()
    excess = soxx_weight - soxx_cap
    guarded["SOXX"] = soxx_cap

    destination = second_asset if second_asset != "SOXX" and risk_off < 1.25 else cash_etf
    guarded[destination] = guarded.get(destination, 0.0) + excess
    return guarded


def get_conviction_weights(top_score: float, second_score: float):
    gap = top_score - second_score
    if gap < 0.005:
        w_top = 0.60
    elif gap < 0.015:
        w_top = 0.70
    elif gap < 0.030:
        w_top = 0.80
    elif gap < 0.050:
        w_top = 0.90
    else:
        w_top = 1.00
    return w_top, 1.0 - w_top, gap


def tqqq_replace_fraction(top_asset: str, top_score: float, second_score: float, overlay_info: dict, date: pd.Timestamp):
    if top_asset != "QQQM":
        return 0.0

    gap = top_score - second_score
    vix_now = float(vix_level.loc[date]) if date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan
    growth = overlay_info["growth_strength"]
    soxx = overlay_info["soxx_strength"]
    risk_off = overlay_info["risk_off_strength"]

    strong = (
        gap >= tiered_tqqq_rule["strong"]["gap_min"]
        and top_score >= tiered_tqqq_rule["strong"]["top_score_min"]
        and growth >= tiered_tqqq_rule["strong"]["growth_min"]
        and soxx >= tiered_tqqq_rule["strong"]["soxx_min"]
        and risk_off <= tiered_tqqq_rule["strong"]["risk_off_max"]
        and (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["strong"]["vix_max"])
    )
    if strong:
        return tiered_tqqq_rule["strong"]["replace_fraction"]

    moderate = (
        gap >= tiered_tqqq_rule["moderate"]["gap_min"]
        and top_score >= tiered_tqqq_rule["moderate"]["top_score_min"]
        and growth >= tiered_tqqq_rule["moderate"]["growth_min"]
        and soxx >= tiered_tqqq_rule["moderate"]["soxx_min"]
        and risk_off <= tiered_tqqq_rule["moderate"]["risk_off_max"]
        and (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["moderate"]["vix_max"])
    )
    if moderate:
        return tiered_tqqq_rule["moderate"]["replace_fraction"]

    return 0.0



def tqqq_dynamic_replace_fraction(top_asset: str, top_score: float, second_score: float, overlay_info: dict, date: pd.Timestamp):
    """Conviction + volatility adjusted TQQQ replacement fraction."""
    if top_asset != "QQQM":
        return 0.0

    gap = top_score - second_score
    growth = float(overlay_info.get("growth_strength", 0.0))
    soxx = float(overlay_info.get("soxx_strength", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    vix_now = float(vix_level.loc[date]) if date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan

    permission = (
        top_score >= 0.008
        and gap >= 0.002
        and growth >= 0.20
        and soxx >= 0.20
        and risk_off <= 1.00
        and (pd.isna(vix_now) or vix_now <= 25.0)
    )
    if not permission:
        return 0.0

    conviction = (gap - 0.002) / (0.020 - 0.002)
    conviction = float(np.clip(conviction, 0.0, 1.0))
    replace_fraction = 0.40 + 0.60 * conviction

    if pd.isna(vix_now):
        vol_adj = 1.0
    elif vix_now < 15:
        vol_adj = 1.1
    elif vix_now < 25:
        vol_adj = 1.0
    elif vix_now < 35:
        vol_adj = 0.8
    else:
        vol_adj = 0.6
    replace_fraction *= vol_adj

    if growth > 1.0 and soxx > 1.0 and risk_off < 0.50:
        replace_fraction += 0.10

    if risk_off > 0.50:
        replace_fraction *= 0.75

    return float(np.clip(replace_fraction, 0.0, 1.0))


def multi_asset_leverage_fraction(asset: str, top_asset: str, score_gap: float, overlay_info: dict, date: pd.Timestamp):
    """Safe 2x leverage gate for XLE->ERX and XLI->UXI."""
    if asset != top_asset:
        return 0.0

    vix_now = float(vix_level.loc[date]) if date is not None and date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))

    if score_gap > 0.015:
        frac = 0.50
    elif score_gap > 0.006:
        frac = 0.25
    else:
        frac = 0.0

    if not pd.isna(vix_now):
        if vix_now > 30:
            frac *= 0.3
        elif vix_now > 25:
            frac *= 0.6

    if risk_off > 0.5:
        frac *= 0.5

    return float(np.clip(frac, 0.0, 0.60))


def soxx_leverage_fraction(top_asset: str, score_gap: float, overlay_info: dict, date: pd.Timestamp, mode: str = "strict"):
    """Research-only 3x semiconductor leverage gate for SOXX->SOXL."""
    if top_asset != "SOXX":
        return 0.0

    growth = float(overlay_info.get("growth_strength", 0.0))
    soxx = float(overlay_info.get("soxx_strength", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    crash = float(overlay_info.get("crash_pressure", 0.0))
    breakdown = float(overlay_info.get("breakdown_score", 0.0))
    vix_now = float(vix_level.loc[date]) if date is not None and date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan

    if mode == "mild":
        permission = (
            score_gap >= 0.015
            and growth >= 0.00
            and soxx >= 0.00
            and risk_off <= 0.75
            and crash <= 1.00
            and breakdown <= 2.0
            and (pd.isna(vix_now) or vix_now <= 26.0)
        )
        if not permission:
            return 0.0
        frac = 0.25
    else:
        permission = (
            score_gap >= 0.010
            and growth >= 0.50
            and soxx >= 0.50
            and risk_off <= 0.50
            and crash <= 0.75
            and breakdown <= 1.0
            and (pd.isna(vix_now) or vix_now <= 24.0)
        )
        if not permission:
            return 0.0

        if score_gap >= 0.030 and growth >= 1.00 and soxx >= 1.00 and risk_off <= 0.25:
            frac = 0.50
        else:
            frac = 0.25

    if not pd.isna(vix_now):
        if vix_now > 22:
            frac *= 0.6
        elif vix_now < 16:
            frac *= 1.1

    return float(np.clip(frac, 0.0, 0.55))


def conditional_breakdown_defense_level(overlay_info: dict) -> str:
    """
    Conditional trigger:
    - Divergence alone does nothing.
    - Defense activates only when SOXX/QQQM actually starts breaking.
    """
    if not use_conditional_breakdown_defense:
        return "off"

    divergence = float(overlay_info.get("tech_real_economy_divergence", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    breakdown_score = float(overlay_info.get("breakdown_score", 0.0))

    for level in ["danger", "warning", "watch"]:
        rule = conditional_breakdown_rules[level]
        if (
            divergence >= rule["divergence_min"]
            and breakdown_score >= rule["breakdown_score_min"]
            and risk_off >= rule["risk_off_min"]
        ):
            return level
    return "normal"


def apply_conditional_breakdown_defense(exec_weights: dict, overlay_info: dict) -> dict:
    """
    Reduce TQQQ only after tech leadership starts failing.
    Freed TQQQ exposure is first moved to QQQM; optional cash buffer then scales down risk.
    """
    level = conditional_breakdown_defense_level(overlay_info)
    if level in ("off", "normal"):
        return exec_weights

    rule = conditional_breakdown_rules[level]
    tqqq_multiplier = rule["tqqq_multiplier"]
    cash_buffer = rule["cash_buffer"]

    old_tqqq = exec_weights.get("TQQQ", 0.0)
    new_tqqq = old_tqqq * tqqq_multiplier
    freed_from_tqqq = old_tqqq - new_tqqq

    exec_weights["TQQQ"] = new_tqqq
    exec_weights["QQQM"] = exec_weights.get("QQQM", 0.0) + freed_from_tqqq

    if cash_buffer > 0:
        for asset in list(exec_weights.keys()):
            if asset != cash_etf:
                exec_weights[asset] *= (1.0 - cash_buffer)
        exec_weights[cash_etf] = exec_weights.get(cash_etf, 0.0) + cash_buffer

    total = sum(exec_weights.values())
    if total > 0 and abs(total - 1.0) > 1e-8:
        for asset in exec_weights:
            exec_weights[asset] /= total

    return exec_weights


def apply_soxx_execution_defense(exec_weights: dict, overlay_info: dict, mode: str = "none") -> dict:
    """Research-only SOXX/SOXL execution risk control."""
    if mode != "semi_defense":
        return exec_weights

    soxx_exposure = exec_weights.get("SOXX", 0.0)
    soxl_exposure = exec_weights.get("SOXL", 0.0)
    total_semi = soxx_exposure + soxl_exposure
    if total_semi <= 0:
        return exec_weights

    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    crash = float(overlay_info.get("crash_pressure", 0.0))
    breakdown = float(overlay_info.get("breakdown_score", 0.0))
    soxx_5d_now = float(overlay_info.get("soxx_5d", 0.0))
    soxx_10d_now = float(overlay_info.get("soxx_10d", 0.0))
    soxx_dd_now = float(overlay_info.get("soxx_dd_21", 0.0))

    stress = 0
    stress += 1 if risk_off > 0.50 else 0
    stress += 1 if risk_off > 1.00 else 0
    stress += 1 if crash > 0.50 else 0
    stress += 1 if breakdown >= 2.0 else 0
    stress += 1 if soxx_5d_now <= -0.030 else 0
    stress += 1 if soxx_10d_now <= -0.050 else 0
    stress += 1 if soxx_dd_now <= -0.080 else 0

    guarded = exec_weights.copy()

    if stress >= 1 and soxl_exposure > 0:
        guarded["SOXX"] = guarded.get("SOXX", 0.0) + soxl_exposure
        guarded["SOXL"] = 0.0

    if stress >= 4:
        cut = 0.50 * guarded.get("SOXX", 0.0)
        guarded["SOXX"] -= cut
        guarded[cash_etf] = guarded.get(cash_etf, 0.0) + cut
    elif stress >= 3:
        cut = 0.30 * guarded.get("SOXX", 0.0)
        guarded["SOXX"] -= cut
        destination = cash_etf if risk_off > 0.75 else "QQQM"
        guarded[destination] = guarded.get(destination, 0.0) + cut
    elif stress >= 2:
        cut = 0.15 * guarded.get("SOXX", 0.0)
        guarded["SOXX"] -= cut
        guarded["QQQM"] = guarded.get("QQQM", 0.0) + cut

    total = sum(guarded.values())
    if total > 0 and abs(total - 1.0) > 1e-8:
        for asset in guarded:
            guarded[asset] /= total

    return guarded


def build_execution_weights(
    signal_weights: dict,
    overlay_fraction: float,
    sector_etfs: list,
    top_asset=None,
    score_gap=0.0,
    overlay_info=None,
    date=None,
    soxx_execution_mode: str = "none",
    soxx_leverage_mode: str = "strict",
):
    exec_universe = ["TQQQ", "ERX", "UXI"] + (["SOXL"] if "SOXX" in sector_etfs else []) + sector_etfs + [cash_etf]
    exec_weights = {a: 0.0 for a in exec_universe}
    overlay_info = overlay_info or {}

    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    vix_now = float(vix_level.loc[date]) if date is not None and date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan

    if risk_off > 1.5 or (not pd.isna(vix_now) and vix_now > 32):
        exec_weights[cash_etf] = 1.0
        return exec_weights

    if risk_off > 1.0 or (not pd.isna(vix_now) and vix_now > 28):
        defensive_cash = 0.50
    else:
        defensive_cash = 0.0

    qqqm_signal = signal_weights.get("QQQM", 0.0)
    exec_weights["TQQQ"] = qqqm_signal * overlay_fraction
    exec_weights["QQQM"] = qqqm_signal * (1.0 - overlay_fraction)

    xle_signal = signal_weights.get("XLE", 0.0)
    if xle_signal > 0:
        frac = multi_asset_leverage_fraction("XLE", top_asset, score_gap, overlay_info, date)
        exec_weights["ERX"] = xle_signal * frac
        exec_weights["XLE"] = xle_signal * (1.0 - frac)

    xli_signal = signal_weights.get("XLI", 0.0)
    if xli_signal > 0:
        frac = multi_asset_leverage_fraction("XLI", top_asset, score_gap, overlay_info, date)
        exec_weights["UXI"] = xli_signal * frac
        exec_weights["XLI"] = xli_signal * (1.0 - frac)

    soxx_signal = signal_weights.get("SOXX", 0.0)
    if soxx_signal > 0 and "SOXX" in sector_etfs:
        frac = soxx_leverage_fraction(top_asset, score_gap, overlay_info, date, mode=soxx_leverage_mode)
        exec_weights["SOXL"] = soxx_signal * frac
        exec_weights["SOXX"] = soxx_signal * (1.0 - frac)

    for asset in sector_etfs:
        if asset not in ["QQQM", "XLE", "XLI", "SOXX"]:
            exec_weights[asset] = signal_weights.get(asset, 0.0)

    exec_weights[cash_etf] = signal_weights.get(cash_etf, 0.0)

    if defensive_cash > 0:
        for asset in exec_weights:
            if asset != cash_etf:
                exec_weights[asset] *= (1.0 - defensive_cash)
        exec_weights[cash_etf] = defensive_cash

    exec_weights = apply_conditional_breakdown_defense(exec_weights, overlay_info)
    exec_weights = apply_soxx_execution_defense(exec_weights, overlay_info, mode=soxx_execution_mode)

    return exec_weights

# ============================================================
# 8. STRATEGY RUNNER
# ============================================================
def run_strategy(
    model_name: str,
    sector_etfs: list,
    features_by_asset: dict,
    overlay_style: str = "v1",
    tqqq_style: str = "tiered",
    soxx_admission_mode: str = "none",
    soxx_execution_mode: str = "none",
    soxx_leverage_mode: str = "strict",
):
    signal_universe = sector_etfs + [cash_etf]
    exec_universe = ["TQQQ", "ERX", "UXI"] + (["SOXL"] if "SOXX" in sector_etfs else []) + sector_etfs + [cash_etf]
    dates = prices.index

    min_needed = max(train_window, 252) + 1
    max_loc = len(dates) - forward_return_days
    rebalance_locs = list(range(min_needed, max_loc, rebalance_step))

    portfolio_daily_returns = pd.Series(index=dates, dtype=float)
    current_exec_weights = {a: 0.0 for a in exec_universe}
    current_exec_weights[cash_etf] = 1.0

    rebalance_records = []
    turnover_list = []

    for i, loc in enumerate(rebalance_locs):
        rebalance_date = dates[loc]

        x_train, y_train = build_train_data(features_by_asset, sector_etfs, loc, train_window)
        if x_train is None or len(x_train) < 50:
            continue

        model = RandomForestRegressor(**rf_params)
        model.fit(x_train, y_train)

        raw_preds = {}
        for asset in sector_etfs:
            x_today = get_today_features(features_by_asset, asset, rebalance_date)
            if x_today is None:
                continue
            raw_preds[asset] = float(model.predict(x_today)[0])

        if len(raw_preds) < 2:
            continue

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, rebalance_date, sector_etfs, overlay_style=overlay_style)
        adjusted_preds = apply_soxx_admission_filter(adjusted_preds, overlay_info, mode=soxx_admission_mode)
        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset, top_score = ranked[0]
        second_asset, second_score = ranked[1]

        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)

        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        signal_weights = apply_expanded_012_guardrails(
            signal_weights,
            top_asset,
            second_asset,
            overlay_info,
            sector_etfs,
        )

        if tqqq_style == "dynamic":
            overlay_fraction = tqqq_dynamic_replace_fraction(
                top_asset, top_score, second_score, overlay_info, rebalance_date
            )
        else:
            overlay_fraction = tqqq_replace_fraction(
                top_asset, top_score, second_score, overlay_info, rebalance_date
            )

        exec_weights = build_execution_weights(
            signal_weights,
            overlay_fraction,
            sector_etfs,
            top_asset=top_asset,
            score_gap=score_gap,
            overlay_info=overlay_info,
            date=rebalance_date,
            soxx_execution_mode=soxx_execution_mode,
            soxx_leverage_mode=soxx_leverage_mode,
        )

        overlay_info["conditional_breakdown_defense_level"] = conditional_breakdown_defense_level(overlay_info)

        latest_like = {
            "exec_weights": exec_weights,
            "risk_off_strength": overlay_info.get("risk_off_strength", 0.0),
            "growth_strength": overlay_info.get("growth_strength", 0.0),
            "soxx_strength": overlay_info.get("soxx_strength", 0.0),
            "score_gap": score_gap,
            "top_score": top_score,
        }
        latest_like = apply_v2_continuous_tqqq_alert(latest_like)
        exec_weights = latest_like["exec_weights"]

        overlay_info["v2_tqqq_scale"] = latest_like.get("v2_tqqq_scale", 1.0)
        overlay_info["v2_alert_action"] = latest_like.get("v2_alert_action", "NONE")
        overlay_info["conditional_breakdown_defense_level"] = conditional_breakdown_defense_level(overlay_info)

        turnover = compute_turnover(current_exec_weights, exec_weights, exec_universe)
        turnover_list.append(turnover)

        next_loc = rebalance_locs[i + 1] if i + 1 < len(rebalance_locs) else max_loc
        hold_dates = dates[loc + 1: next_loc + 1]
        if len(hold_dates) == 0:
            continue

        hold_rets = pd.Series(index=hold_dates, data=0.0)
        for asset, w in exec_weights.items():
            if w != 0:
                hold_rets = hold_rets.add(
                    w * asset_returns[asset].reindex(hold_dates).fillna(0.0),
                    fill_value=0.0,
                )

        cost = turnover * transaction_cost
        hold_rets.iloc[0] -= cost
        portfolio_daily_returns.loc[hold_dates] = hold_rets.values
        current_exec_weights = exec_weights.copy()

        row = {
            "model": model_name,
            "date": rebalance_date,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "turnover": turnover,
            "tx_cost_applied": cost,
            "overlay_fraction": overlay_fraction,
            "v2_tqqq_scale": overlay_info.get("v2_tqqq_scale", 1.0),
            "v2_alert_action": overlay_info.get("v2_alert_action", "NONE"),
            **overlay_info,
        }

        for a in sector_etfs:
            row[f"raw_pred_{a}"] = raw_preds.get(a, np.nan)
            row[f"adj_pred_{a}"] = adjusted_preds.get(a, np.nan)
            row[f"signal_w_{a}"] = signal_weights.get(a, 0.0)
            row[f"exec_w_{a}"] = exec_weights.get(a, 0.0)

        row[f"signal_w_{cash_etf}"] = signal_weights.get(cash_etf, 0.0)
        row[f"exec_w_{cash_etf}"] = exec_weights.get(cash_etf, 0.0)
        row["exec_w_TQQQ"] = exec_weights.get("TQQQ", 0.0)
        row["exec_w_ERX"] = exec_weights.get("ERX", 0.0)
        row["exec_w_UXI"] = exec_weights.get("UXI", 0.0)
        row["exec_w_SOXL"] = exec_weights.get("SOXL", 0.0)

        rebalance_records.append(row)

    portfolio_daily_returns = portfolio_daily_returns.dropna()
    rebalance_df = pd.DataFrame(rebalance_records)
    avg_turnover = float(np.mean(turnover_list)) if turnover_list else np.nan

    return portfolio_daily_returns, rebalance_df, avg_turnover
# ============================================================
# 9. LATEST RECOMMENDATION
# ============================================================
def get_latest_recommendation(
    model_name: str,
    sector_etfs: list,
    features_by_asset: dict,
    overlay_style: str = "v1",
    tqqq_style: str = "tiered",
    soxx_admission_mode: str = "none",
    soxx_execution_mode: str = "none",
    soxx_leverage_mode: str = "strict",
):
    signal_universe = sector_etfs + [cash_etf]
    dates = prices.index

    for latest_loc in range(len(dates) - 1, train_window, -1):
        latest_date = dates[latest_loc]

        x_train, y_train = build_train_data(features_by_asset, sector_etfs, latest_loc, train_window)
        if x_train is None or len(x_train) < 50:
            continue

        model = RandomForestRegressor(**rf_params)
        model.fit(x_train, y_train)

        raw_preds = {}
        for asset in sector_etfs:
            x_today = get_today_features(features_by_asset, asset, latest_date)
            if x_today is None:
                continue
            raw_preds[asset] = float(model.predict(x_today)[0])

        if len(raw_preds) < 2:
            continue

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, latest_date, sector_etfs, overlay_style=overlay_style)
        adjusted_preds = apply_soxx_admission_filter(adjusted_preds, overlay_info, mode=soxx_admission_mode)
        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset, top_score = ranked[0]
        second_asset, second_score = ranked[1]
        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)

        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        signal_weights = apply_expanded_012_guardrails(
            signal_weights,
            top_asset,
            second_asset,
            overlay_info,
            sector_etfs,
        )

        if tqqq_style == "dynamic":
            overlay_fraction = tqqq_dynamic_replace_fraction(top_asset, top_score, second_score, overlay_info, latest_date)
        else:
            overlay_fraction = tqqq_replace_fraction(top_asset, top_score, second_score, overlay_info, latest_date)
        exec_weights = build_execution_weights(
            signal_weights,
            overlay_fraction,
            sector_etfs,
            top_asset=top_asset,
            score_gap=score_gap,
            overlay_info=overlay_info,
            date=latest_date,
            soxx_execution_mode=soxx_execution_mode,
            soxx_leverage_mode=soxx_leverage_mode,
        )

        overlay_info["conditional_breakdown_defense_level"] = conditional_breakdown_defense_level(overlay_info)

        feature_importance_df = pd.DataFrame({
            "feature": x_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return {
            "model": model_name,
            "date": latest_date,
            "feature_date": latest_date,
            "yield_curve": float(yield_curve.loc[latest_date]),
            "vix_level": float(vix_level.loc[latest_date]),
            "raw_predictions": raw_preds,
            "adjusted_predictions": adjusted_preds,
            "signal_weights": signal_weights,
            "exec_weights": exec_weights,
            "feature_importance": feature_importance_df,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "overlay_fraction": overlay_fraction,
            "conditional_breakdown_defense_level": conditional_breakdown_defense_level(overlay_info),
            **overlay_info,
        }
    return None

# ============================================================
# 10. PRINT / SAVE HELPERS
# ============================================================
def print_weights(title: str, weights: dict, order: list):
    print(f"\n=== {title} ===")
    for asset in order:
        print(f"{asset}: {weights.get(asset, 0.0):.1%}")


def print_latest(latest: dict, sector_etfs: list):
    if latest is None:
        print("No latest recommendation available.")
        return
    print(f"\n=== Latest Recommendation: {latest['model']} ===")
    print("Signal date:", latest["date"].date())

    print("\nRaw predicted next-period returns:")
    for k, v in sorted(latest["raw_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print("\nAdjusted predicted next-period returns:")
    for k, v in sorted(latest["adjusted_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print(f"\nOverlay fraction on QQQM sleeve: {latest['overlay_fraction']:.1%}")
    print(f"Top asset: {latest['top_asset']}")
    print(f"Second asset: {latest['second_asset']}")
    print(f"Top score: {latest['top_score']:.4f}")
    print(f"Second score: {latest['second_score']:.4f}")
    print(f"Score gap: {latest['score_gap']:.4f}")
    print(f"Growth strength: {latest['growth_strength']:.3f}")
    print(f"SOXX strength: {latest['soxx_strength']:.3f}")
    print(f"Risk-off strength: {latest['risk_off_strength']:.3f}")
    print(f"Overlay style: {latest.get('overlay_style', 'v1')}")
    print(f"Copper strength: {latest['copper_strength']:.3f}")
    print(f"Industrial strength: {latest['industrial_strength']:.3f}")
    print(f"Materials strength: {latest['materials_strength']:.3f}")
    print(f"USD 3M strength: {latest['usd_3m_strength']:.3f}")
    print(f"Credit strength: {latest['credit_strength']:.3f}")
    print(f"Tech/real-economy divergence: {latest['tech_real_economy_divergence']:.3f}")
    print(f"Crash pressure: {latest['crash_pressure']:.3f}")
    print(f"Divergence defense level: {latest.get('conditional_breakdown_defense_level', 'normal')}")

    print_weights("Suggested SIGNAL Weights", latest["signal_weights"], sector_etfs + [cash_etf])
    print_weights("Suggested EXECUTED Weights", latest["exec_weights"], ["TQQQ", "ERX", "UXI"] + sector_etfs + [cash_etf])

    print("\n=== Latest Feature Importance Summary ===")
    print(latest["feature_importance"].head(25).to_string(index=False))


def save_latest(prefix: str, latest: dict):
    if latest is None:
        return
    latest_df = pd.DataFrame([{
        "signal_date": latest["date"],
        "latest_data_date": prices.index[-1],
        "feature_date": latest["feature_date"],
        "yield_curve": latest["yield_curve"],
        "vix_level": latest["vix_level"],
        "top_asset": latest["top_asset"],
        "second_asset": latest["second_asset"],
        "top_score": latest["top_score"],
        "second_score": latest["second_score"],
        "score_gap": latest["score_gap"],
        "overlay_fraction": latest["overlay_fraction"],
        "overlay_style": latest.get("overlay_style", "v1"),
        "v2_tqqq_scale": latest.get("v2_tqqq_scale", 1.0),
        "v2_alert_action": latest.get("v2_alert_action", "NONE"),
        "war_strength": latest["war_strength"],
        "growth_strength": latest["growth_strength"],
        "risk_off_strength": latest["risk_off_strength"],
        "soxx_strength": latest["soxx_strength"],
        "copper_strength": latest["copper_strength"],
        "copper_3m_strength": latest["copper_3m_strength"],
        "industrial_strength": latest["industrial_strength"],
        "materials_strength": latest["materials_strength"],
        "usd_3m_strength": latest["usd_3m_strength"],
        "hyg_strength": latest["hyg_strength"],
        "credit_strength": latest["credit_strength"],
        "tech_real_economy_divergence": latest["tech_real_economy_divergence"],
        "crash_pressure": latest["crash_pressure"],
        "conditional_breakdown_defense_level": latest.get("conditional_breakdown_defense_level", "normal"),
        **{f"signal_w_{k}": v for k, v in latest["signal_weights"].items()},
        **{f"exec_w_{k}": v for k, v in latest["exec_weights"].items()},
        **{f"raw_pred_{k}": v for k, v in latest["raw_predictions"].items()},
        **{f"adj_pred_{k}": v for k, v in latest["adjusted_predictions"].items()},
    }])
    latest_df.to_csv(f"{prefix}_latest_recommendation.csv", index=False)
    latest["feature_importance"].to_csv(f"{prefix}_feature_importance.csv", index=False)

                                        
def build_expected_return_calibration(rebalance_df: pd.DataFrame, sector_etfs: list) -> tuple[pd.DataFrame, np.ndarray]:
    if rebalance_df is None or rebalance_df.empty:
        return pd.DataFrame(), np.array([])

    rows = []
    forward_returns = {
        asset: prices[asset].shift(-forward_return_days) / prices[asset] - 1.0
        for asset in sector_etfs
        if asset in prices.columns
    }

    for _, row in rebalance_df.iterrows():
        date = pd.to_datetime(row.get("date"))
        if date not in prices.index:
            continue
        top_asset = row.get("top_asset")
        second_asset = row.get("second_asset")
        for asset in sector_etfs:
            pred_key = f"adj_pred_{asset}"
            if pred_key not in row or asset not in forward_returns:
                continue
            pred = row.get(pred_key)
            realized = forward_returns[asset].get(date, np.nan)
            if pd.isna(pred) or pd.isna(realized):
                continue
            rows.append({
                "date": date,
                "asset": asset,
                "adjusted_expected_return": float(pred),
                "realized_next_10d_return": float(realized),
                "was_top_asset": asset == top_asset,
                "was_second_asset": asset == second_asset,
            })

    history = pd.DataFrame(rows)
    if history.empty:
        return pd.DataFrame(), np.array([])

    try:
        deciles, bins = pd.qcut(
            history["adjusted_expected_return"],
            q=10,
            labels=False,
            duplicates="drop",
            retbins=True,
        )
        history["score_decile"] = deciles.astype(float) + 1
    except Exception:
        bins = np.array([])
        history["score_decile"] = np.nan

    calibration = (
        history.dropna(subset=["score_decile"])
        .groupby("score_decile")
        .agg(
            sample_count=("realized_next_10d_return", "size"),
            avg_expected_return=("adjusted_expected_return", "mean"),
            avg_realized_return=("realized_next_10d_return", "mean"),
            median_realized_return=("realized_next_10d_return", "median"),
            hit_rate=("realized_next_10d_return", lambda s: float((s > 0).mean())),
            avg_error=("realized_next_10d_return", lambda s: float((s - history.loc[s.index, "adjusted_expected_return"]).mean())),
            top_asset_rate=("was_top_asset", "mean"),
        )
        .reset_index()
        .sort_values("score_decile")
    )

    calibration["score_decile"] = calibration["score_decile"].astype(int)
    history.to_csv("model_c_plus_full_universe_expected_return_history.csv", index=False)
    return calibration, bins


def build_expected_return_trading_scores(latest: dict, sector_etfs: list, calibration: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    if latest is None:
        return pd.DataFrame()

    preds = latest.get("adjusted_predictions", {})
    raw_preds = latest.get("raw_predictions", {})
    latest_date = pd.to_datetime(latest["date"])
    current = []

    for asset in sector_etfs:
        if asset not in preds or asset not in prices.columns:
            continue
        px = prices[asset].loc[:latest_date]
        daily = px.pct_change()
        vol_21d = daily.tail(21).std() * np.sqrt(252)
        vol_63d = daily.tail(63).std() * np.sqrt(252)
        ten_day_vol = vol_63d * np.sqrt(forward_return_days / 252.0)
        adjusted = float(preds.get(asset, np.nan))
        raw = float(raw_preds.get(asset, np.nan))
        risk_adjusted = adjusted / ten_day_vol if ten_day_vol and not pd.isna(ten_day_vol) and ten_day_vol > 0 else np.nan

        if len(bins) > 1 and not pd.isna(adjusted):
            decile = int(np.searchsorted(bins, adjusted, side="right"))
            decile = max(1, min(decile, len(bins) - 1))
        else:
            decile = np.nan

        cal_row = calibration[calibration["score_decile"] == decile] if not calibration.empty and not pd.isna(decile) else pd.DataFrame()
        if cal_row.empty:
            calibrated_return = np.nan
            hit_rate = np.nan
            sample_count = 0
        else:
            calibrated_return = float(cal_row.iloc[0]["avg_realized_return"])
            hit_rate = float(cal_row.iloc[0]["hit_rate"])
            sample_count = int(cal_row.iloc[0]["sample_count"])

        current.append({
            "signal_date": latest_date,
            "asset": asset,
            "raw_expected_10d_return": raw,
            "adjusted_expected_10d_return": adjusted,
            "calibrated_bucket_realized_10d_return": calibrated_return,
            "historical_bucket_hit_rate": hit_rate,
            "historical_bucket_sample_count": sample_count,
            "score_decile": decile,
            "annualized_vol_21d": vol_21d,
            "annualized_vol_63d": vol_63d,
            "expected_return_per_10d_vol": risk_adjusted,
            "is_production_universe": asset in UPGRADED_SECTOR_ETFS,
        })

    scores = pd.DataFrame(current)
    if scores.empty:
        return scores

    scores["expected_return_rank"] = scores["adjusted_expected_10d_return"].rank(ascending=False, method="min").astype(int)
    scores["risk_adjusted_rank"] = scores["expected_return_per_10d_vol"].rank(ascending=False, method="min").astype(int)
    expected_pct = scores["adjusted_expected_10d_return"].rank(pct=True)
    risk_pct = scores["expected_return_per_10d_vol"].rank(pct=True)
    reliability = scores["historical_bucket_hit_rate"].fillna(0.5).clip(0.0, 1.0)
    sample_scale = (scores["historical_bucket_sample_count"].fillna(0) / 30.0).clip(0.25, 1.0)
    scores["tradable_score_0_100"] = (
        100.0 * (0.55 * expected_pct + 0.35 * risk_pct + 0.10 * reliability) * sample_scale
    ).round(1)

    conditions = [
        (scores["historical_bucket_sample_count"] >= 30)
        & (scores["historical_bucket_hit_rate"] >= 0.55)
        & (scores["calibrated_bucket_realized_10d_return"] > 0),
        (scores["historical_bucket_sample_count"] >= 15)
        & (scores["historical_bucket_hit_rate"] >= 0.50),
    ]
    scores["confidence_label"] = np.select(conditions, ["HIGH", "MEDIUM"], default="LOW")
    return scores.sort_values(["tradable_score_0_100", "adjusted_expected_10d_return"], ascending=False)



def apply_v2_continuous_tqqq_alert(latest: dict) -> dict:
    """
    V2 execution overlay.
    Keeps model prediction unchanged.
    Only adjusts final TQQQ exposure into QQQM.
    """

    if latest is None:
        return latest

    exec_weights = latest["exec_weights"].copy()

    risk_off = float(latest.get("risk_off_strength", 0.0))
    growth = float(latest.get("growth_strength", 0.0))
    soxx = float(latest.get("soxx_strength", 0.0))
    score_gap = float(latest.get("score_gap", 0.0))
    top_score = float(latest.get("top_score", 0.0))

    if top_score <= 0 or risk_off >= 1.50 or exec_weights.get("BIL", 0.0) >= 0.99:
        for a in exec_weights:
            exec_weights[a] = 0.0
        exec_weights["BIL"] = 1.0

        latest["exec_weights"] = exec_weights
        latest["v2_tqqq_scale"] = 0.0
        latest["v2_alert_action"] = "HARD_EXIT_TO_BIL"
        return latest

    tqqq_scale = 1.0

    if risk_off > 0.30:
        tqqq_scale *= 0.85
    if risk_off > 0.50:
        tqqq_scale *= 0.70
    if risk_off > 0.75:
        tqqq_scale *= 0.50
    if risk_off > 1.00:
        tqqq_scale *= 0.35

    if soxx < 0.50:
        tqqq_scale *= 0.80
    if soxx < 0.00:
        tqqq_scale *= 0.50

    if growth < 0.50:
        tqqq_scale *= 0.80
    if growth < 0.00:
        tqqq_scale *= 0.50

    if score_gap < 0.010:
        tqqq_scale *= 0.85
    if score_gap < 0.003:
        tqqq_scale *= 0.70

    old_tqqq = exec_weights.get("TQQQ", 0.0)
    new_tqqq = old_tqqq * tqqq_scale
    moved_to_qqqm = old_tqqq - new_tqqq

    exec_weights["TQQQ"] = new_tqqq
    exec_weights["QQQM"] = exec_weights.get("QQQM", 0.0) + moved_to_qqqm

    if risk_off > 1.25:
        cash_add = 0.20
        for a in exec_weights:
            if a != "BIL":
                exec_weights[a] *= (1.0 - cash_add)
        exec_weights["BIL"] = exec_weights.get("BIL", 0.0) + cash_add

    total = sum(exec_weights.values())
    if total > 0:
        for a in exec_weights:
            exec_weights[a] /= total

    latest["exec_weights"] = exec_weights
    latest["v2_tqqq_scale"] = tqqq_scale
    latest["v2_alert_action"] = "V2_TQQQ_TO_QQQM_OVERLAY"

    return latest
# ============================================================
# 11. PRODUCTION RUN: CURRENT BEST MODEL + DIVERGENCE ALERTS
# ============================================================
# This script keeps your trading model unchanged.
#
# Important:
#   Divergence / SOXX breakdown is ALERT ONLY.
#   It does NOT automatically reduce TQQQ.
#
# Why:
#   Previous tests showed no-defense was best in the tested 2023-2026 period.
#   So divergence is useful as monitoring information, not yet proven as a trading rule.


PRODUCTION_PREFIX = "model_c_plus_current_best_with_divergence_alerts"

# Make sure conditional defense is OFF.
# We still calculate divergence and breakdown variables for monitoring.
use_conditional_breakdown_defense = False


def classify_divergence_alert(latest: dict) -> str:
    """
    Alert-only classification.
    It does not change weights.
    """
    divergence = float(latest.get("tech_real_economy_divergence", 0.0))
    breakdown_score = float(latest.get("breakdown_score", 0.0))
    risk_off = float(latest.get("risk_off_strength", 0.0))
    soxx = float(latest.get("soxx_strength", 0.0))
    growth = float(latest.get("growth_strength", 0.0))

    # Highest concern: divergence + actual tech breakdown + risk-off rising.
    if divergence >= 3.0 and breakdown_score >= 2 and risk_off >= 0.0:
        return "DANGER: high divergence + SOXX/QQQM breakdown + risk-off rising"

    if divergence >= 2.5 and breakdown_score >= 1 and risk_off >= -0.25:
        return "WARNING: high divergence + early SOXX/QQQM weakness"

    if divergence >= 2.5 and soxx > 1.0 and growth > 1.0:
        return "WATCH: narrow tech-led rally; no action unless SOXX breaks"

    if breakdown_score >= 2 and risk_off >= 0.0:
        return "WARNING: tech breakdown pressure, but divergence not extreme"

    return "NORMAL"


def build_alert_row(latest: dict) -> pd.DataFrame:
    """
    Save a compact alert dashboard CSV.
    """
    row = {
        "signal_date": latest["date"],
        "latest_data_date": prices.index[-1],
        "top_asset": latest["top_asset"],
        "second_asset": latest["second_asset"],
        "top_score": latest["top_score"],
        "second_score": latest["second_score"],
        "score_gap": latest["score_gap"],
        "overlay_fraction": latest["overlay_fraction"],

        "alert_level": classify_divergence_alert(latest),

        "growth_strength": latest.get("growth_strength", np.nan),
        "soxx_strength": latest.get("soxx_strength", np.nan),
        "risk_off_strength": latest.get("risk_off_strength", np.nan),
        "industrial_strength": latest.get("industrial_strength", np.nan),
        "materials_strength": latest.get("materials_strength", np.nan),
        "copper_strength": latest.get("copper_strength", np.nan),
        "credit_strength": latest.get("credit_strength", np.nan),

        "tech_real_economy_divergence": latest.get("tech_real_economy_divergence", np.nan),
        "crash_pressure": latest.get("crash_pressure", np.nan),
        "breakdown_score": latest.get("breakdown_score", np.nan),

        "soxx_5d": latest.get("soxx_5d", np.nan),
        "soxx_10d": latest.get("soxx_10d", np.nan),
        "soxx_dd_21": latest.get("soxx_dd_21", np.nan),
        "qqqm_5d": latest.get("qqqm_5d", np.nan),
        "qqqm_10d": latest.get("qqqm_10d", np.nan),
        "qqqm_dd_21": latest.get("qqqm_dd_21", np.nan),

        **{f"signal_w_{k}": v for k, v in latest["signal_weights"].items()},
        **{f"exec_w_{k}": v for k, v in latest["exec_weights"].items()},
        **{f"raw_pred_{k}": v for k, v in latest["raw_predictions"].items()},
        **{f"adj_pred_{k}": v for k, v in latest["adjusted_predictions"].items()},
    }
    return pd.DataFrame([row])


def print_alert_dashboard(latest: dict):
    print("\n========================")
    print("DIVERGENCE / BREAKDOWN ALERT DASHBOARD")
    print("========================")
    print("Alert:", classify_divergence_alert(latest))
    print(f"Tech/real-economy divergence: {latest.get('tech_real_economy_divergence', np.nan):.3f}")
    print(f"Crash pressure:               {latest.get('crash_pressure', np.nan):.3f}")
    print(f"Breakdown score:              {latest.get('breakdown_score', np.nan):.1f}")
    print(f"SOXX 5d:                      {latest.get('soxx_5d', np.nan):.3f}")
    print(f"SOXX 10d:                     {latest.get('soxx_10d', np.nan):.3f}")
    print(f"SOXX 21d drawdown:            {latest.get('soxx_dd_21', np.nan):.3f}")
    print(f"QQQM 5d:                      {latest.get('qqqm_5d', np.nan):.3f}")
    print(f"QQQM 10d:                     {latest.get('qqqm_10d', np.nan):.3f}")
    print(f"QQQM 21d drawdown:            {latest.get('qqqm_dd_21', np.nan):.3f}")

    print("\nInterpretation:")
    print("- NORMAL: no special warning.")
    print("- WATCH: divergence is high, but tech still leads. Monitor only.")
    print("- WARNING: divergence plus early SOXX/QQQM weakness. Be careful with new TQQQ buys.")
    print("- DANGER: divergence plus breakdown plus risk-off. Consider manual risk reduction.")


# ============================================================
# 12. RUN CURRENT BEST MODEL
# ============================================================
print("\nBuilding upgraded feature set...")
upgraded_features = build_features_by_asset(UPGRADED_SECTOR_ETFS)

print("\nRunning current best production model...")
current_returns, current_rebalance, current_turnover = run_strategy(
    "CURRENT_BEST_WITH_DIVERGENCE_ALERTS",
    UPGRADED_SECTOR_ETFS,
    upgraded_features,
    overlay_style="hybrid",
    tqqq_style="dynamic",
)

if len(current_returns) == 0:
    raise ValueError("Current best model produced no returns.")

summary_df = pd.DataFrame([
    performance_summary("CURRENT_BEST_WITH_DIVERGENCE_ALERTS", current_returns, current_turnover)
])

print("\n=== CURRENT BEST PERFORMANCE SUMMARY ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

latest = get_latest_recommendation(
    "CURRENT_BEST_WITH_DIVERGENCE_ALERTS",
    UPGRADED_SECTOR_ETFS,
    upgraded_features,
    overlay_style="hybrid",
    tqqq_style="dynamic",
)

latest = apply_v2_continuous_tqqq_alert(latest)
print_latest(latest, UPGRADED_SECTOR_ETFS)
print_alert_dashboard(latest)

alert_df = build_alert_row(latest)

# ============================================================
# 13. SAVE OUTPUTS
# ============================================================
current_returns.to_csv(f"{PRODUCTION_PREFIX}_portfolio_daily_returns.csv", header=["portfolio_return"])
current_rebalance.to_csv(f"{PRODUCTION_PREFIX}_rebalance_log.csv", index=False)
summary_df.to_csv(f"{PRODUCTION_PREFIX}_performance_summary.csv", index=False)
save_latest(PRODUCTION_PREFIX, latest)
alert_df.to_csv(f"{PRODUCTION_PREFIX}_alert_dashboard.csv", index=False)

print("\nSaved:")
print(f"- {PRODUCTION_PREFIX}_portfolio_daily_returns.csv")
print(f"- {PRODUCTION_PREFIX}_rebalance_log.csv")
print(f"- {PRODUCTION_PREFIX}_performance_summary.csv")
print(f"- {PRODUCTION_PREFIX}_latest_recommendation.csv")
print(f"- {PRODUCTION_PREFIX}_feature_importance.csv")
print(f"- {PRODUCTION_PREFIX}_alert_dashboard.csv")

if run_full_expected_return_study:
    full_expected_prefix = "model_c_plus_full_universe_expected_returns"
    expanded_candidate_prefix = "model_c_plus_expanded_execution_candidate"

    print("\nBuilding full-universe expected-return feature set...")
    full_expected_features = build_features_by_asset(FULL_EXPECTED_RETURN_ETFS)

    print("\nBacktesting full-universe expected-return strategy...")
    full_expected_returns, full_expected_rebalance, full_expected_turnover = run_strategy(
        "FULL_UNIVERSE_EXPECTED_RETURNS_RESEARCH",
        FULL_EXPECTED_RETURN_ETFS,
        full_expected_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
    )

    if len(full_expected_returns) == 0:
        raise ValueError("Full-universe expected-return model produced no returns.")

    full_expected_summary = pd.DataFrame([
        performance_summary("FULL_UNIVERSE_EXPECTED_RETURNS_RESEARCH", full_expected_returns, full_expected_turnover)
    ])
    current_common = current_returns.loc[current_returns.index.intersection(full_expected_returns.index)]
    full_expected_common = full_expected_returns.loc[current_common.index]
    full_expected_compare = pd.DataFrame([
        performance_summary("CURRENT_BEST_COMMON", current_common, current_turnover),
        performance_summary("FULL_UNIVERSE_EXPECTED_RETURNS_COMMON", full_expected_common, full_expected_turnover),
    ])

    print("\n=== FULL-UNIVERSE EXPECTED-RETURN PERFORMANCE SUMMARY ===")
    print(full_expected_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== FULL-UNIVERSE COMMON-PERIOD COMPARISON ===")
    print(full_expected_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    calibration, calibration_bins = build_expected_return_calibration(full_expected_rebalance, FULL_EXPECTED_RETURN_ETFS)

    print("\nRunning full-universe expected-return latest score...")
    latest_full_expected = get_latest_recommendation(
        "FULL_UNIVERSE_EXPECTED_RETURNS_RESEARCH",
        FULL_EXPECTED_RETURN_ETFS,
        full_expected_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
    )

    print_latest(latest_full_expected, FULL_EXPECTED_RETURN_ETFS)
    trading_scores = build_expected_return_trading_scores(
        latest_full_expected,
        FULL_EXPECTED_RETURN_ETFS,
        calibration,
        calibration_bins,
    )

    print("\n=== FULL-UNIVERSE TRADABLE SCOREBOARD ===")
    if trading_scores.empty:
        print("No trading scores available.")
    else:
        print(trading_scores.head(16).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    full_expected_returns.to_csv(f"{full_expected_prefix}_daily_returns.csv", header=["portfolio_return"])
    full_expected_rebalance.to_csv(f"{full_expected_prefix}_rebalance_log.csv", index=False)
    full_expected_summary.to_csv(f"{full_expected_prefix}_performance_summary.csv", index=False)
    full_expected_compare.to_csv(f"{full_expected_prefix}_compare_current_best.csv", index=False)
    calibration.to_csv(f"{full_expected_prefix}_calibration.csv", index=False)
    trading_scores.to_csv(f"{full_expected_prefix}_trading_scores.csv", index=False)
    save_latest(full_expected_prefix, latest_full_expected)

    expanded_candidate_returns = full_expected_returns.copy()
    expanded_candidate_rebalance = full_expected_rebalance.copy()
    expanded_candidate_summary = full_expected_summary.copy()
    expanded_candidate_compare = full_expected_compare.copy()
    expanded_candidate_latest = latest_full_expected.copy()

    expanded_candidate_summary["model"] = "EXPANDED_EXECUTION_CANDIDATE_SHADOW"
    expanded_candidate_compare["model"] = expanded_candidate_compare["model"].replace({
        "FULL_UNIVERSE_EXPECTED_RETURNS_COMMON": "EXPANDED_EXECUTION_CANDIDATE_COMMON",
    })
    expanded_candidate_latest["model"] = "EXPANDED_EXECUTION_CANDIDATE_SHADOW"
    if not expanded_candidate_rebalance.empty and "model" in expanded_candidate_rebalance.columns:
        expanded_candidate_rebalance["model"] = "EXPANDED_EXECUTION_CANDIDATE_SHADOW"

    expanded_candidate_returns.to_csv(f"{expanded_candidate_prefix}_daily_returns.csv", header=["portfolio_return"])
    expanded_candidate_rebalance.to_csv(f"{expanded_candidate_prefix}_rebalance_log.csv", index=False)
    expanded_candidate_summary.to_csv(f"{expanded_candidate_prefix}_performance_summary.csv", index=False)
    expanded_candidate_compare.to_csv(f"{expanded_candidate_prefix}_compare_current_best.csv", index=False)
    trading_scores.to_csv(f"{expanded_candidate_prefix}_trading_scores.csv", index=False)
    calibration.to_csv(f"{expanded_candidate_prefix}_calibration.csv", index=False)
    save_latest(expanded_candidate_prefix, expanded_candidate_latest)

    print("\nSaved full-universe expected-return outputs:")
    print(f"- {full_expected_prefix}_daily_returns.csv")
    print(f"- {full_expected_prefix}_rebalance_log.csv")
    print(f"- {full_expected_prefix}_performance_summary.csv")
    print(f"- {full_expected_prefix}_compare_current_best.csv")
    print(f"- {full_expected_prefix}_calibration.csv")
    print(f"- {full_expected_prefix}_trading_scores.csv")
    print(f"- {full_expected_prefix}_latest_recommendation.csv")
    print(f"- {full_expected_prefix}_feature_importance.csv")
    print("\nSaved expanded execution candidate outputs:")
    print(f"- {expanded_candidate_prefix}_daily_returns.csv")
    print(f"- {expanded_candidate_prefix}_rebalance_log.csv")
    print(f"- {expanded_candidate_prefix}_performance_summary.csv")
    print(f"- {expanded_candidate_prefix}_compare_current_best.csv")
    print(f"- {expanded_candidate_prefix}_calibration.csv")
    print(f"- {expanded_candidate_prefix}_trading_scores.csv")
    print(f"- {expanded_candidate_prefix}_latest_recommendation.csv")
    print(f"- {expanded_candidate_prefix}_feature_importance.csv")

if run_soxx_012a_test:
    soxx_prefix = "model_c_plus_soxx_leverage_012A"

    print("\nBuilding 012A SOXX feature set...")
    soxx_features = build_features_by_asset(SOXX_012A_SECTOR_ETFS)

    print("\nRunning 012A SOXX + SOXL research model...")
    soxx_returns, soxx_rebalance, soxx_turnover = run_strategy(
        "SOXX_LEVERAGE_012A_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
    )

    if len(soxx_returns) == 0:
        raise ValueError("012A SOXX model produced no returns.")

    soxx_summary = pd.DataFrame([
        performance_summary("SOXX_LEVERAGE_012A_RESEARCH", soxx_returns, soxx_turnover)
    ])

    current_common = current_returns.loc[current_returns.index.intersection(soxx_returns.index)]
    soxx_common = soxx_returns.loc[current_common.index]
    soxx_compare = pd.DataFrame([
        performance_summary("CURRENT_BEST_COMMON", current_common, current_turnover),
        performance_summary("SOXX_LEVERAGE_012A_COMMON", soxx_common, soxx_turnover),
    ])

    latest_soxx = get_latest_recommendation(
        "SOXX_LEVERAGE_012A_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
    )
    latest_soxx = apply_v2_continuous_tqqq_alert(latest_soxx)

    print("\n=== 012A SOXX + SOXL PERFORMANCE SUMMARY ===")
    print(soxx_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 012A COMMON-PERIOD COMPARISON ===")
    print(soxx_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_latest(latest_soxx, SOXX_012A_SECTOR_ETFS)

    soxx_returns.to_csv(f"{soxx_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
    soxx_rebalance.to_csv(f"{soxx_prefix}_rebalance_log.csv", index=False)
    soxx_summary.to_csv(f"{soxx_prefix}_performance_summary.csv", index=False)
    soxx_compare.to_csv(f"{soxx_prefix}_compare_current_best.csv", index=False)
    subperiod_diagnostics("SOXX_LEVERAGE_012A_RESEARCH", soxx_returns).to_csv(
        f"{soxx_prefix}_subperiod_diagnostics.csv", index=False
    )
    exposure_diagnostics("SOXX_LEVERAGE_012A_RESEARCH", soxx_rebalance).to_csv(
        f"{soxx_prefix}_exposure_diagnostics.csv", index=False
    )
    save_latest(soxx_prefix, latest_soxx)

    print("\nSaved 012A research outputs:")
    print(f"- {soxx_prefix}_portfolio_daily_returns.csv")
    print(f"- {soxx_prefix}_rebalance_log.csv")
    print(f"- {soxx_prefix}_performance_summary.csv")
    print(f"- {soxx_prefix}_compare_current_best.csv")
    print(f"- {soxx_prefix}_subperiod_diagnostics.csv")
    print(f"- {soxx_prefix}_exposure_diagnostics.csv")
    print(f"- {soxx_prefix}_latest_recommendation.csv")
    print(f"- {soxx_prefix}_feature_importance.csv")

if run_soxx_012b_test:
    soxx_strict_prefix = "model_c_plus_soxx_strict_admission_012B"

    print("\nBuilding 012B SOXX strict-admission feature set...")
    soxx_strict_features = build_features_by_asset(SOXX_012A_SECTOR_ETFS)

    print("\nRunning 012B SOXX strict-admission research model...")
    soxx_strict_returns, soxx_strict_rebalance, soxx_strict_turnover = run_strategy(
        "SOXX_STRICT_ADMISSION_012B_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_strict_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_admission_mode="strict",
    )

    if len(soxx_strict_returns) == 0:
        raise ValueError("012B SOXX strict-admission model produced no returns.")

    soxx_strict_summary = pd.DataFrame([
        performance_summary("SOXX_STRICT_ADMISSION_012B_RESEARCH", soxx_strict_returns, soxx_strict_turnover)
    ])

    current_common = current_returns.loc[current_returns.index.intersection(soxx_strict_returns.index)]
    soxx_strict_common = soxx_strict_returns.loc[current_common.index]
    soxx_strict_compare = pd.DataFrame([
        performance_summary("CURRENT_BEST_COMMON", current_common, current_turnover),
        performance_summary("SOXX_STRICT_ADMISSION_012B_COMMON", soxx_strict_common, soxx_strict_turnover),
    ])

    latest_soxx_strict = get_latest_recommendation(
        "SOXX_STRICT_ADMISSION_012B_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_strict_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_admission_mode="strict",
    )
    latest_soxx_strict = apply_v2_continuous_tqqq_alert(latest_soxx_strict)

    print("\n=== 012B SOXX STRICT-ADMISSION PERFORMANCE SUMMARY ===")
    print(soxx_strict_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 012B COMMON-PERIOD COMPARISON ===")
    print(soxx_strict_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_latest(latest_soxx_strict, SOXX_012A_SECTOR_ETFS)

    soxx_strict_returns.to_csv(f"{soxx_strict_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
    soxx_strict_rebalance.to_csv(f"{soxx_strict_prefix}_rebalance_log.csv", index=False)
    soxx_strict_summary.to_csv(f"{soxx_strict_prefix}_performance_summary.csv", index=False)
    soxx_strict_compare.to_csv(f"{soxx_strict_prefix}_compare_current_best.csv", index=False)
    save_latest(soxx_strict_prefix, latest_soxx_strict)

    print("\nSaved 012B research outputs:")
    print(f"- {soxx_strict_prefix}_portfolio_daily_returns.csv")
    print(f"- {soxx_strict_prefix}_rebalance_log.csv")
    print(f"- {soxx_strict_prefix}_performance_summary.csv")
    print(f"- {soxx_strict_prefix}_compare_current_best.csv")
    print(f"- {soxx_strict_prefix}_latest_recommendation.csv")
    print(f"- {soxx_strict_prefix}_feature_importance.csv")

if run_soxx_012c_test:
    soxx_hurdle_prefix = "model_c_plus_soxx_hurdle_012C"

    print("\nBuilding 012C SOXX hurdle feature set...")
    soxx_hurdle_features = build_features_by_asset(SOXX_012A_SECTOR_ETFS)

    print("\nRunning 012C SOXX hurdle research model...")
    soxx_hurdle_returns, soxx_hurdle_rebalance, soxx_hurdle_turnover = run_strategy(
        "SOXX_HURDLE_012C_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_hurdle_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_admission_mode="hurdle",
    )

    if len(soxx_hurdle_returns) == 0:
        raise ValueError("012C SOXX hurdle model produced no returns.")

    soxx_hurdle_summary = pd.DataFrame([
        performance_summary("SOXX_HURDLE_012C_RESEARCH", soxx_hurdle_returns, soxx_hurdle_turnover)
    ])

    current_common = current_returns.loc[current_returns.index.intersection(soxx_hurdle_returns.index)]
    soxx_hurdle_common = soxx_hurdle_returns.loc[current_common.index]
    soxx_hurdle_compare = pd.DataFrame([
        performance_summary("CURRENT_BEST_COMMON", current_common, current_turnover),
        performance_summary("SOXX_HURDLE_012C_COMMON", soxx_hurdle_common, soxx_hurdle_turnover),
    ])

    latest_soxx_hurdle = get_latest_recommendation(
        "SOXX_HURDLE_012C_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_hurdle_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_admission_mode="hurdle",
    )
    latest_soxx_hurdle = apply_v2_continuous_tqqq_alert(latest_soxx_hurdle)

    print("\n=== 012C SOXX HURDLE PERFORMANCE SUMMARY ===")
    print(soxx_hurdle_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 012C COMMON-PERIOD COMPARISON ===")
    print(soxx_hurdle_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_latest(latest_soxx_hurdle, SOXX_012A_SECTOR_ETFS)

    soxx_hurdle_returns.to_csv(f"{soxx_hurdle_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
    soxx_hurdle_rebalance.to_csv(f"{soxx_hurdle_prefix}_rebalance_log.csv", index=False)
    soxx_hurdle_summary.to_csv(f"{soxx_hurdle_prefix}_performance_summary.csv", index=False)
    soxx_hurdle_compare.to_csv(f"{soxx_hurdle_prefix}_compare_current_best.csv", index=False)
    save_latest(soxx_hurdle_prefix, latest_soxx_hurdle)

    print("\nSaved 012C research outputs:")
    print(f"- {soxx_hurdle_prefix}_portfolio_daily_returns.csv")
    print(f"- {soxx_hurdle_prefix}_rebalance_log.csv")
    print(f"- {soxx_hurdle_prefix}_performance_summary.csv")
    print(f"- {soxx_hurdle_prefix}_compare_current_best.csv")
    print(f"- {soxx_hurdle_prefix}_latest_recommendation.csv")
    print(f"- {soxx_hurdle_prefix}_feature_importance.csv")

if run_soxx_012d_test:
    soxx_defense_prefix = "model_c_plus_soxx_execution_defense_012D"

    print("\nBuilding 012D SOXX execution-defense feature set...")
    soxx_defense_features = build_features_by_asset(SOXX_012A_SECTOR_ETFS)

    print("\nRunning 012D SOXX execution-defense research model...")
    soxx_defense_returns, soxx_defense_rebalance, soxx_defense_turnover = run_strategy(
        "SOXX_EXECUTION_DEFENSE_012D_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_defense_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_execution_mode="semi_defense",
    )

    if len(soxx_defense_returns) == 0:
        raise ValueError("012D SOXX execution-defense model produced no returns.")

    soxx_defense_summary = pd.DataFrame([
        performance_summary("SOXX_EXECUTION_DEFENSE_012D_RESEARCH", soxx_defense_returns, soxx_defense_turnover)
    ])

    current_common = current_returns.loc[current_returns.index.intersection(soxx_defense_returns.index)]
    soxx_defense_common = soxx_defense_returns.loc[current_common.index]
    soxx_defense_compare = pd.DataFrame([
        performance_summary("CURRENT_BEST_COMMON", current_common, current_turnover),
        performance_summary("SOXX_EXECUTION_DEFENSE_012D_COMMON", soxx_defense_common, soxx_defense_turnover),
    ])

    latest_soxx_defense = get_latest_recommendation(
        "SOXX_EXECUTION_DEFENSE_012D_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_defense_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_execution_mode="semi_defense",
    )
    latest_soxx_defense = apply_v2_continuous_tqqq_alert(latest_soxx_defense)

    print("\n=== 012D SOXX EXECUTION-DEFENSE PERFORMANCE SUMMARY ===")
    print(soxx_defense_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 012D COMMON-PERIOD COMPARISON ===")
    print(soxx_defense_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_latest(latest_soxx_defense, SOXX_012A_SECTOR_ETFS)

    soxx_defense_returns.to_csv(f"{soxx_defense_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
    soxx_defense_rebalance.to_csv(f"{soxx_defense_prefix}_rebalance_log.csv", index=False)
    soxx_defense_summary.to_csv(f"{soxx_defense_prefix}_performance_summary.csv", index=False)
    soxx_defense_compare.to_csv(f"{soxx_defense_prefix}_compare_current_best.csv", index=False)
    save_latest(soxx_defense_prefix, latest_soxx_defense)

    print("\nSaved 012D research outputs:")
    print(f"- {soxx_defense_prefix}_portfolio_daily_returns.csv")
    print(f"- {soxx_defense_prefix}_rebalance_log.csv")
    print(f"- {soxx_defense_prefix}_performance_summary.csv")
    print(f"- {soxx_defense_prefix}_compare_current_best.csv")
    print(f"- {soxx_defense_prefix}_latest_recommendation.csv")
    print(f"- {soxx_defense_prefix}_feature_importance.csv")

if run_soxx_012e_test:
    soxx_mild_prefix = "model_c_plus_soxx_mild_soxl_012E"

    print("\nBuilding 012E mild SOXL feature set...")
    soxx_mild_features = build_features_by_asset(SOXX_012A_SECTOR_ETFS)

    print("\nRunning 012E mild SOXL research model...")
    soxx_mild_returns, soxx_mild_rebalance, soxx_mild_turnover = run_strategy(
        "SOXX_MILD_SOXL_012E_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_mild_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_leverage_mode="mild",
    )

    if len(soxx_mild_returns) == 0:
        raise ValueError("012E mild SOXL model produced no returns.")

    soxx_mild_summary = pd.DataFrame([
        performance_summary("SOXX_MILD_SOXL_012E_RESEARCH", soxx_mild_returns, soxx_mild_turnover)
    ])

    current_common = current_returns.loc[current_returns.index.intersection(soxx_mild_returns.index)]
    soxx_mild_common = soxx_mild_returns.loc[current_common.index]
    soxx_mild_compare = pd.DataFrame([
        performance_summary("CURRENT_BEST_COMMON", current_common, current_turnover),
        performance_summary("SOXX_MILD_SOXL_012E_COMMON", soxx_mild_common, soxx_mild_turnover),
    ])

    latest_soxx_mild = get_latest_recommendation(
        "SOXX_MILD_SOXL_012E_RESEARCH",
        SOXX_012A_SECTOR_ETFS,
        soxx_mild_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
        soxx_leverage_mode="mild",
    )
    latest_soxx_mild = apply_v2_continuous_tqqq_alert(latest_soxx_mild)

    print("\n=== 012E MILD SOXL PERFORMANCE SUMMARY ===")
    print(soxx_mild_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 012E COMMON-PERIOD COMPARISON ===")
    print(soxx_mild_compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_latest(latest_soxx_mild, SOXX_012A_SECTOR_ETFS)

    soxx_mild_returns.to_csv(f"{soxx_mild_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
    soxx_mild_rebalance.to_csv(f"{soxx_mild_prefix}_rebalance_log.csv", index=False)
    soxx_mild_summary.to_csv(f"{soxx_mild_prefix}_performance_summary.csv", index=False)
    soxx_mild_compare.to_csv(f"{soxx_mild_prefix}_compare_current_best.csv", index=False)
    subperiod_diagnostics("SOXX_MILD_SOXL_012E_RESEARCH", soxx_mild_returns).to_csv(
        f"{soxx_mild_prefix}_subperiod_diagnostics.csv", index=False
    )
    exposure_diagnostics("SOXX_MILD_SOXL_012E_RESEARCH", soxx_mild_rebalance).to_csv(
        f"{soxx_mild_prefix}_exposure_diagnostics.csv", index=False
    )
    save_latest(soxx_mild_prefix, latest_soxx_mild)

    print("\nSaved 012E research outputs:")
    print(f"- {soxx_mild_prefix}_portfolio_daily_returns.csv")
    print(f"- {soxx_mild_prefix}_rebalance_log.csv")
    print(f"- {soxx_mild_prefix}_performance_summary.csv")
    print(f"- {soxx_mild_prefix}_compare_current_best.csv")
    print(f"- {soxx_mild_prefix}_subperiod_diagnostics.csv")
    print(f"- {soxx_mild_prefix}_exposure_diagnostics.csv")
    print(f"- {soxx_mild_prefix}_latest_recommendation.csv")
    print(f"- {soxx_mild_prefix}_feature_importance.csv")

if run_expanded_012_test:
    expanded_prefix = "model_c_plus_universe_expansion_012"

    print("\nBuilding 012 expanded universe feature set...")
    expanded_features = build_features_by_asset(EXPANDED_012_SECTOR_ETFS)

    print("\nRunning 012 expanded universe research model...")
    expanded_returns, expanded_rebalance, expanded_turnover = run_strategy(
        "UNIVERSE_EXPANSION_012_RESEARCH",
        EXPANDED_012_SECTOR_ETFS,
        expanded_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
    )

    if len(expanded_returns) == 0:
        raise ValueError("012 expanded universe model produced no returns.")

    expanded_summary = pd.DataFrame([
        performance_summary("UNIVERSE_EXPANSION_012_RESEARCH", expanded_returns, expanded_turnover)
    ])

    current_common = current_returns.loc[current_returns.index.intersection(expanded_returns.index)]
    expanded_common = expanded_returns.loc[current_common.index]
    compare_summary = pd.DataFrame([
        performance_summary("CURRENT_BEST_COMMON", current_common, current_turnover),
        performance_summary("UNIVERSE_EXPANSION_012_COMMON", expanded_common, expanded_turnover),
    ])

    latest_expanded = get_latest_recommendation(
        "UNIVERSE_EXPANSION_012_RESEARCH",
        EXPANDED_012_SECTOR_ETFS,
        expanded_features,
        overlay_style="hybrid",
        tqqq_style="dynamic",
    )
    latest_expanded = apply_v2_continuous_tqqq_alert(latest_expanded)

    print("\n=== 012 EXPANDED UNIVERSE PERFORMANCE SUMMARY ===")
    print(expanded_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 012 COMMON-PERIOD COMPARISON ===")
    print(compare_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_latest(latest_expanded, EXPANDED_012_SECTOR_ETFS)

    expanded_returns.to_csv(f"{expanded_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
    expanded_rebalance.to_csv(f"{expanded_prefix}_rebalance_log.csv", index=False)
    expanded_summary.to_csv(f"{expanded_prefix}_performance_summary.csv", index=False)
    compare_summary.to_csv(f"{expanded_prefix}_compare_current_best.csv", index=False)
    save_latest(expanded_prefix, latest_expanded)

    print("\nSaved 012 research outputs:")
    print(f"- {expanded_prefix}_portfolio_daily_returns.csv")
    print(f"- {expanded_prefix}_rebalance_log.csv")
    print(f"- {expanded_prefix}_performance_summary.csv")
    print(f"- {expanded_prefix}_compare_current_best.csv")
    print(f"- {expanded_prefix}_latest_recommendation.csv")
    print(f"- {expanded_prefix}_feature_importance.csv")
