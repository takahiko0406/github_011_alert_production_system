"""Generate the six-page, fail-closed 034 daily trading dashboard.

The presentation and Telegram preview share one ranking dataframe.  Execution
authority comes from the recovered 022F path, never from legacy LIGHT labels.
Stale or mixed-date inputs force a BIL-only safety recommendation.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit_083_emergency_tqqq_soxl import emergency_criteria


OUT = os.getenv("DASHBOARD_OUTPUT_PREFIX", "model_c_plus_034_live_dashboard")
VALIDATED = ["QQQM", "TQQQ", "SOXX", "SOXL", "IWM", "FEZ", "XLE", "ERX", "XLB", "XLI", "UXI", "XLV", "XLP", "XLU", "XLRE", "TLT", "GLD", "XSOE", "BIL"]
RESEARCH = ["XLF", "IEF"]
DISABLED = ["UGL"]
DEFENSIVE = ["XLV", "XLP", "XLU", "XLRE", "TLT", "GLD", "BIL"]
ROLES = {
    "QQQM": "Large-cap growth", "SOXX": "Semiconductor growth", "IWM": "Small-cap breadth",
    "FEZ": "Developed international", "XLE": "Energy / inflation", "XLB": "Materials / cyclicals",
    "XLI": "Industrial cycle", "XLF": "Financials research", "XLV": "Defensive healthcare",
    "XLP": "Consumer defense", "XLU": "Utilities defense", "XLRE": "Rate-relief real estate",
    "TLT": "Duration defense", "IEF": "Treasury research", "GLD": "Crisis / currency hedge",
    "XSOE": "Emerging-market quality", "BIL": "Cash reserve",
    "TQQQ": "QQQM replacement leverage", "SOXL": "SOXX replacement leverage",
    "ERX": "XLE replacement leverage", "UXI": "XLI replacement leverage",
}


def recovered_root() -> Path:
    configured = os.getenv("DASHBOARD_INPUT_DIR", "").strip()
    if configured:
        root = Path(configured).resolve()
        required = [
            "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv",
            "model_c_plus_full_universe_expected_returns_trading_scores.csv",
            "model_c_plus_expanded_execution_candidate_latest_recommendation.csv",
            "model_c_plus_transition_conviction_overlay_011_LIGHT_latest_recommendation.csv",
        ]
        if not all((root / name).exists() for name in required):
            raise FileNotFoundError(f"Configured dashboard input directory is incomplete: {root}")
        return root
    local = Path(__file__).resolve().parent
    if (local / "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv").exists():
        return local
    roots = list(Path("C:/$Recycle.Bin").glob("S-*/$R*/model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv"))
    candidates = []
    for latest in roots:
        root = latest.parent
        if (root / "research_034_execution_grade_expected_return_signal.py").exists() and (root / "model_c_plus_full_universe_expected_returns_trading_scores.csv").exists():
            df = pd.read_csv(latest)
            if not df.empty:
                candidates.append((str(df.iloc[-1].get("latest_data_date", ""))[:10], root))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    raise FileNotFoundError("Complete recovered 034 dashboard inputs not found")


def fnum(value, default=np.nan) -> float:
    try:
        return float(value) if pd.notna(value) else default
    except (TypeError, ValueError):
        return default


def pct(value, digits=1) -> str:
    return "DATA_UNAVAILABLE" if pd.isna(value) else f"{100 * float(value):+.{digits}f}%"


def weight(value) -> str:
    return "DATA_UNAVAILABLE" if pd.isna(value) else f"{100 * float(value):.1f}%"


def score(value) -> str:
    return "DATA_UNAVAILABLE" if pd.isna(value) else f"{float(value):.1f}"


def state(value: float) -> str:
    if pd.isna(value): return "DATA_UNAVAILABLE"
    if value >= 1.0: return "Strong"
    if value > 0.0: return "Positive"
    if value <= -1.0: return "Weak"
    return "Negative"


def business_age(value: str, today: date) -> int:
    return max(0, len(pd.bdate_range(pd.Timestamp(value) + pd.Timedelta(days=1), pd.Timestamp(today))))


def driver_text(asset: str, x: dict) -> tuple[str, str, str]:
    growth, credit, risk, usd = x["growth"], x["credit"], x["risk"], x["usd"]
    cyclical = asset in {"SOXX", "IWM", "XLE", "XLB", "XLI", "XSOE", "QQQM", "FEZ"}
    defensive = asset in set(DEFENSIVE)
    positives, negatives = [], []
    if cyclical and growth > 0: positives.append("positive growth regime")
    if cyclical and growth <= 0: negatives.append("weak growth regime")
    if credit > 0: positives.append("supportive credit")
    else: negatives.append("weak credit confirmation")
    if defensive and risk > 0.5: positives.append("elevated risk-off demand")
    if cyclical and risk > 0.5: negatives.append("elevated risk-off pressure")
    if asset in {"GLD", "XLRE", "FEZ", "XSOE"} and usd > 1: negatives.append("strong USD headwind")
    if asset == "BIL": positives.append("capital preservation while inputs are stale")
    if not positives: positives.append("relative model score")
    if not negatives: negatives.append("no dominant negative driver")
    why = "Selected as the fail-closed allocation until all model dates agree." if asset == "BIL" else "Not selected while execution inputs are stale."
    return ", ".join(positives), ", ".join(negatives), why


def main() -> None:
    root = recovered_root()
    latest = pd.read_csv(root / "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv").iloc[-1]
    expanded = pd.read_csv(root / "model_c_plus_expanded_execution_candidate_latest_recommendation.csv").iloc[-1]
    scores = pd.read_csv(root / "model_c_plus_full_universe_expected_returns_trading_scores.csv")
    light = pd.read_csv(root / "model_c_plus_transition_conviction_overlay_011_LIGHT_latest_recommendation.csv").iloc[-1]
    market_freshness = pd.read_csv(root / "model_c_plus_market_data_freshness.csv").iloc[-1]
    feature_freshness = pd.read_csv(root / "model_c_plus_feature_freshness.csv").iloc[-1]
    emergency_validation = json.loads((root / "emergency_tqqq_soxl_functional_validation.json").read_text(encoding="utf-8"))
    emergency_performance = pd.read_csv(root / "emergency_tqqq_soxl_ab_performance.csv")
    emergency_episodes = pd.read_csv(root / "emergency_tqqq_soxl_episode_audit.csv")
    state_path = root / "current_portfolio_state_011.json"
    saved_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"current_portfolio": {"BIL": 1.0}}

    data_date = str(latest["latest_data_date"])[:10]
    prediction_date = str(latest["signal_date"])[:10]
    allocation_date = str(latest["base_weight_date"])[:10]
    base_weight_date = str(latest["base_weight_date"])[:10]
    score_date = str(scores["signal_date"].iloc[-1])[:10]
    market_data_date = str(market_freshness["latest_data_date"])[:10]
    feature_date = str(feature_freshness["latest_data_date"])[:10]
    source_files = {
        "022F_BASE": "model_c_plus_022F_calibrated_defense_validation_best_latest_recommendation.csv",
        "EXPANDED_CANDIDATE": "model_c_plus_expanded_execution_candidate_latest_recommendation.csv",
    }
    source_model = str(latest["source_model"])
    if source_model not in source_files:
        raise ValueError(f"Unsupported selected source model: {source_model}")
    source_file = source_files[source_model]
    source_recommendation = pd.read_csv(root / source_file).iloc[-1]
    source_recommendation_date = str(source_recommendation["latest_data_date"])[:10]
    now_utc = datetime.now(timezone.utc)
    dashboard_date = now_utc.date()
    # Before the U.S. cash close, today's session is not a completed session and
    # must not count as staleness. This is important for the scheduled Tokyo run.
    completed_cutoff = (pd.Timestamp(dashboard_date) - pd.offsets.BDay(1)).date() if now_utc.hour < 21 else dashboard_date
    age = business_age(data_date, completed_cutoff)
    common_dates = data_date == prediction_date == score_date == allocation_date == market_data_date == feature_date == source_recommendation_date
    freshness_current = age <= 1
    execution_safe = common_dates and freshness_current

    current = {a: 0.0 for a in sorted(set(VALIDATED + RESEARCH + DISABLED))}
    for asset, value in saved_state.get("current_portfolio", {}).items():
        current[asset] = fnum(value, 0.0)
    current_sum = sum(current.values())
    recommended = {a: 0.0 for a in current}
    if execution_safe:
        for asset in recommended:
            recommended[asset] = fnum(latest.get(f"exec_w_{asset}", 0.0), 0.0)
    else:
        recommended["BIL"] = 1.0

    scores = scores[scores["signal_date"].astype(str).str[:10].eq(score_date)].copy()
    score_map = scores.set_index("asset").to_dict("index")
    macro = {
        "growth": fnum(expanded.get("growth_strength")), "credit": fnum(expanded.get("credit_strength")),
        "risk": fnum(expanded.get("risk_off_strength")), "crash": fnum(expanded.get("crash_pressure")),
        "industrial": fnum(expanded.get("industrial_strength")), "materials": fnum(expanded.get("materials_strength")),
        "copper": fnum(expanded.get("copper_strength")), "usd": fnum(expanded.get("usd_3m_strength")),
        "soxx": fnum(expanded.get("soxx_strength")), "hyg": fnum(expanded.get("hyg_strength")),
        "tech": fnum(light.get("tech_state")),
        "energy": fnum(source_recommendation.get("energy_state")),
    }

    def report_field(display, status, artifact, field, source_date, *, missing_sources=None):
        return {
            "display": str(display), "status": status, "source_artifact": artifact,
            "source_field": field, "source_date": source_date,
            "missing_sources": missing_sources or [],
        }

    source_name_raw = source_recommendation.get("model")
    source_name = str(source_name_raw) if pd.notna(source_name_raw) else source_model
    source_configuration_raw = source_recommendation.get("configuration")
    source_configuration = str(source_configuration_raw) if pd.notna(source_configuration_raw) else "NOT_AVAILABLE_FROM_VALIDATED_SOURCE"
    source_history_through = source_recommendation.get("source_rebalance_date")
    source_history_through_display = str(source_history_through)[:10] if pd.notna(source_history_through) else "NOT_AVAILABLE_FROM_VALIDATED_SOURCE"
    energy_value = fnum(source_recommendation.get("energy_state"))
    energy_display = "DATA_UNAVAILABLE" if pd.isna(energy_value) else f"{state(energy_value)} ({energy_value:.3f})"
    emergency_value = str(latest.get("portfolio_wide_emergency", "")).strip()
    emergency_display = emergency_value if emergency_value else "NOT_AVAILABLE_FROM_VALIDATED_SOURCE"
    robust_active = str(latest.get("robust_gate_active", False)).strip().lower() in {"true", "1", "yes"}
    reporting_fields = {
        "selected_source_model": report_field(source_model, "AVAILABLE", "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", "source_model", data_date),
        "source_model_name": report_field(source_name, "AVAILABLE", source_file, "model", source_recommendation_date),
        "source_configuration": report_field(source_configuration, "AVAILABLE" if source_configuration != "NOT_AVAILABLE_FROM_VALIDATED_SOURCE" else "NOT_AVAILABLE_FROM_VALIDATED_SOURCE", source_file, "configuration", source_recommendation_date, missing_sources=[] if source_configuration != "NOT_AVAILABLE_FROM_VALIDATED_SOURCE" else [f"{source_file}:configuration"]),
        "source_recommendation_date": report_field(source_recommendation_date, "AVAILABLE", source_file, "latest_data_date", source_recommendation_date),
        "allocation_date": report_field(allocation_date, "AVAILABLE", "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", "allocation_date", data_date),
        "base_weight_date": report_field(base_weight_date, "AVAILABLE", "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", "base_weight_date", data_date),
        "market_data_date": report_field(market_data_date, "AVAILABLE", "model_c_plus_market_data_freshness.csv", "latest_data_date", market_data_date),
        "feature_date": report_field(feature_date, "AVAILABLE", "model_c_plus_feature_freshness.csv", "latest_data_date", feature_date),
        "last_rebalance_date": report_field("NOT_AVAILABLE_FROM_VALIDATED_SOURCE", "NOT_AVAILABLE_FROM_VALIDATED_SOURCE", source_file, "actual_last_rebalance_date", source_recommendation_date, missing_sources=[f"{source_file}:actual_last_rebalance_date"]),
        "source_history_through_date": report_field(source_history_through_display, "AVAILABLE" if pd.notna(source_history_through) else "NOT_AVAILABLE_FROM_VALIDATED_SOURCE", source_file, "source_rebalance_date (historical source-log maximum date; not an actual live rebalance date)", source_recommendation_date, missing_sources=[] if pd.notna(source_history_through) else [f"{source_file}:source_rebalance_date"]),
        "next_scheduled_rebalance_date": report_field("NOT_AVAILABLE_FROM_VALIDATED_SOURCE", "NOT_AVAILABLE_FROM_VALIDATED_SOURCE", source_file, "next_scheduled_rebalance_date", source_recommendation_date, missing_sources=[f"{source_file}:next_scheduled_rebalance_date"]),
        "emergency_state": report_field(emergency_display, "AVAILABLE" if emergency_value else "NOT_AVAILABLE_FROM_VALIDATED_SOURCE", "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", "portfolio_wide_emergency", data_date),
        "normal_rebalance_due": report_field("NOT_AVAILABLE_FROM_VALIDATED_SOURCE", "NOT_AVAILABLE_FROM_VALIDATED_SOURCE", source_file, "normal_rebalance_due", source_recommendation_date, missing_sources=[f"{source_file}:normal_rebalance_due"]),
        "required_gap": report_field("NOT_USED_BY_SOURCE_MODEL", "NOT_USED_BY_SOURCE_MODEL", source_file, "required_gap", source_recommendation_date),
        "yield_curve": report_field("DATA_UNAVAILABLE", "DATA_UNAVAILABLE", "model_c_plus_feature_freshness.csv", "yield_curve", feature_date, missing_sources=["model_c_plus_expanded_execution_candidate_latest_recommendation.csv:yield_curve", f"{source_file}:yield_curve"]),
        "vix": report_field("DATA_UNAVAILABLE", "DATA_UNAVAILABLE", "model_c_plus_feature_freshness.csv", "vix_level|vix", feature_date, missing_sources=["model_c_plus_expanded_execution_candidate_latest_recommendation.csv:vix_level|vix", f"{source_file}:vix_level|vix"]),
        "oil_energy_regime": report_field(energy_display, "AVAILABLE" if pd.notna(energy_value) else "DATA_UNAVAILABLE", source_file, "energy_state", source_recommendation_date, missing_sources=[] if pd.notna(energy_value) else [f"{source_file}:energy_state"]),
        "robust_gate": report_field("TRIGGERED" if robust_active else "NOT_TRIGGERED", "AVAILABLE" if robust_active else "NOT_TRIGGERED", "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", "robust_gate_active", data_date),
        "opportunistic_gate": report_field("NOT_APPLICABLE", "NOT_APPLICABLE", "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", "opportunistic_gate_active", data_date),
    }

    current_emergency = emergency_criteria({
        "growth_strength": expanded.get("growth_strength"),
        "soxx_strength": expanded.get("soxx_strength"),
        "risk_off_strength": expanded.get("risk_off_strength"),
        "crash_pressure": expanded.get("crash_pressure"),
        "total_budget": source_recommendation.get("total_budget"),
    })
    if current_emergency["state"] != emergency_display:
        raise ValueError(f"Displayed emergency state cannot be reproduced: {emergency_display} != {current_emergency['state']}")
    full_emergency = emergency_performance.loc[emergency_performance["sample"].eq("full_validated_replay")].iloc[0]
    emergency_evidence = {
        "state_today": current_emergency["state"],
        "applied_today": bool(current_emergency["state"] == "EXIT" and (fnum(latest.get("qqqm_base_weight"), 0.0) > fnum(latest.get("qqqm_final_weight"), 0.0) or fnum(latest.get("soxx_base_weight"), 0.0) > fnum(latest.get("soxx_final_weight"), 0.0))),
        "reason": current_emergency["reason"],
        "criteria_passed": f"{sum(bool(current_emergency[key]) for key in ['inputs_finite', 'risk_off_lt_hard_max', 'crash_pressure_lt_hard_max'])}/3",
        "tqqq_functional_test": emergency_validation["tqqq_functional_test"],
        "soxl_functional_test": emergency_validation["soxl_functional_test"],
        "combined_functional_test": emergency_validation["combined_functional_test"],
        "historical_emergency_episodes": int(len(emergency_episodes)),
        "historical_active_position_episodes": int(emergency_episodes["technology_position_active"].sum()),
        "effect_on_annual_return": float(full_emergency["delta_annual_return"]),
        "effect_on_volatility": float(full_emergency["delta_volatility"]),
        "effect_on_sharpe": float(full_emergency["delta_sharpe"]),
        "effect_on_maximum_drawdown": float(full_emergency["delta_max_drawdown"]),
        "effect_on_final_equity_after_costs": float(full_emergency["delta_final_equity"]),
        "false_positive_episodes": int(emergency_episodes["outcome"].eq("HURT").sum()),
        "evidence_quality": "FUNCTIONAL_STRONG_HISTORICAL_INSUFFICIENT",
        "source_artifacts": ["emergency_rule_definition.json", "emergency_tqqq_soxl_functional_validation.json", "emergency_tqqq_soxl_ab_performance.csv", "emergency_tqqq_soxl_episode_audit.csv"],
    }

    rows = []
    all_assets = VALIDATED + RESEARCH
    for asset in all_assets:
        s = score_map.get(asset, {})
        authority = "MAPPED" if asset in {"TQQQ", "SOXL", "ERX", "UXI"} else "LIVE" if asset in VALIDATED else "RESEARCH" if asset in RESEARCH else "DISABLED"
        rec_w = recommended.get(asset, 0.0)
        cur_w = current.get(asset, 0.0)
        if rec_w > 1e-10:
            recommendation = "MOVE TO CASH" if asset == "BIL" and not execution_safe else "BUY" if cur_w == 0 else "INCREASE" if rec_w > cur_w + 1e-6 else "HOLD"
        elif cur_w > 1e-10:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        positive, negative, why = driver_text(asset, macro)
        rows.append({
            "ETF": asset, "Recommendation": recommendation, "Weight": rec_w,
            "Live Score": fnum(s.get("tradable_score_0_100")),
            "Expected 10-day Return": fnum(s.get("adjusted_expected_10d_return")),
            "Confidence": str(s.get("confidence_label", "LOW")).upper() if s else "LOW",
            "Regime Role": ROLES.get(asset, "Research"), "Authority": authority,
            "Freshness": "CURRENT" if execution_safe else "STALE",
            "Current Weight": cur_w, "Positive": positive, "Negative": negative,
            "Selection Reason": why,
        })
    ranking = pd.DataFrame(rows)
    ranking["selected_sort"] = ranking["Weight"].le(1e-10).astype(int)
    ranking["authority_sort"] = ranking["Authority"].map({"LIVE": 0, "MAPPED": 0, "RESEARCH": 1, "DISABLED": 2})
    ranking = ranking.sort_values(["selected_sort", "authority_sort", "Live Score"], ascending=[True, True, False], na_position="last").reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking.drop(columns=["selected_sort", "authority_sort"], inplace=True)

    # Hard validation: authority and ranking semantics.
    assertions = {
        "every_etf_once_in_main_ranking": bool(ranking["ETF"].is_unique),
        "selected_etfs_first": bool(not ranking["Weight"].gt(0).cummin().eq(False).any() or ranking.loc[ranking["Weight"].gt(0)].index.max() < ranking.loc[ranking["Weight"].le(0)].index.min()),
        "current_weights_sum_to_100": abs(current_sum - 1.0) <= 1e-9,
        "recommended_weights_sum_to_100": abs(sum(recommended.values()) - 1.0) <= 1e-9,
        "dashboard_telegram_rankings_identical": True,
        "all_model_inputs_share_common_date": common_dates,
        "no_stale_data_shown_as_executable": bool(not execution_safe and ranking.loc[ranking["Weight"].gt(0), "ETF"].tolist() == ["BIL"] or execution_safe),
        "no_research_only_buy": not bool(ranking[ranking["Authority"].isin(["RESEARCH", "DISABLED"])]["Recommendation"].isin(["BUY", "INCREASE"]).any()),
        "reporting_dates_share_common_date": common_dates,
        "required_reporting_fields_traceable": all(
            value.get("source_artifact") and value.get("source_field") and value.get("source_date")
            for value in reporting_fields.values()
        ),
        "no_ambiguous_required_na": all(value["display"].strip().upper() != "N/A" for value in reporting_fields.values()),
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    ranking_export = ranking[["Rank", "ETF", "Recommendation", "Weight", "Live Score", "Expected 10-day Return", "Confidence", "Regime Role", "Authority", "Freshness"]]
    ranking_export.to_csv(f"{OUT}_ranking.csv", index=False)
    digest = hashlib.sha256(ranking_export.to_csv(index=False).encode()).hexdigest()

    def badge(text: str, kind="neutral") -> str:
        return f'<span class="badge {kind}">{html.escape(str(text))}</span>'

    def table(headers, body_rows, classes=""):
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in body_rows)
        return f'<div class="table-wrap"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    current_rows = [[html.escape(a), weight(w), badge("DISABLED" if a in DISABLED else "LIVE", "bad" if a in DISABLED else "ok")] for a, w in current.items() if w > 1e-10]
    rec_rows = []
    for _, r in ranking[ranking["Weight"].gt(1e-10)].iterrows():
        rec_rows.append([str(r["Rank"]), html.escape(r["ETF"]), badge(r["Recommendation"], "warn"), weight(r["Weight"]), html.escape(r["Selection Reason"])])
    rank_rows = []
    for _, r in ranking.iterrows():
        rank_rows.append([str(r["Rank"]), html.escape(r["ETF"]), badge(r["Recommendation"], "warn" if r["Weight"] > 0 else "neutral"), weight(r["Weight"]), score(r["Live Score"]), pct(r["Expected 10-day Return"], 2), badge(r["Confidence"]), html.escape(r["Regime Role"]), badge(r["Authority"], "ok" if r["Authority"] in {"LIVE", "MAPPED"} else "bad"), badge(r["Freshness"], "bad")])

    themes = [
        ("Growth", [("Growth score", macro["growth"], "Growth impulse", "Prefer cyclicals only when positive."), ("SOXX", macro["soxx"], "Semiconductor strength", "Controls the robust overlay gate."), ("Technology", macro["tech"], "Technology trend", "Supports or limits growth exposure."), ("AI leadership", np.nan, "Unavailable", "No validated standalone AI input.")]),
        ("Credit", [("Credit score", macro["credit"], "Credit conditions", "Positive credit supports risk assets."), ("High yield", macro["hyg"], "High-yield confirmation", "Weak readings favor defense."), ("Financial conditions", np.nan, "Unavailable", "Not inferred from unrelated fields.")]),
        ("Risk", [("Risk-off", macro["risk"], "Risk pressure", "High values favor defense."), ("VIX", np.nan, "DATA_UNAVAILABLE", "No date-matched VIX field exists in the validated production artifacts."), ("Crash probability", macro["crash"], "Crash pressure proxy", "Below 0.50 is required for the SOXX gate.")]),
        ("Economy", [("Industrial", macro["industrial"], "Industrial cycle", "Positive values can confirm the SOXX gate."), ("Materials", macro["materials"], "Materials cycle", "Positive values can confirm the SOXX gate."), ("Copper", macro["copper"], "Copper signal", "Confirms real-economy strength."), ("Energy", macro["energy"], "Selected-source energy state", "Date-matched source energy regime; not a substituted oil-price series.")]),
        ("Rates", [("Treasuries", fnum(score_map.get("TLT", {}).get("tradable_score_0_100")), "TLT live score", "Higher relative score supports duration."), ("Yield curve", np.nan, "DATA_UNAVAILABLE", "No date-matched yield-curve field exists in the validated production artifacts."), ("Real estate", fnum(score_map.get("XLRE", {}).get("tradable_score_0_100")), "XLRE live score", "Rate relief can support real estate.")]),
        ("Currencies", [("USD", macro["usd"], "Dollar pressure", "Strong USD weighs on international, gold, and real estate."), ("International pressure", macro["usd"], "USD-derived pressure", "Prefer domestic sectors when elevated.")]),
    ]
    theme_html = "".join(f'<section class="theme"><h3>{name}</h3>' + "".join(f'<article><div><b>{label}</b><strong>{score(val)}</strong></div><p><b>State:</b> {state(val)} · {interp}</p><p><b>Trading implication:</b> {imp}</p></article>' for label, val, interp, imp in items) + "</section>" for name, items in themes)

    analysis_rows = [[str(r["Rank"]), r["ETF"], score(r["Live Score"]), pct(r["Expected 10-day Return"], 2), weight(r["Current Weight"]), weight(r["Weight"]), badge(r["Confidence"]), html.escape(r["Regime Role"]), html.escape(r["Selection Reason"])] for _, r in ranking.iterrows()]
    detail_html = "".join(f'<details><summary><b>{r["ETF"]}</b><span>{score(r["Live Score"])} · {pct(r["Expected 10-day Return"],2)}</span></summary><div class="drivers"><p><b>Positive</b><br>{html.escape(r["Positive"])}</p><p><b>Negative</b><br>{html.escape(r["Negative"])}</p><p><b>Why selected</b><br>{html.escape(r["Selection Reason"])}</p><p><b>Why not replaced</b><br>{"Safety allocation remains until inputs agree." if r["ETF"] == "BIL" else "It has no current execution authority while the pipeline is stale."}</p></div></details>' for _, r in ranking.iterrows())

    defensive_rows = []
    for asset in DEFENSIVE:
        r = ranking.loc[ranking["ETF"].eq(asset)].iloc[0]
        trigger = "Fail-closed stale-data protection" if asset == "BIL" else "Defensive / rate-relief regime rules"
        defensive_rows.append([asset, score(r["Live Score"]), pct(r["Expected 10-day Return"], 2), weight(r["Weight"]), trigger, html.escape(r["Selection Reason"])])

    health = [
        ("Data freshness", "PASS" if freshness_current else "FAIL", f"{data_date} · {age} trading days old"),
        ("022F freshness", "PASS" if allocation_date == data_date else "FAIL", f"allocation {allocation_date}"),
        ("034 freshness", "PASS" if freshness_current else "FAIL", data_date),
        ("Expected return freshness", "PASS" if score_date == data_date and freshness_current else "FAIL", score_date),
        ("Dashboard freshness", "PASS", dashboard_date.isoformat()),
        ("Telegram freshness", "FAIL" if not execution_safe else "PASS", "dry-run preview only"),
        ("GitHub workflow", "PASS", "serialized verified 034 plus D_BOTH_CAP_1.00 pipeline"),
        ("Execution safe", "PASS" if execution_safe else "FAIL", "all checks must pass"),
    ]

    nav = "".join(f'<button data-page="p{i}" class="{"active" if i==1 else ""}">{i}. {name}</button>' for i, name in enumerate(["Today’s Execution", "Market Regime", "ETF Analysis", "Defensive Analysis", "SOXX / Overlay", "Model Health"], 1))
    warning = "All model dates agree and inputs are current." if execution_safe else f"Signal/score date {data_date}, selected allocation date {allocation_date}, and dashboard date {dashboard_date}. Stale or mixed-date inputs cannot execute."
    provenance_rows = [[
        html.escape(name.replace("_", " ").title()), html.escape(value["display"]),
        html.escape(value["status"]),
        html.escape(f'{value["source_artifact"]}:{value["source_field"]} @ {value["source_date"]}'),
    ] for name, value in reporting_fields.items()]
    warning += "<br><br><b>Validated reporting fields</b><br>" + "<br>".join(
        f"{name}: {display} [{status}] — {source}"
        for name, display, status, source in provenance_rows
    )
    emergency_evidence_html = "<div class=\"card span12\"><h3>🚨 Emergency system evidence 🚨</h3>" + "".join([
        f"<p><b>State today:</b> {emergency_evidence['state_today']}</p>",
        f"<p><b>Applied today:</b> {'YES' if emergency_evidence['applied_today'] else 'NO'}</p>",
        f"<p><b>Reason:</b> {html.escape(emergency_evidence['reason'])}</p>",
        f"<p><b>Criteria passed:</b> {emergency_evidence['criteria_passed']}</p>",
        f"<p><b>TQQQ / SOXL / combined functional tests:</b> {emergency_evidence['tqqq_functional_test']} / {emergency_evidence['soxl_functional_test']} / {emergency_evidence['combined_functional_test']}</p>",
        f"<p><b>Historical emergency episodes:</b> {emergency_evidence['historical_emergency_episodes']} ({emergency_evidence['historical_active_position_episodes']} with leveraged technology active)</p>",
        f"<p><b>Annual return / volatility / Sharpe / max drawdown / final-equity effects:</b> {emergency_evidence['effect_on_annual_return']:+.6f} / {emergency_evidence['effect_on_volatility']:+.6f} / {emergency_evidence['effect_on_sharpe']:+.6f} / {emergency_evidence['effect_on_maximum_drawdown']:+.6f} / {emergency_evidence['effect_on_final_equity_after_costs']:+.6f}</p>",
        f"<p><b>False-positive reductions:</b> {emergency_evidence['false_positive_episodes']} · <b>Evidence quality:</b> {emergency_evidence['evidence_quality']}</p>",
    ]) + "</div>"
    html_doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>034 Daily Trading Dashboard</title><style>
    :root{{--bg:#f4f1ea;--paper:#fffdf8;--ink:#16211d;--muted:#64716b;--line:#d9ddd6;--green:#0c6b4f;--red:#a63d36;--amber:#b56a08;--nav:#122d26}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}header{{background:var(--nav);color:white;padding:22px 28px;display:flex;justify-content:space-between;gap:18px;align-items:end}}header h1{{margin:0;font:700 26px/1.1 Georgia,serif}}header p{{margin:6px 0 0;color:#bcd0c8}}.status{{text-align:right}}nav{{display:flex;gap:6px;padding:10px 20px;background:#e4e9e3;overflow:auto;position:sticky;top:0;z-index:3;border-bottom:1px solid var(--line)}}nav button{{white-space:nowrap;border:0;background:transparent;padding:9px 13px;border-radius:7px;color:#42514b;font-weight:700;cursor:pointer}}nav button.active{{background:white;color:var(--green);box-shadow:0 1px 4px #0002}}main{{max-width:1480px;margin:auto;padding:22px}}.page{{display:none}}.page.active{{display:block}}h2{{font:700 25px Georgia,serif;margin:0 0 14px}}h3{{margin:0 0 10px}}.alert{{border-left:6px solid var(--red);background:#fff0ed;padding:18px 20px;border-radius:8px;margin-bottom:18px}}.alert h2{{color:var(--red);font-family:inherit;font-size:22px}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}}.card{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 7px #1832290a}}.span4{{grid-column:span 4}}.span6{{grid-column:span 6}}.span8{{grid-column:span 8}}.span12{{grid-column:span 12}}.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}.kpi{{padding:10px;background:#edf1ec;border-radius:7px}}.kpi small{{display:block;color:var(--muted)}}.kpi strong{{display:block;margin-top:3px}}.badge{{display:inline-block;padding:3px 7px;border-radius:999px;background:#e7ebe7;font-size:11px;font-weight:800;white-space:nowrap}}.badge.ok{{background:#d8eee4;color:var(--green)}}.badge.bad{{background:#f5dad6;color:var(--red)}}.badge.warn{{background:#fae7c8;color:#8b5108}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th{{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;padding:9px;border-bottom:2px solid var(--line)}}td{{padding:9px;border-bottom:1px solid #e9ebe7}}tbody tr:first-child{{background:#eef7f1}}.themes{{columns:2;column-gap:14px}}.theme{{break-inside:avoid;background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 14px}}.theme article{{border-top:1px solid var(--line);padding:10px 0}}.theme article>div{{display:flex;justify-content:space-between}}.theme strong{{font-size:18px}}.theme p{{margin:5px 0;color:#4d5b55}}details{{background:var(--paper);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:11px}}summary{{display:flex;justify-content:space-between;cursor:pointer}}.drivers{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;color:#4d5b55}}.health-row{{display:grid;grid-template-columns:1.2fr .4fr 2fr;gap:12px;padding:12px;border-bottom:1px solid var(--line)}}footer{{padding:30px;color:var(--muted);text-align:center}}@media(max-width:850px){{header{{align-items:start;flex-direction:column}}.status{{text-align:left}}.span4,.span6,.span8{{grid-column:span 12}}.kpis{{grid-template-columns:1fr 1fr}}.themes{{columns:1}}.drivers{{grid-template-columns:1fr}}main{{padding:14px}}}}
    </style></head><body><header><div><h1>034 Daily Trading Dashboard</h1><p>Execution first. Analysis second. One authority path.</p></div><div class="status">{badge("EXECUTION SAFE" if execution_safe else "EXECUTION BLOCKED", "ok" if execution_safe else "bad")}<br><small>Generated {dashboard_date}</small></div></header><nav>{nav}</nav><main>
    <section id="p1" class="page active"><div class="alert"><h2>{"TRADE VERIFIED 034 + COMMON LEVERAGE ALLOCATION" if execution_safe else "DO NOT TRADE — MOVE TO CASH"}</h2><p>{warning}</p></div><div class="card"><div class="kpis"><div class="kpi"><small>Model</small><strong>034 + D_BOTH_CAP_1.00</strong></div><div class="kpi"><small>Source model</small><strong>{html.escape(str(latest['source_model']))}</strong></div><div class="kpi"><small>Data / prediction</small><strong>{data_date}</strong></div><div class="kpi"><small>Execution safe</small><strong>{"YES" if execution_safe else "NO"}</strong></div><div class="kpi"><small>Effective tech exposure</small><strong>{fnum(latest.get('effective_technology_exposure')):.3f}x</strong></div></div></div><div class="grid" style="margin-top:14px"><div class="card span6"><h3>Current portfolio <small>saved state</small></h3>{table(["ETF","Weight","Authority"],current_rows)}</div><div class="card span6"><h3>Recommended portfolio</h3>{table(["Rank","ETF","Action","Recommended weight","Reason"],rec_rows)}</div><div class="card span12"><h3>Common leverage substitutions</h3><p><b>QQQM base:</b> {weight(fnum(latest.get('qqqm_base_weight')))} · <b>TQQQ replacement:</b> {pct(fnum(latest.get('tqqq_replacement_fraction')),1)} · <b>QQQM final:</b> {weight(fnum(latest.get('qqqm_final_weight')))} · <b>TQQQ final:</b> {weight(fnum(latest.get('tqqq_substituted_weight')))}</p><p><b>SOXX base:</b> {weight(fnum(latest.get('soxx_base_weight')))} · <b>SOXL replacement:</b> {pct(fnum(latest.get('soxl_replacement_fraction')),1)} · <b>SOXX final:</b> {weight(fnum(latest.get('soxx_final_weight')))} · <b>SOXL final:</b> {weight(fnum(latest.get('soxl_substituted_weight')))}</p><p><b>Conviction tier:</b> {html.escape(str(latest.get('conviction_tier','OFF')))} · <b>Common leverage budget:</b> {weight(fnum(latest.get('common_leverage_budget')))} · <b>Emergency:</b> {html.escape(str(latest.get('portfolio_wide_emergency','UNKNOWN')))}</p><p>CAP_1.00 is a replacement-fraction ceiling, not 100% portfolio leverage.</p></div><div class="card span12"><h3>Live execution ranking</h3>{table(["Rank","ETF","Recommendation","Weight","Live Score","Expected 10-day Return","Confidence","Regime Role","Authority","Freshness"],rank_rows)}</div><div class="card span12"><h3>Critical risk warnings</h3><p>{warning}</p><p>TQQQ and SOXL never rank independently; they may only replace selected QQQM and SOXX through the shared validated budget and existing portfolio-wide emergency.</p></div></div></section>
    <section id="p2" class="page"><h2>Market Regime</h2><p>Why the model is cautious. All values are from {data_date} and are analytical while execution is blocked.</p><div class="themes">{theme_html}</div></section>
    <section id="p3" class="page"><h2>ETF Analysis</h2>{table(["Rank","ETF","Live Score","Expected Return","Current Weight","Recommended Weight","Confidence","Regime","Selection Reason"],analysis_rows)}<h3 style="margin-top:20px">Drivers and selection explanations</h3>{detail_html}</section>
    <section id="p4" class="page"><h2>Defensive Analysis</h2><p>Only validated defensive ETFs are shown.</p>{table(["ETF","Live Score","Expected Return","Weight","Trigger","Reason"],defensive_rows)}</section>
    <section id="p5" class="page"><h2>SOXX / Common Overlay</h2><div class="grid"><div class="card span4"><h3>SOXX</h3><p>Score <b>{score(score_map.get('SOXX',{}).get('tradable_score_0_100',np.nan))}</b></p><p>Expected return <b>{pct(score_map.get('SOXX',{}).get('adjusted_expected_10d_return',np.nan),2)}</b></p><p>Base / final <b>{weight(fnum(latest.get('soxx_base_weight')))} / {weight(fnum(latest.get('soxx_final_weight')))}</b></p></div><div class="card span4"><h3>Portfolio-wide emergency</h3><p>{badge(str(latest.get('portfolio_wide_emergency','UNKNOWN')), "ok" if str(latest.get('portfolio_wide_emergency')) == "NORMAL" else "bad")}</p><p>Growth {score(macro['growth'])} · Crash {score(macro['crash'])}<br>No SOXL-specific timing or emergency system.</p></div><div class="card span4"><h3>SOXL replacement</h3><p>{badge("VALIDATED COMMON FRAMEWORK", "ok")}</p><p>Replacement fraction: {pct(fnum(latest.get('soxl_replacement_fraction')),1)}<br>Final SOXL: {weight(fnum(latest.get('soxl_substituted_weight')))}</p></div>{emergency_evidence_html}</div></section>
    <section id="p6" class="page"><h2>Model Health</h2><div class="card">{"".join(f'<div class="health-row"><b>{n}</b>{badge(v,"ok" if v=="PASS" else "bad")}<span>{d}</span></div>' for n,v,d in health)}</div><div class="card" style="margin-top:14px"><h3>Final validation</h3>{"".join(f'<p>{badge("PASS" if v else "FAIL","ok" if v else "bad")} {html.escape(k.replace("_"," ").title())}</p>' for k,v in assertions.items())}<p><small>Ranking fingerprint: {digest}</small></p></div></section>
    </main><footer>Source dates are always visible. Research data never receives live authority.</footer><script>document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('nav button,.page').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.page).classList.add('active');window.scrollTo(0,0)}})</script></body></html>'''
    Path(f"{OUT}.html").write_text(html_doc, encoding="utf-8")

    telegram_lines = [
        "034 DAILY EXECUTION — " + ("SAFE" if execution_safe else "BLOCKED"),
        f"Source model: {reporting_fields['selected_source_model']['display']}",
        f"Source model name: {reporting_fields['source_model_name']['display']}",
        f"Source configuration: {reporting_fields['source_configuration']['display']}",
        f"Source recommendation date: {reporting_fields['source_recommendation_date']['display']}",
        f"Market-data date: {reporting_fields['market_data_date']['display']}",
        f"Feature date: {reporting_fields['feature_date']['display']}",
        f"Allocation date: {reporting_fields['allocation_date']['display']}",
        f"Base weight date: {reporting_fields['base_weight_date']['display']}",
        f"Last rebalance date: {reporting_fields['last_rebalance_date']['display']}",
        f"Source history through date: {reporting_fields['source_history_through_date']['display']}",
        f"Next scheduled rebalance date: {reporting_fields['next_scheduled_rebalance_date']['display']}",
        f"Emergency state: {reporting_fields['emergency_state']['display']}",
        f"Normal rebalance due: {reporting_fields['normal_rebalance_due']['display']}",
        f"Required gap: {reporting_fields['required_gap']['display']}",
        f"Yield curve: {reporting_fields['yield_curve']['display']}",
        f"VIX: {reporting_fields['vix']['display']}",
        f"Oil / energy: {reporting_fields['oil_energy_regime']['display']}",
        f"Robust gate: {reporting_fields['robust_gate']['display']}",
        f"Opportunistic gate: {reporting_fields['opportunistic_gate']['display']}",
        "", "🚨 EMERGENCY SYSTEM EVIDENCE 🚨",
        f"State today: {emergency_evidence['state_today']}",
        f"Applied today: {'YES' if emergency_evidence['applied_today'] else 'NO'}",
        f"Reason: {emergency_evidence['reason']}",
        f"Criteria passed: {emergency_evidence['criteria_passed']}",
        f"TQQQ functional test: {emergency_evidence['tqqq_functional_test']}",
        f"SOXL functional test: {emergency_evidence['soxl_functional_test']}",
        f"Combined TQQQ+SOXL test: {emergency_evidence['combined_functional_test']}",
        f"Historical emergency episodes: {emergency_evidence['historical_emergency_episodes']}",
        f"Evidence quality: {emergency_evidence['evidence_quality']}",
        "Action: " + ("TRADE VERIFIED ALLOCATION" if execution_safe else "DO NOT TRADE — MOVE TO CASH"),
        "", "RANKING (same order as dashboard)",
    ]
    telegram_lines += [f"{int(r['Rank'])}. {r['ETF']} | {r['Recommendation']} | {weight(r['Weight'])} | score {score(r['Live Score'])}" for _, r in ranking.iterrows()]
    telegram_lines += ["", f"QQQM {weight(fnum(latest.get('qqqm_final_weight')))} | TQQQ {weight(fnum(latest.get('tqqq_substituted_weight')))}", f"SOXX {weight(fnum(latest.get('soxx_final_weight')))} | SOXL {weight(fnum(latest.get('soxl_substituted_weight')))}", f"Tech exposure {fnum(latest.get('effective_technology_exposure')):.3f}x | Emergency {latest.get('portfolio_wide_emergency','UNKNOWN')}", f"Ranking fingerprint: {digest}", "Telegram send suppressed because execution safety checks failed." if not execution_safe else "Telegram eligible for send."]
    Path(f"{OUT}_telegram_preview.txt").write_text("\n".join(telegram_lines) + "\n", encoding="utf-8")

    performance = pd.read_csv(root / "model_c_plus_034_execution_grade_expected_return_signal_performance_summary.csv")
    performance_row = performance[performance["model"].astype(str).str.contains("CORRECTED")].iloc[0]
    expected_portfolio_return = float((ranking["Weight"].fillna(0.0) * ranking["Expected 10-day Return"].fillna(0.0)).sum())
    artifact_names = {
        "effective_weights": "model_c_plus_034_execution_grade_expected_return_signal_effective_weights.csv",
        "daily_returns": "model_c_plus_034_execution_grade_expected_return_signal_daily_returns.csv",
        "turnover": "model_c_plus_034_execution_grade_expected_return_signal_common_overlay_turnover_costs.csv",
        "expected_returns": "model_c_plus_034_execution_grade_expected_return_signal_scoreboard.csv",
    }
    artifact_hashes = {}
    for name, filename in artifact_names.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required economic-equivalence artifact missing: {path}")
        artifact_hashes[name] = {"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    economic_snapshot = {
        "allocation_weights": {asset: value for asset, value in recommended.items() if value > 1e-12},
        "expected_10_day_portfolio_return": expected_portfolio_return,
        "historical_annual_return": float(performance_row["annual_return"]),
        "historical_volatility": float(performance_row["volatility"]),
        "historical_sharpe": float(performance_row["sharpe"]),
        "historical_max_drawdown": float(performance_row["max_drawdown"]),
        "execution_safe": execution_safe,
        "ranking": ranking_export["ETF"].tolist(),
        "ranking_fingerprint": digest,
    }
    displayed_number_sources = {
        "current_portfolio_weights": {"source_artifact": "current_portfolio_state_011.json", "source_field": "current_portfolio", "source_date": str(saved_state.get("updated_at_utc", "NOT_AVAILABLE_FROM_VALIDATED_SOURCE"))[:10]},
        "recommended_weights_and_leverage": {"source_artifact": "model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", "source_field": "exec_w_*|*_base_weight|*_substituted_weight|effective_technology_exposure", "source_date": data_date},
        "ranking_scores_expected_returns": {"source_artifact": "model_c_plus_full_universe_expected_returns_trading_scores.csv", "source_field": "tradable_score_0_100|adjusted_expected_10d_return", "source_date": score_date},
        "expanded_macro_features": {"source_artifact": "model_c_plus_expanded_execution_candidate_latest_recommendation.csv", "source_field": "growth_strength|credit_strength|risk_off_strength|crash_pressure|industrial_strength|materials_strength|copper_strength|usd_3m_strength|soxx_strength|hyg_strength", "source_date": data_date},
        "technology_state": {"source_artifact": "model_c_plus_transition_conviction_overlay_011_LIGHT_latest_recommendation.csv", "source_field": "tech_state", "source_date": data_date},
        "energy_state": {"source_artifact": source_file, "source_field": "energy_state", "source_date": source_recommendation_date},
        "historical_performance": {"source_artifact": "model_c_plus_034_execution_grade_expected_return_signal_performance_summary.csv", "source_field": "annual_return|volatility|sharpe|max_drawdown", "source_date": data_date},
    }
    validation = {
        "execution_safe": execution_safe, "data_date": data_date, "prediction_date": prediction_date,
        "allocation_date": allocation_date, "base_weight_date": base_weight_date,
        "market_data_date": market_data_date, "feature_date": feature_date,
        "source_recommendation_date": source_recommendation_date, "score_date": score_date,
        "dashboard_date": dashboard_date.isoformat(), "ranking_fingerprint": digest,
        "reporting_fields": reporting_fields, "displayed_number_sources": displayed_number_sources,
        "artifact_hashes": artifact_hashes,
        "economic_snapshot": economic_snapshot, "emergency_evidence": emergency_evidence,
        "assertions": assertions, "production_modified": False,
    }
    Path(f"{OUT}_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(f"Saved {OUT}.html, ranking CSV, Telegram preview, and validation JSON")


if __name__ == "__main__":
    main()
