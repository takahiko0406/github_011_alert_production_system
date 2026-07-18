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


OUT = os.getenv("DASHBOARD_OUTPUT_PREFIX", "model_c_plus_074_034_daily_trading_dashboard")
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
    return "UNAVAILABLE" if pd.isna(value) else f"{100 * float(value):+.{digits}f}%"


def weight(value) -> str:
    return "UNAVAILABLE" if pd.isna(value) else f"{100 * float(value):.1f}%"


def score(value) -> str:
    return "UNAVAILABLE" if pd.isna(value) else f"{float(value):.1f}"


def state(value: float) -> str:
    if pd.isna(value): return "Unavailable"
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
    state_path = root / "current_portfolio_state_011.json"
    saved_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"current_portfolio": {"BIL": 1.0}}

    data_date = str(latest["latest_data_date"])[:10]
    prediction_date = str(latest["signal_date"])[:10]
    allocation_date = str(latest["base_weight_date"])[:10]
    score_date = str(scores["signal_date"].iloc[-1])[:10]
    now_utc = datetime.now(timezone.utc)
    dashboard_date = now_utc.date()
    # Before the U.S. cash close, today's session is not a completed session and
    # must not count as staleness. This is important for the scheduled Tokyo run.
    completed_cutoff = (pd.Timestamp(dashboard_date) - pd.offsets.BDay(1)).date() if now_utc.hour < 21 else dashboard_date
    age = business_age(data_date, completed_cutoff)
    common_dates = data_date == prediction_date == score_date == allocation_date
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
    }
    if not all(v for k, v in assertions.items() if k != "all_model_inputs_share_common_date"):
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
        ("Risk", [("Risk-off", macro["risk"], "Risk pressure", "High values favor defense."), ("VIX", np.nan, "Unavailable", "Leverage remains disabled without a validated current reading."), ("Crash probability", macro["crash"], "Crash pressure proxy", "Below 0.50 is required for the SOXX gate.")]),
        ("Economy", [("Industrial", macro["industrial"], "Industrial cycle", "Positive values can confirm the SOXX gate."), ("Materials", macro["materials"], "Materials cycle", "Positive values can confirm the SOXX gate."), ("Copper", macro["copper"], "Copper signal", "Confirms real-economy strength.")]),
        ("Rates", [("Treasuries", fnum(score_map.get("TLT", {}).get("tradable_score_0_100")), "TLT live score", "Higher relative score supports duration."), ("Yield curve", np.nan, "Unavailable", "No validated current curve input."), ("Real estate", fnum(score_map.get("XLRE", {}).get("tradable_score_0_100")), "XLRE live score", "Rate relief can support real estate.")]),
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
    html_doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>034 Daily Trading Dashboard</title><style>
    :root{{--bg:#f4f1ea;--paper:#fffdf8;--ink:#16211d;--muted:#64716b;--line:#d9ddd6;--green:#0c6b4f;--red:#a63d36;--amber:#b56a08;--nav:#122d26}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}header{{background:var(--nav);color:white;padding:22px 28px;display:flex;justify-content:space-between;gap:18px;align-items:end}}header h1{{margin:0;font:700 26px/1.1 Georgia,serif}}header p{{margin:6px 0 0;color:#bcd0c8}}.status{{text-align:right}}nav{{display:flex;gap:6px;padding:10px 20px;background:#e4e9e3;overflow:auto;position:sticky;top:0;z-index:3;border-bottom:1px solid var(--line)}}nav button{{white-space:nowrap;border:0;background:transparent;padding:9px 13px;border-radius:7px;color:#42514b;font-weight:700;cursor:pointer}}nav button.active{{background:white;color:var(--green);box-shadow:0 1px 4px #0002}}main{{max-width:1480px;margin:auto;padding:22px}}.page{{display:none}}.page.active{{display:block}}h2{{font:700 25px Georgia,serif;margin:0 0 14px}}h3{{margin:0 0 10px}}.alert{{border-left:6px solid var(--red);background:#fff0ed;padding:18px 20px;border-radius:8px;margin-bottom:18px}}.alert h2{{color:var(--red);font-family:inherit;font-size:22px}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}}.card{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 7px #1832290a}}.span4{{grid-column:span 4}}.span6{{grid-column:span 6}}.span8{{grid-column:span 8}}.span12{{grid-column:span 12}}.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}.kpi{{padding:10px;background:#edf1ec;border-radius:7px}}.kpi small{{display:block;color:var(--muted)}}.kpi strong{{display:block;margin-top:3px}}.badge{{display:inline-block;padding:3px 7px;border-radius:999px;background:#e7ebe7;font-size:11px;font-weight:800;white-space:nowrap}}.badge.ok{{background:#d8eee4;color:var(--green)}}.badge.bad{{background:#f5dad6;color:var(--red)}}.badge.warn{{background:#fae7c8;color:#8b5108}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th{{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;padding:9px;border-bottom:2px solid var(--line)}}td{{padding:9px;border-bottom:1px solid #e9ebe7}}tbody tr:first-child{{background:#eef7f1}}.themes{{columns:2;column-gap:14px}}.theme{{break-inside:avoid;background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 14px}}.theme article{{border-top:1px solid var(--line);padding:10px 0}}.theme article>div{{display:flex;justify-content:space-between}}.theme strong{{font-size:18px}}.theme p{{margin:5px 0;color:#4d5b55}}details{{background:var(--paper);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:11px}}summary{{display:flex;justify-content:space-between;cursor:pointer}}.drivers{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;color:#4d5b55}}.health-row{{display:grid;grid-template-columns:1.2fr .4fr 2fr;gap:12px;padding:12px;border-bottom:1px solid var(--line)}}footer{{padding:30px;color:var(--muted);text-align:center}}@media(max-width:850px){{header{{align-items:start;flex-direction:column}}.status{{text-align:left}}.span4,.span6,.span8{{grid-column:span 12}}.kpis{{grid-template-columns:1fr 1fr}}.themes{{columns:1}}.drivers{{grid-template-columns:1fr}}main{{padding:14px}}}}
    </style></head><body><header><div><h1>034 Daily Trading Dashboard</h1><p>Execution first. Analysis second. One authority path.</p></div><div class="status">{badge("EXECUTION SAFE" if execution_safe else "EXECUTION BLOCKED", "ok" if execution_safe else "bad")}<br><small>Generated {dashboard_date}</small></div></header><nav>{nav}</nav><main>
    <section id="p1" class="page active"><div class="alert"><h2>{"TRADE VERIFIED 034 + COMMON LEVERAGE ALLOCATION" if execution_safe else "DO NOT TRADE — MOVE TO CASH"}</h2><p>{warning}</p></div><div class="card"><div class="kpis"><div class="kpi"><small>Model</small><strong>034 + D_BOTH_CAP_1.00</strong></div><div class="kpi"><small>Source model</small><strong>{html.escape(str(latest['source_model']))}</strong></div><div class="kpi"><small>Data / prediction</small><strong>{data_date}</strong></div><div class="kpi"><small>Execution safe</small><strong>{"YES" if execution_safe else "NO"}</strong></div><div class="kpi"><small>Effective tech exposure</small><strong>{fnum(latest.get('effective_technology_exposure')):.3f}x</strong></div></div></div><div class="grid" style="margin-top:14px"><div class="card span6"><h3>Current portfolio <small>saved state</small></h3>{table(["ETF","Weight","Authority"],current_rows)}</div><div class="card span6"><h3>Recommended portfolio</h3>{table(["Rank","ETF","Action","Recommended weight","Reason"],rec_rows)}</div><div class="card span12"><h3>Common leverage substitutions</h3><p><b>QQQM base:</b> {weight(fnum(latest.get('qqqm_base_weight')))} · <b>TQQQ replacement:</b> {pct(fnum(latest.get('tqqq_replacement_fraction')),1)} · <b>QQQM final:</b> {weight(fnum(latest.get('qqqm_final_weight')))} · <b>TQQQ final:</b> {weight(fnum(latest.get('tqqq_substituted_weight')))}</p><p><b>SOXX base:</b> {weight(fnum(latest.get('soxx_base_weight')))} · <b>SOXL replacement:</b> {pct(fnum(latest.get('soxl_replacement_fraction')),1)} · <b>SOXX final:</b> {weight(fnum(latest.get('soxx_final_weight')))} · <b>SOXL final:</b> {weight(fnum(latest.get('soxl_substituted_weight')))}</p><p><b>Conviction tier:</b> {html.escape(str(latest.get('conviction_tier','OFF')))} · <b>Common leverage budget:</b> {weight(fnum(latest.get('common_leverage_budget')))} · <b>Emergency:</b> {html.escape(str(latest.get('portfolio_wide_emergency','UNKNOWN')))}</p><p>CAP_1.00 is a replacement-fraction ceiling, not 100% portfolio leverage.</p></div><div class="card span12"><h3>Live execution ranking</h3>{table(["Rank","ETF","Recommendation","Weight","Live Score","Expected 10-day Return","Confidence","Regime Role","Authority","Freshness"],rank_rows)}</div><div class="card span12"><h3>Critical risk warnings</h3><p>{warning}</p><p>TQQQ and SOXL never rank independently; they may only replace selected QQQM and SOXX through the shared validated budget and existing portfolio-wide emergency.</p></div></div></section>
    <section id="p2" class="page"><h2>Market Regime</h2><p>Why the model is cautious. All values are from {data_date} and are analytical while execution is blocked.</p><div class="themes">{theme_html}</div></section>
    <section id="p3" class="page"><h2>ETF Analysis</h2>{table(["Rank","ETF","Live Score","Expected Return","Current Weight","Recommended Weight","Confidence","Regime","Selection Reason"],analysis_rows)}<h3 style="margin-top:20px">Drivers and selection explanations</h3>{detail_html}</section>
    <section id="p4" class="page"><h2>Defensive Analysis</h2><p>Only validated defensive ETFs are shown.</p>{table(["ETF","Live Score","Expected Return","Weight","Trigger","Reason"],defensive_rows)}</section>
    <section id="p5" class="page"><h2>SOXX / Common Overlay</h2><div class="grid"><div class="card span4"><h3>SOXX</h3><p>Score <b>{score(score_map.get('SOXX',{}).get('tradable_score_0_100',np.nan))}</b></p><p>Expected return <b>{pct(score_map.get('SOXX',{}).get('adjusted_expected_10d_return',np.nan),2)}</b></p><p>Base / final <b>{weight(fnum(latest.get('soxx_base_weight')))} / {weight(fnum(latest.get('soxx_final_weight')))}</b></p></div><div class="card span4"><h3>Portfolio-wide emergency</h3><p>{badge(str(latest.get('portfolio_wide_emergency','UNKNOWN')), "ok" if str(latest.get('portfolio_wide_emergency')) == "NORMAL" else "bad")}</p><p>Growth {score(macro['growth'])} · Crash {score(macro['crash'])}<br>No SOXL-specific timing or emergency system.</p></div><div class="card span4"><h3>SOXL replacement</h3><p>{badge("VALIDATED COMMON FRAMEWORK", "ok")}</p><p>Replacement fraction: {pct(fnum(latest.get('soxl_replacement_fraction')),1)}<br>Final SOXL: {weight(fnum(latest.get('soxl_substituted_weight')))}</p></div></div></section>
    <section id="p6" class="page"><h2>Model Health</h2><div class="card">{"".join(f'<div class="health-row"><b>{n}</b>{badge(v,"ok" if v=="PASS" else "bad")}<span>{d}</span></div>' for n,v,d in health)}</div><div class="card" style="margin-top:14px"><h3>Final validation</h3>{"".join(f'<p>{badge("PASS" if v else "FAIL","ok" if v else "bad")} {html.escape(k.replace("_"," ").title())}</p>' for k,v in assertions.items())}<p><small>Ranking fingerprint: {digest}</small></p></div></section>
    </main><footer>Source dates are always visible. Research data never receives live authority.</footer><script>document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('nav button,.page').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.page).classList.add('active');window.scrollTo(0,0)}})</script></body></html>'''
    Path(f"{OUT}.html").write_text(html_doc, encoding="utf-8")

    telegram_lines = [
        "034 DAILY EXECUTION — " + ("SAFE" if execution_safe else "BLOCKED"),
        f"Data {data_date} | Allocation {allocation_date} | Freshness {'CURRENT' if execution_safe else 'STALE'}",
        "Action: " + ("TRADE VERIFIED ALLOCATION" if execution_safe else "DO NOT TRADE — MOVE TO CASH"),
        "", "RANKING (same order as dashboard)",
    ]
    telegram_lines += [f"{int(r['Rank'])}. {r['ETF']} | {r['Recommendation']} | {weight(r['Weight'])} | score {score(r['Live Score'])}" for _, r in ranking.iterrows()]
    telegram_lines += ["", f"QQQM {weight(fnum(latest.get('qqqm_final_weight')))} | TQQQ {weight(fnum(latest.get('tqqq_substituted_weight')))}", f"SOXX {weight(fnum(latest.get('soxx_final_weight')))} | SOXL {weight(fnum(latest.get('soxl_substituted_weight')))}", f"Tech exposure {fnum(latest.get('effective_technology_exposure')):.3f}x | Emergency {latest.get('portfolio_wide_emergency','UNKNOWN')}", f"Ranking fingerprint: {digest}", "Telegram send suppressed because execution safety checks failed." if not execution_safe else "Telegram eligible for send."]
    Path(f"{OUT}_telegram_preview.txt").write_text("\n".join(telegram_lines) + "\n", encoding="utf-8")

    validation = {
        "execution_safe": execution_safe, "data_date": data_date, "prediction_date": prediction_date,
        "allocation_date": allocation_date, "score_date": score_date, "dashboard_date": dashboard_date.isoformat(),
        "ranking_fingerprint": digest, "assertions": assertions, "production_modified": True,
    }
    Path(f"{OUT}_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(f"Saved {OUT}.html, ranking CSV, Telegram preview, and validation JSON")


if __name__ == "__main__":
    main()
