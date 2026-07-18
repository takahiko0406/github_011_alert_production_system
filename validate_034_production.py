"""Fail-closed common-date and execution-authority validator for production 034."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from common_technology_leverage_080 import TECHNOLOGY_EXPOSURE_LIMIT


ROOT = Path(__file__).resolve().parent
VALIDATED = {"QQQM", "TQQQ", "SOXX", "SOXL", "IWM", "FEZ", "XLE", "ERX", "XLB", "XLI", "UXI", "XLV", "XLP", "XLU", "XLRE", "TLT", "GLD", "XSOE", "BIL"}
FORBIDDEN = {"XLF", "IEF", "TNA", "UGL"}


def latest_date(path: str, columns=("latest_data_date", "signal_date", "date")) -> str:
    df = pd.read_csv(ROOT / path)
    if df.empty:
        raise ValueError(f"Empty required artifact: {path}")
    row = df.iloc[-1]
    for col in columns:
        if col in row and pd.notna(row[col]):
            return str(row[col])[:10]
    raise ValueError(f"No date in {path}")


def main() -> None:
    artifacts = {
        "market_data": ("model_c_plus_market_data_freshness.csv", ("latest_data_date",)),
        "features": ("model_c_plus_feature_freshness.csv", ("latest_data_date",)),
        "022f": ("model_c_plus_022F_calibrated_defense_validation_best_latest_recommendation.csv", ("latest_data_date",)),
        "expanded": ("model_c_plus_expanded_execution_candidate_latest_recommendation.csv", ("latest_data_date",)),
        "expected_returns": ("model_c_plus_full_universe_expected_returns_trading_scores.csv", ("signal_date",)),
        "034": ("model_c_plus_034_execution_grade_expected_return_signal_latest_recommendation.csv", ("latest_data_date",)),
    }
    dates = {name: latest_date(path, cols) for name, (path, cols) in artifacts.items()}
    common = set(dates.values())
    if len(common) != 1:
        raise ValueError(f"Common-date validation failed: {dates}")
    common_date = next(iter(common))

    latest = pd.read_csv(ROOT / artifacts["034"][0]).iloc[-1]
    if str(latest.get("model")) != "034_EXECUTION_GRADE_EXPECTED_RETURN_SIGNAL":
        raise ValueError("034 is not authoritative")
    if str(latest.get("source_model")) not in {"022F_BASE", "EXPANDED_CANDIDATE"}:
        raise ValueError("Invalid 034 source model")
    weights = {c[7:]: float(latest[c]) for c in latest.index if c.startswith("exec_w_") and pd.notna(latest[c])}
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError(f"Final weights sum to {sum(weights.values())}")
    invalid = {a: w for a, w in weights.items() if w > 1e-12 and a not in VALIDATED}
    if invalid:
        raise ValueError(f"Nonvalidated selected ETFs: {invalid}")
    if any(weights.get(a, 0.0) > 1e-12 for a in FORBIDDEN):
        raise ValueError("Forbidden asset has live weight")
    if weights.get("TQQQ", 0.0) > 1e-12 and float(latest.get("qqqm_base_weight", 0.0)) <= 0:
        raise ValueError("TQQQ lacks selected QQQM base authority")
    if weights.get("SOXL", 0.0) > 1e-12 and float(latest.get("soxx_base_weight", 0.0)) <= 0:
        raise ValueError("SOXL lacks selected SOXX base authority")
    if weights.get("ERX", 0.0) > 1e-12 and float(latest.get("xle_base_weight", 0.0)) <= 0:
        raise ValueError("ERX lacks selected XLE base authority")
    if weights.get("UXI", 0.0) > 1e-12 and float(latest.get("xli_base_weight", 0.0)) <= 0:
        raise ValueError("UXI lacks selected XLI base authority")
    if str(latest.get("common_leverage_configuration")) != "D_BOTH_CAP_1.00":
        raise ValueError("Wrong common leverage configuration")
    if str(latest.get("common_leverage_framework")) != "TQQQ_STYLE_SHARED_BUDGET":
        raise ValueError("Common leverage metadata missing")
    if str(latest.get("leveraged_assets_rank_independently")).lower() not in {"false", "0"}:
        raise ValueError("Leveraged ETFs may not rank independently")
    effective_technology = weights.get("QQQM", 0.0) + weights.get("SOXX", 0.0) + 3.0 * (weights.get("TQQQ", 0.0) + weights.get("SOXL", 0.0))
    if effective_technology > TECHNOLOGY_EXPOSURE_LIMIT:
        raise ValueError(f"Technology exposure exceeds limit: {effective_technology}")

    dashboard = json.loads((ROOT / "model_c_plus_034_live_dashboard_validation.json").read_text(encoding="utf-8"))
    if not dashboard.get("execution_safe"):
        raise ValueError(f"Dashboard execution_safe is false: {dashboard}")
    reporting_dates = {
        "data_date": dashboard.get("data_date"), "market_data_date": dashboard.get("market_data_date"),
        "feature_date": dashboard.get("feature_date"), "allocation_date": dashboard.get("allocation_date"),
        "source_recommendation_date": dashboard.get("source_recommendation_date"),
    }
    if set(reporting_dates.values()) != {common_date}:
        raise ValueError(f"Dashboard reporting date mismatch: {reporting_dates}")
    reporting_fields = dashboard.get("reporting_fields", {})
    required_reporting_fields = {
        "selected_source_model", "source_model_name", "source_configuration",
        "source_recommendation_date", "allocation_date", "market_data_date", "feature_date",
        "last_rebalance_date", "next_scheduled_rebalance_date", "emergency_state",
        "normal_rebalance_due", "required_gap", "yield_curve", "vix",
        "oil_energy_regime", "robust_gate", "opportunistic_gate",
    }
    missing_reporting = sorted(required_reporting_fields - set(reporting_fields))
    if missing_reporting:
        raise ValueError(f"Required dashboard reporting fields missing: {missing_reporting}")
    allowed_statuses = {
        "AVAILABLE", "NOT_APPLICABLE", "NOT_USED_BY_SOURCE_MODEL", "NOT_TRIGGERED",
        "DATA_UNAVAILABLE", "NOT_AVAILABLE_FROM_VALIDATED_SOURCE",
    }
    for name in sorted(required_reporting_fields):
        field = reporting_fields[name]
        display = str(field.get("display", "")).strip()
        if not display or display.upper() == "N/A":
            raise ValueError(f"Ambiguous required dashboard field: {name}={display!r}")
        if field.get("status") not in allowed_statuses:
            raise ValueError(f"Invalid explicit status for {name}: {field.get('status')}")
        if not field.get("source_artifact") or not field.get("source_field") or not field.get("source_date"):
            raise ValueError(f"Displayed field lacks traceable source: {name}={field}")
        if field.get("source_date") != common_date:
            raise ValueError(f"Displayed field source is not date matched: {name}={field}")
        if field.get("status") in {"DATA_UNAVAILABLE", "NOT_AVAILABLE_FROM_VALIDATED_SOURCE"} and not field.get("missing_sources"):
            raise ValueError(f"Unavailable field lacks exact missing source: {name}={field}")
    displayed_number_sources = dashboard.get("displayed_number_sources", {})
    required_numeric_categories = {
        "current_portfolio_weights", "recommended_weights_and_leverage",
        "ranking_scores_expected_returns", "expanded_macro_features", "technology_state",
        "energy_state", "historical_performance",
    }
    if set(displayed_number_sources) != required_numeric_categories:
        raise ValueError(f"Displayed numeric provenance incomplete: {sorted(displayed_number_sources)}")
    for name, source in displayed_number_sources.items():
        if not source.get("source_artifact") or not source.get("source_field") or not source.get("source_date"):
            raise ValueError(f"Displayed number lacks a traceable source: {name}={source}")
        if not (ROOT / source["source_artifact"]).is_file():
            raise ValueError(f"Displayed numeric source artifact missing: {name}={source}")
    ranking_path = ROOT / "model_c_plus_034_live_dashboard_ranking.csv"
    digest = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
    # Generator hashes normalized CSV text; recompute in exactly the same form.
    ranking = pd.read_csv(ranking_path)
    normalized = hashlib.sha256(ranking.to_csv(index=False).encode()).hexdigest()
    expected = dashboard.get("ranking_fingerprint")
    telegram = (ROOT / "model_c_plus_034_live_dashboard_telegram_preview.txt").read_text(encoding="utf-8")
    if normalized != expected or expected not in telegram:
        raise ValueError(f"Dashboard/Telegram ranking mismatch ({digest})")
    dashboard_html = (ROOT / "model_c_plus_034_live_dashboard.html").read_text(encoding="utf-8")
    for name in sorted(required_reporting_fields):
        display = str(reporting_fields[name]["display"])
        if display not in dashboard_html or display not in telegram:
            raise ValueError(f"Dashboard/Telegram canonical field disagreement: {name}={display}")

    artifact_hashes = dashboard.get("artifact_hashes", {})
    for name in ("effective_weights", "daily_returns", "turnover", "expected_returns"):
        artifact = artifact_hashes.get(name, {})
        path = ROOT / str(artifact.get("file", ""))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
            raise ValueError(f"Economic artifact changed after dashboard generation: {name}")
    snapshot = dashboard.get("economic_snapshot", {})
    snapshot_weights = {str(a): float(w) for a, w in snapshot.get("allocation_weights", {}).items()}
    live_weights = {a: w for a, w in weights.items() if w > 1e-12}
    if snapshot_weights != live_weights:
        raise ValueError(f"Allocation economics changed: {snapshot_weights} != {live_weights}")
    performance = pd.read_csv(ROOT / "model_c_plus_034_execution_grade_expected_return_signal_performance_summary.csv")
    performance_row = performance[performance["model"].astype(str).str.contains("CORRECTED")].iloc[0]
    metric_fields = {
        "historical_annual_return": "annual_return", "historical_volatility": "volatility",
        "historical_sharpe": "sharpe", "historical_max_drawdown": "max_drawdown",
    }
    for snapshot_name, column in metric_fields.items():
        if not math.isclose(float(snapshot[snapshot_name]), float(performance_row[column]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Historical metric changed: {snapshot_name}")
    expected_portfolio = float((ranking["Weight"].fillna(0.0) * ranking["Expected 10-day Return"].fillna(0.0)).sum())
    if not math.isclose(float(snapshot["expected_10_day_portfolio_return"]), expected_portfolio, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Expected 10-day portfolio return changed")
    if snapshot.get("ranking") != ranking["ETF"].tolist() or snapshot.get("ranking_fingerprint") != expected:
        raise ValueError("ETF ranking or ranking fingerprint changed")
    if bool(snapshot.get("execution_safe")) != bool(dashboard.get("execution_safe")):
        raise ValueError("Execution-safe state changed")

    output = pd.DataFrame([{"component": name, "date": value, "status": "PASS"} for name, value in dates.items()])
    output.to_csv(ROOT / "model_c_plus_034_freshness_validation.csv", index=False)
    (ROOT / "model_c_plus_034_execution_ready.json").write_text(json.dumps({
        "execution_safe": True, "common_date": common_date, "source_model": latest["source_model"],
        "weights_sum": sum(weights.values()), "selected_assets": [a for a, w in weights.items() if w > 1e-12],
        "ranking_fingerprint": expected, "effective_technology_exposure": effective_technology,
        "common_leverage_configuration": "D_BOTH_CAP_1.00",
    }, indent=2), encoding="utf-8")
    print(f"034 PRODUCTION VALIDATION PASS: {common_date}")


if __name__ == "__main__":
    main()
