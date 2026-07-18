"""Fail-closed common-date and execution-authority validator for production 034."""

import hashlib
import json
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
    if dashboard.get("data_date") != common_date or dashboard.get("allocation_date") != common_date:
        raise ValueError("Dashboard date mismatch")
    ranking_path = ROOT / "model_c_plus_034_live_dashboard_ranking.csv"
    digest = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
    # Generator hashes normalized CSV text; recompute in exactly the same form.
    ranking = pd.read_csv(ranking_path)
    normalized = hashlib.sha256(ranking.to_csv(index=False).encode()).hexdigest()
    expected = dashboard.get("ranking_fingerprint")
    telegram = (ROOT / "model_c_plus_034_live_dashboard_telegram_preview.txt").read_text(encoding="utf-8")
    if normalized != expected or expected not in telegram:
        raise ValueError(f"Dashboard/Telegram ranking mismatch ({digest})")

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
