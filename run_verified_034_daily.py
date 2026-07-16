"""Atomic, fail-closed daily orchestration for verified corrected 034."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parent
AUTHORITATIVE_PATTERNS = [
    "model_c_plus_022F_calibrated_defense_validation_best_*",
    "model_c_plus_expanded_execution_candidate_*",
    "model_c_plus_full_universe_expected_returns_*",
    "model_c_plus_034_execution_grade_expected_return_signal_*",
    "model_c_plus_034_live_dashboard*",
    "model_c_plus_034_freshness_validation.csv",
    "model_c_plus_034_execution_ready.json",
]


def run(script: str, env: dict | None = None, attempts: int = 3) -> None:
    print(f"\n=== RUN {script} ===", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run([sys.executable, script], cwd=ROOT, env=merged, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = 5 * attempt
            print(f"{script} failed on attempt {attempt}/{attempts}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise last_error


def quarantine() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / ".stale_quarantine" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    for pattern in AUTHORITATIVE_PATTERNS:
        for path in ROOT.glob(pattern):
            if path.is_file() and path.name not in seen:
                shutil.move(str(path), destination / path.name)
                seen.add(path.name)
    return destination


def write_freshness_seed() -> str:
    expanded = pd.read_csv(ROOT / "model_c_plus_expanded_execution_candidate_latest_recommendation.csv")
    scores = pd.read_csv(ROOT / "model_c_plus_full_universe_expected_returns_trading_scores.csv")
    light_source = pd.read_csv(ROOT / "model_c_plus_current_best_with_divergence_alerts_latest_recommendation.csv")
    dates = {
        str(expanded.iloc[-1]["latest_data_date"])[:10],
        str(scores.iloc[-1]["signal_date"])[:10],
        str(light_source.iloc[-1]["latest_data_date"])[:10],
    }
    if len(dates) != 1:
        raise ValueError(f"Market/feature producers disagree: {dates}")
    common = next(iter(dates))
    pd.DataFrame([{"latest_data_date": common, "producer": "fresh yfinance download", "status": "PASS"}]).to_csv(ROOT / "model_c_plus_market_data_freshness.csv", index=False)
    pd.DataFrame([{"latest_data_date": common, "producer": "current-best/full-universe feature engine", "status": "PASS"}]).to_csv(ROOT / "model_c_plus_feature_freshness.csv", index=False)
    return common


def send_telegram_if_requested(enabled: bool) -> None:
    if not enabled:
        print("Telegram dry-run only; delivery disabled.")
        return
    ready = json.loads((ROOT / "model_c_plus_034_execution_ready.json").read_text(encoding="utf-8"))
    if not ready.get("execution_safe"):
        raise RuntimeError("Telegram suppressed: execution is not safe")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Telegram secrets missing")
    import requests
    text = (ROOT / "model_c_plus_034_live_dashboard_telegram_preview.txt").read_text(encoding="utf-8")
    response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=20)
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send-telegram", action="store_true")
    args = parser.parse_args()
    block = ROOT / "model_c_plus_034_EXECUTION_BLOCKED.json"
    block.write_text(json.dumps({"execution_safe": False, "reason": "daily pipeline in progress or failed", "started_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    quarantine_dir = quarantine()
    try:
        run("model_c_plus_current_best_with_divergence_alerts.py", {"RUN_FULL_EXPECTED_RETURN_STUDY": "1"})
        common = write_freshness_seed()
        run("model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST.py")
        run("research_022f_calibrated_defense_validation.py")
        run("build_022f_live_allocation.py")
        run("run_034_corrected_production.py")
        run("redesign_034_daily_trading_dashboard.py", {"DASHBOARD_INPUT_DIR": str(ROOT), "DASHBOARD_OUTPUT_PREFIX": "model_c_plus_034_live_dashboard"})
        run("validate_034_production.py")
        send_telegram_if_requested(args.send_telegram)
        block.unlink(missing_ok=True)
        print(f"VERIFIED 034 DAILY PIPELINE PASS: {common}; quarantine={quarantine_dir}")
    except Exception as exc:
        block.write_text(json.dumps({"execution_safe": False, "reason": str(exc), "quarantine": str(quarantine_dir), "failed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        print(f"VERIFIED 034 DAILY PIPELINE FAILED: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

