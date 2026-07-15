"""
Production runner: VolTarget011 Light + Early Alert Dashboard

What this does
--------------
1. Runs your existing production model script.
2. Reads latest recommendation CSV.
3. Compares current saved portfolio vs latest model portfolio.
4. Triggers alert if:

       portfolio_diff >= 1.20
       score_gap >= 0.005
       cooldown >= 5 trading days

5. Writes alert fields into the latest dashboard CSV.
6. Appends dashboard history.
7. Sends Telegram message.

Files this script uses
----------------------
Input/output latest CSV:
    model_c_plus_vol_target_leverage_budget_010_latest_recommendation.csv

State file:
    current_portfolio_state_011.json

Output dashboard CSV:
    model_c_plus_vol_target_leverage_budget_010_latest_recommendation_ALERT_DASHBOARD.csv

Output history CSV:
    model_c_plus_vol_target_leverage_budget_010_alert_dashboard_history.csv

How to use
----------
Normal daily GitHub run:
    python run_011_daily_with_alert_dashboard.py

After you actually rebalance manually:
    python run_011_daily_with_alert_dashboard.py --mark-rebalanced

Environment variables / GitHub Secrets
--------------------------------------
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID

Optional:
MODEL_SCRIPT_NAME
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# The script will try these in order if MODEL_SCRIPT_NAME is not set.
MODEL_SCRIPT_CANDIDATES = [
    "model_c_plus_current_best_LIGHT_PRODUCTION_DASHBOARD.py",
    "model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION.py",
    "model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST.py",
    "model_c_plus_vol_target_leverage_budget_010.py",
]

LATEST_CSV = SCRIPT_DIR / "model_c_plus_vol_target_leverage_budget_010_latest_recommendation.csv"

ALERT_DASHBOARD_CSV = SCRIPT_DIR / "model_c_plus_vol_target_leverage_budget_010_latest_recommendation_ALERT_DASHBOARD.csv"
ALERT_HISTORY_CSV = SCRIPT_DIR / "model_c_plus_vol_target_leverage_budget_010_alert_dashboard_history.csv"

STATE_FILE = SCRIPT_DIR / "current_portfolio_state_011.json"

EXEC_ASSETS = ["TQQQ", "ERX", "UXI", "QQQM", "XLE", "XSOE", "XLI", "XLB", "BIL"]

PORTFOLIO_DIFF_THRESHOLD = 1.20
SCORE_GAP_THRESHOLD = 0.005
COOLDOWN_TRADING_DAYS = 5

TRANSACTION_COST = 0.001


# ============================================================
# BASIC HELPERS
# ============================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def pct(x) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "N/A"


def fmt(x, digits=4) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def normalize_weights(w: dict) -> dict:
    out = {}
    for a in EXEC_ASSETS:
        out[a] = max(0.0, safe_float(w.get(a, 0.0), 0.0))
    s = sum(out.values())
    if s <= 0:
        out = {a: 0.0 for a in EXEC_ASSETS}
        out["BIL"] = 1.0
        return out
    return {k: v / s for k, v in out.items()}


def portfolio_diff(old_w: dict, new_w: dict) -> float:
    old_w = normalize_weights(old_w)
    new_w = normalize_weights(new_w)
    return float(sum(abs(new_w.get(a, 0.0) - old_w.get(a, 0.0)) for a in EXEC_ASSETS))


def leveraged_weight(w: dict) -> float:
    return float(w.get("TQQQ", 0.0) + w.get("UXI", 0.0) + w.get("ERX", 0.0))


def format_weights(w: dict, min_weight=0.004) -> str:
    w = normalize_weights(w)
    lines = []
    for a in EXEC_ASSETS:
        if abs(w.get(a, 0.0)) >= min_weight:
            lines.append(f"{a}: {pct(w[a])}")
    return "\n".join(lines) if lines else "No active weights"


# ============================================================
# MODEL RUN
# ============================================================

def choose_model_script() -> Path:
    env_name = os.getenv("MODEL_SCRIPT_NAME", "").strip()
    if env_name:
        p = SCRIPT_DIR / env_name
        if not p.exists():
            raise FileNotFoundError(f"MODEL_SCRIPT_NAME was set but file not found: {p}")
        return p

    for name in MODEL_SCRIPT_CANDIDATES:
        p = SCRIPT_DIR / name
        if p.exists():
            return p

    raise FileNotFoundError(
        "No production model script found. Set GitHub secret/env MODEL_SCRIPT_NAME, or add one of:\n"
        + "\n".join(MODEL_SCRIPT_CANDIDATES)
    )


def run_model() -> Path:
    model_script = choose_model_script()
    print("Running model script:", model_script)

    subprocess.run([sys.executable, str(model_script)], cwd=str(SCRIPT_DIR), check=True)

    if not LATEST_CSV.exists():
        raise FileNotFoundError(f"Latest recommendation CSV not found after model run: {LATEST_CSV}")

    return model_script


# ============================================================
# LATEST ROW PARSING
# ============================================================

def extract_weights(row: dict) -> dict:
    w = {}
    for a in EXEC_ASSETS:
        w[a] = safe_float(row.get(f"exec_w_{a}", 0.0), 0.0)
    return normalize_weights(w)


def extract_signal_date(row: dict) -> str:
    for c in ["signal_date", "date", "latest_data_date", "aligned_date"]:
        if c in row and str(row[c]) not in ["nan", "NaT", "None", ""]:
            return str(row[c])[:10]
    return "N/A"


def extract_score_gap(row: dict) -> float:
    for c in ["score_gap", "score_gap_inferred"]:
        if c in row:
            return safe_float(row.get(c), np.nan)

    # fallback from score columns
    top = safe_float(row.get("top_score"), np.nan)
    second = safe_float(row.get("second_score"), np.nan)
    if not pd.isna(top) and not pd.isna(second):
        return top - second

    return np.nan


def extract_top_asset(row: dict) -> str:
    for c in ["top_asset", "top_asset_inferred"]:
        if c in row and str(row[c]) not in ["nan", "NaT", "None", ""]:
            return str(row[c])
    return "N/A"


def load_latest_row():
    df = pd.read_csv(LATEST_CSV)
    if df.empty:
        raise ValueError(f"Latest CSV is empty: {LATEST_CSV}")
    latest = df.iloc[-1].to_dict()
    return df, latest


# ============================================================
# STATE AND ALERT LOGIC
# ============================================================

def default_state(latest_weights: dict | None = None) -> dict:
    if latest_weights is None:
        latest_weights = {a: 0.0 for a in EXEC_ASSETS}
        latest_weights["BIL"] = 1.0

    return {
        "current_portfolio": normalize_weights(latest_weights),
        "last_rebalance_date": None,
        "last_alert_date": None,
        "trading_days_since_alert": 999,
        "trading_days_since_rebalance": 999,
        "updated_at_utc": now_utc_iso(),
    }


def update_day_counters(state: dict) -> dict:
    # GitHub runs daily on weekdays. This approximates trading days.
    state["trading_days_since_alert"] = int(state.get("trading_days_since_alert", 999)) + 1
    state["trading_days_since_rebalance"] = int(state.get("trading_days_since_rebalance", 999)) + 1
    return state


def evaluate_alert(state: dict, latest_weights: dict, score_gap: float) -> dict:
    current_w = normalize_weights(state.get("current_portfolio", {}))
    latest_weights = normalize_weights(latest_weights)

    diff = portfolio_diff(current_w, latest_weights)

    cooldown_days = int(state.get("trading_days_since_alert", 999))
    cooldown_ok = cooldown_days >= COOLDOWN_TRADING_DAYS

    score_gap_ok = (not pd.isna(score_gap)) and score_gap >= SCORE_GAP_THRESHOLD
    diff_ok = diff >= PORTFOLIO_DIFF_THRESHOLD

    is_alert = bool(diff_ok and score_gap_ok and cooldown_ok)

    reasons = []
    reasons.append(f"portfolio_diff={diff:.4f} threshold={PORTFOLIO_DIFF_THRESHOLD:.2f}")
    reasons.append(f"score_gap={fmt(score_gap)} threshold={SCORE_GAP_THRESHOLD:.3f}")
    reasons.append(f"cooldown_days={cooldown_days} threshold={COOLDOWN_TRADING_DAYS}")

    if is_alert:
        status = "ALERT"
        alert_type = "EARLY_REBALANCE_ALERT"
        action = "Review early rebalance before normal schedule."
    else:
        status = "NO_ALERT"
        alert_type = "NONE"
        action = "No early rebalance alert. Follow normal schedule."

    return {
        "alert_status": status,
        "alert_type": alert_type,
        "alert_reason": "; ".join(reasons),
        "is_early_rebalance_alert": is_alert,
        "portfolio_diff": diff,
        "score_gap": score_gap,
        "cooldown_ok": cooldown_ok,
        "score_gap_ok": score_gap_ok,
        "portfolio_diff_ok": diff_ok,
        "suggested_action": action,
        "current_portfolio_before_alert": current_w,
    }


def mark_rebalanced(state: dict, latest_weights: dict, signal_date: str) -> dict:
    state["current_portfolio"] = normalize_weights(latest_weights)
    state["last_rebalance_date"] = signal_date
    state["trading_days_since_rebalance"] = 0
    state["updated_at_utc"] = now_utc_iso()
    return state


def mark_alert_sent(state: dict, signal_date: str) -> dict:
    state["last_alert_date"] = signal_date
    state["trading_days_since_alert"] = 0
    state["updated_at_utc"] = now_utc_iso()
    return state


# ============================================================
# DASHBOARD OUTPUT
# ============================================================

def write_alert_dashboard(original_df: pd.DataFrame, latest_row: dict, alert: dict, state: dict, model_script: Path, latest_weights: dict) -> dict:
    enriched = dict(latest_row)

    signal_date = extract_signal_date(latest_row)
    top_asset = extract_top_asset(latest_row)

    current_w = alert["current_portfolio_before_alert"]

    enriched.update({
        "alert_run_time_utc": now_utc_iso(),
        "alert_status": alert["alert_status"],
        "alert_type": alert["alert_type"],
        "is_early_rebalance_alert": alert["is_early_rebalance_alert"],
        "alert_reason": alert["alert_reason"],
        "suggested_action": alert["suggested_action"],
        "portfolio_diff": alert["portfolio_diff"],
        "portfolio_diff_threshold": PORTFOLIO_DIFF_THRESHOLD,
        "score_gap_used_for_alert": alert["score_gap"],
        "score_gap_threshold": SCORE_GAP_THRESHOLD,
        "cooldown_days_since_last_alert": state.get("trading_days_since_alert", 999),
        "cooldown_threshold_days": COOLDOWN_TRADING_DAYS,
        "cooldown_ok": alert["cooldown_ok"],
        "portfolio_diff_ok": alert["portfolio_diff_ok"],
        "score_gap_ok": alert["score_gap_ok"],
        "model_script_used": model_script.name,
        "alert_signal_date": signal_date,
        "alert_top_asset": top_asset,
        "current_leveraged_weight_before_alert": leveraged_weight(current_w),
        "new_model_leveraged_weight": leveraged_weight(latest_weights),
    })

    for a in EXEC_ASSETS:
        enriched[f"current_w_{a}"] = current_w.get(a, 0.0)
        enriched[f"new_model_w_{a}"] = latest_weights.get(a, 0.0)

    out_df = pd.DataFrame([enriched])
    out_df.to_csv(ALERT_DASHBOARD_CSV, index=False)

    if ALERT_HISTORY_CSV.exists():
        hist = pd.read_csv(ALERT_HISTORY_CSV)
        hist = pd.concat([hist, out_df], ignore_index=True)
    else:
        hist = out_df

    # Keep one row per alert run time / signal date combination.
    hist.to_csv(ALERT_HISTORY_CSV, index=False)

    return enriched


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[TELEGRAM SKIPPED] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        print(text)
        return False

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, data=payload, timeout=20)
        print("Telegram status:", r.status_code)
        return r.ok
    except Exception as e:
        print("Telegram send failed:", e)
        print(text)
        return False


def build_message(alert: dict, latest_row: dict, latest_weights: dict, state: dict) -> str:
    signal_date = extract_signal_date(latest_row)
    top_asset = extract_top_asset(latest_row)
    score_gap = alert["score_gap"]

    current_w = alert["current_portfolio_before_alert"]

    emoji = "🚨" if alert["is_early_rebalance_alert"] else "✅"
    title = "VolTarget011 EARLY REBALANCE ALERT" if alert["is_early_rebalance_alert"] else "VolTarget011 Daily Dashboard"

    return f"""{emoji} <b>{title}</b>

Date: {signal_date}
Top asset: {top_asset}
Alert status: <b>{alert['alert_status']}</b>

Portfolio diff: {fmt(alert['portfolio_diff'])}
Score gap: {fmt(score_gap)}
Cooldown days: {state.get('trading_days_since_alert', 999)}

<b>Current saved portfolio</b>
{format_weights(current_w)}

<b>New model portfolio</b>
{format_weights(latest_weights)}

<b>Reason</b>
{alert['alert_reason']}

<b>Action</b>
{alert['suggested_action']}
""".strip()


# ============================================================
# MAIN
# ============================================================

def main():
    mark_rebalanced_flag = "--mark-rebalanced" in sys.argv
    no_run_model_flag = "--no-run-model" in sys.argv

    print("\n=== VolTarget011 Daily Production + Alert Dashboard ===")
    print("Folder:", SCRIPT_DIR)

    if not no_run_model_flag:
        model_script = run_model()
    else:
        model_script = choose_model_script()
        print("Skipping model run because --no-run-model was supplied.")

    original_df, latest_row = load_latest_row()
    latest_weights = extract_weights(latest_row)
    signal_date = extract_signal_date(latest_row)
    score_gap = extract_score_gap(latest_row)

    # Initialize state if missing.
    state = load_json(STATE_FILE, default_state(latest_weights))

    if mark_rebalanced_flag:
        state = mark_rebalanced(state, latest_weights, signal_date)
        save_json(STATE_FILE, state)

        text = f"""✅ <b>VolTarget011 marked as rebalanced</b>

Date: {signal_date}

Saved current portfolio:
{format_weights(latest_weights)}
"""
        send_telegram(text)
        print(text)
        return

    state = update_day_counters(state)
    alert = evaluate_alert(state, latest_weights, score_gap)

    # Mark alert cooldown only if actual alert fired.
    if alert["is_early_rebalance_alert"]:
        state = mark_alert_sent(state, signal_date)

    enriched = write_alert_dashboard(original_df, latest_row, alert, state, model_script, latest_weights)
    save_json(STATE_FILE, state)

    message = build_message(alert, latest_row, latest_weights, state)
    sent = send_telegram(message)

    print("\n" + message)
    print("\nTelegram sent:", sent)
    print("Saved:", ALERT_DASHBOARD_CSV)
    print("Saved:", ALERT_HISTORY_CSV)
    print("Saved:", STATE_FILE)

    # Important line for GitHub logs.
    print(f"\nDASHBOARD ALERT STATUS: {alert['alert_status']}")
    print(f"EARLY REBALANCE ALERT: {alert['is_early_rebalance_alert']}")


if __name__ == "__main__":
    main()
