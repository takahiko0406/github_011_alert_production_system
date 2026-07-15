# VolTarget011 Daily Alert Dashboard Production System

## What it does

Every weekday before the US market, GitHub Actions runs your model and writes alert status into the dashboard.

Alert rule:

```text
portfolio_diff >= 1.20
score_gap >= 0.005
cooldown >= 5 trading days
```

If triggered, dashboard CSV will show:

```text
alert_status = ALERT
alert_type = EARLY_REBALANCE_ALERT
```

If not triggered:

```text
alert_status = NO_ALERT
```

## Files

Add these to your repository:

```text
run_011_daily_with_alert_dashboard.py
.github/workflows/run_voltarget011_daily_alert_dashboard.yml
```

## Required GitHub Secrets

Repository → Settings → Secrets and variables → Actions → New repository secret

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

## Recommended GitHub Variable

Repository → Settings → Secrets and variables → Actions → Variables

```text
MODEL_SCRIPT_NAME
```

Example:

```text
model_c_plus_current_best_LIGHT_PRODUCTION_DASHBOARD.py
```

If you do not set it, the runner tries these files:

```text
model_c_plus_current_best_LIGHT_PRODUCTION_DASHBOARD.py
model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION.py
model_c_plus_transition_conviction_overlay_011_LIGHT_EXECUTION_AND_RIGOROUS_TEST.py
model_c_plus_vol_target_leverage_budget_010.py
```

## Output files

```text
model_c_plus_vol_target_leverage_budget_010_latest_recommendation_ALERT_DASHBOARD.csv
model_c_plus_vol_target_leverage_budget_010_alert_dashboard_history.csv
current_portfolio_state_011.json
```

## Manual mark after you actually rebalance

After you actually trade, run the workflow manually with:

```text
mark_rebalanced = true
```

This updates `current_portfolio_state_011.json`.

That is important because the alert system compares:

```text
current saved portfolio
vs
new model portfolio
```
