"""Paths, time-based splits, feature tiers and operating constraints."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "paysim.csv"
PARQUET = ROOT / "data" / "processed" / "paysim.parquet"
FEATURES_PARQUET = ROOT / "data" / "processed" / "features.parquet"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

SEED = 11

# PaySim only generates fraud in these two types. Other types bypass the model
# (a documented business rule, not a learned one).
SCORED_TYPES = ("TRANSFER", "CASH_OUT")

# One step = one simulated hour (743 steps ~ 31 days). Splits are by time so the
# model is always evaluated on transactions from *after* everything it saw.
TRAIN_END = 400      # steps 1..400   (~16.7 days)
VALID_END = 550      # steps 401..550 (~6 days): early stopping, calibration, thresholds
#                      steps 551..743 (~8 days): untouched test period

# Fraud-ops capacity: analysts can review this share of scored transactions.
ALERT_BUDGET = 0.02

# --- feature tiers --------------------------------------------------------------
# A: everything in the file, including post-transaction balances. PaySim cancels
#    fraudulent transactions, so these columns encode the label. Shown only to
#    quantify the leak.
# B: what an operator knows at authorisation time: amount, type, hour, the
#    pre-transaction balances it holds, and the destination's past activity.
# C: B without balances. Robust to PaySim's balance artefacts; the conservative
#    estimate of real-world performance.
BEHAVIOURAL = [
    "is_transfer", "log_amount", "hour", "amount_round_1000", "amount_at_cap",
    "dest_n_prev", "dest_log_amt_prev_mean", "dest_steps_since_last", "dest_is_new",
]
BALANCE = [
    "log_oldbalance_orig", "amount_to_balance", "drains_account", "orig_balance_zero",
    "log_oldbalance_dest", "dest_balance_zero",
]
POST_TRANSACTION = [
    "log_newbalance_orig", "log_newbalance_dest", "error_balance_orig", "error_balance_dest", "isFlaggedFraud",
]
TIERS = {
    "A_all_columns_leaky": BEHAVIOURAL + BALANCE + POST_TRANSACTION,
    "B_authorisation_time": BEHAVIOURAL + BALANCE,
    "C_behavioural_only": BEHAVIOURAL,
}
DEPLOY_TIER = "C_behavioural_only"
