"""Offline (batch) feature computation.

Destination-history features are computed over *all* transaction types in
arrival order and only ever look at earlier transactions, exactly as the online
feature store in ``online.py`` does. ``tests/test_parity.py`` checks the two
implementations agree row for row, which is the guard against training/serving
skew.
"""
from __future__ import annotations

import polars as pl

from .config import PARQUET, RAW_CSV, SCORED_TYPES

CAP = 10_000_000.0  # PaySim caps single transactions at 10M


def load_raw() -> pl.DataFrame:
    if PARQUET.exists():
        return pl.read_parquet(PARQUET)
    df = pl.read_csv(RAW_CSV)
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PARQUET)
    return df


def add_destination_history(df: pl.DataFrame) -> pl.DataFrame:
    """Per-destination running stats from strictly earlier transactions."""
    df = df.with_row_index("txn_id")
    return df.with_columns(
        dest_n_prev=(pl.col("txn_id").cum_count().over("nameDest") - 1).cast(pl.Int32),
        dest_amt_prev_sum=(pl.col("amount").cum_sum().over("nameDest") - pl.col("amount")),
        dest_last_step=pl.col("step").shift(1).over("nameDest"),
    )


def transaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Row-level features; expects the destination-history columns to exist."""
    log1p = lambda c: pl.col(c).log1p()  # noqa: E731
    return df.with_columns(
        is_transfer=(pl.col("type") == "TRANSFER").cast(pl.Int8),
        log_amount=log1p("amount"),
        hour=(pl.col("step") % 24).cast(pl.Int8),
        amount_round_1000=((pl.col("amount") % 1000) == 0).cast(pl.Int8),
        amount_at_cap=(pl.col("amount") >= CAP).cast(pl.Int8),
        dest_log_amt_prev_mean=(pl.col("dest_amt_prev_sum") / pl.col("dest_n_prev").clip(lower_bound=1)).log1p(),
        dest_steps_since_last=(pl.col("step") - pl.col("dest_last_step")).cast(pl.Float32),
        dest_is_new=(pl.col("dest_n_prev") == 0).cast(pl.Int8),
        log_oldbalance_orig=log1p("oldbalanceOrg"),
        amount_to_balance=(pl.col("amount") / (pl.col("oldbalanceOrg") + 1.0)).clip(upper_bound=1e4),
        drains_account=((pl.col("oldbalanceOrg") > 0) & ((pl.col("oldbalanceOrg") - pl.col("amount")).abs() < 0.01)).cast(pl.Int8),
        orig_balance_zero=(pl.col("oldbalanceOrg") == 0).cast(pl.Int8),
        log_oldbalance_dest=log1p("oldbalanceDest"),
        dest_balance_zero=(pl.col("oldbalanceDest") == 0).cast(pl.Int8),
        # --- post-transaction (leaky) ---
        log_newbalance_orig=log1p("newbalanceOrig"),
        log_newbalance_dest=log1p("newbalanceDest"),
        error_balance_orig=(pl.col("oldbalanceOrg") - pl.col("amount") - pl.col("newbalanceOrig")),
        error_balance_dest=(pl.col("oldbalanceDest") + pl.col("amount") - pl.col("newbalanceDest")),
    )


def build(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Full offline feature table for the scored transaction types."""
    df = load_raw() if df is None else df
    df = add_destination_history(df)
    df = df.filter(pl.col("type").is_in(SCORED_TYPES))
    return transaction_features(df)
