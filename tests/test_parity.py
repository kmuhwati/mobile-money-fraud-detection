"""Training/serving skew guard: batch features must equal online features row for row."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection import features  # noqa: E402
from fraud_detection.config import TIERS  # noqa: E402
from fraud_detection.online import FeatureStore  # noqa: E402


def synthetic_stream(n=3000, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], n)
    amount = np.round(rng.lognormal(10, 2, n), 2)
    amount[::50] = 1000.0 * rng.integers(1, 50, len(amount[::50]))      # round amounts
    old = np.round(rng.lognormal(10, 2, n), 2)
    old[::7] = 0.0
    drain = rng.random(n) < 0.05
    amount[drain] = old[drain]
    return pl.DataFrame({
        "step": np.sort(rng.integers(1, 200, n)),
        "type": types,
        "amount": amount,
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": old,
        "newbalanceOrig": np.maximum(old - amount, 0),
        "nameDest": [f"C{j}" for j in rng.integers(0, 300, n)],   # many repeat destinations
        "oldbalanceDest": np.round(rng.lognormal(9, 2, n), 2) * (rng.random(n) > 0.3),
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    })


def test_offline_and_online_features_match():
    stream = synthetic_stream()
    offline = features.build(stream).to_pandas()
    store, online = FeatureStore(), []
    for txn in stream.iter_rows(named=True):
        if FeatureStore.is_scored(txn):
            online.append(store.features(txn))
        store.update(txn)
    assert len(online) == len(offline)
    for col in TIERS["B_authorisation_time"]:
        a = offline[col].to_numpy(float)
        b = np.array([r[col] for r in online], dtype=float)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=col)


def test_destination_history_uses_only_the_past():
    stream = synthetic_stream(n=500)
    offline = features.build(stream).to_pandas()
    first_seen = offline.groupby("nameDest").txn_id.idxmin()
    # The first scored transaction to a destination that was never paid before must have n_prev == 0
    all_rows = features.add_destination_history(stream).to_pandas()
    first_any = all_rows.groupby("nameDest").txn_id.min()
    for dest, idx in first_seen.items():
        if offline.loc[idx, "txn_id"] == first_any[dest]:
            assert offline.loc[idx, "dest_n_prev"] == 0


@pytest.mark.parametrize("old,amount,expected", [(100.0, 100.0, 1), (100.0, 99.0, 0), (0.0, 0.0, 0)])
def test_drains_account_rule(old, amount, expected):
    txn = {"type": "TRANSFER", "amount": amount, "oldbalanceOrg": old, "oldbalanceDest": 0.0, "step": 1, "nameDest": "C1"}
    assert FeatureStore().features(txn)["drains_account"] == expected
