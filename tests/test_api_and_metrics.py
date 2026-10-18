"""API contract tests (need the committed model artifact) and metric sanity checks."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fraud_detection import evaluation  # noqa: E402

HAS_MODEL = (ROOT / "models" / "fraud_lgbm.txt").exists()
TXN = {"step": 10, "type": "TRANSFER", "amount": 181.0, "nameOrig": "C1", "oldbalanceOrg": 181.0,
       "nameDest": "C2", "oldbalanceDest": 0.0}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from service.app import app
    return TestClient(app)


@pytest.mark.skipif(not HAS_MODEL, reason="model artifact not built")
def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.skipif(not HAS_MODEL, reason="model artifact not built")
def test_transfer_is_scored_with_valid_probability(client):
    r = client.post("/score", json=TXN).json()
    assert r["scored"] is True
    assert 0.0 <= r["fraud_probability"] <= 1.0
    assert isinstance(r["alert"], bool)
    # amount == full balance: the drain rule must fire and force an alert
    assert r["rules_triggered"] == ["drains_account"] and r["alert"] is True


@pytest.mark.skipif(not HAS_MODEL, reason="model artifact not built")
def test_partial_transfer_does_not_trigger_rule(client):
    r = client.post("/score", json={**TXN, "amount": 50.0, "nameDest": "C9"}).json()
    assert r["rules_triggered"] == []


@pytest.mark.skipif(not HAS_MODEL, reason="model artifact not built")
def test_payment_bypasses_model(client):
    r = client.post("/score", json={**TXN, "type": "PAYMENT", "nameDest": "M1"}).json()
    assert r["scored"] is False and r["fraud_probability"] is None


@pytest.mark.skipif(not HAS_MODEL, reason="model artifact not built")
def test_post_transaction_fields_are_rejected(client):
    assert client.post("/score", json={**TXN, "newbalanceOrig": 0.0}).status_code == 422


@pytest.mark.skipif(not HAS_MODEL, reason="model artifact not built")
def test_negative_amount_rejected(client):
    assert client.post("/score", json={**TXN, "amount": -5}).status_code == 422


def test_at_budget_counts():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    amt = np.array([100, 1, 50, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    r = evaluation.at_budget(y, s, amt, budget=0.2)
    assert r["alerts"] == 2 and r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["value_recall"] == pytest.approx(100 / 150)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert evaluation.psi(a, rng.normal(0, 1, 20000)) < 0.02
    assert evaluation.psi(a, rng.normal(1, 1, 20000)) > 0.25
