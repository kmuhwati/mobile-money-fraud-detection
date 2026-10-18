"""Real-time fraud scoring API.

POST /score takes a transaction as the operator sees it at authorisation time
and returns a calibrated fraud probability from the behaviour model, any
business rules that fired, an alert decision and the top reason codes. Post-transaction fields are rejected at the schema level, so the
leak found in the offline audit cannot creep back in through the API.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from fraud_detection.online import FeatureStore

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
MODEL_VERSION = "lgbm-behavioural-v1+drain-rule"


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")  # e.g. newbalanceOrig -> 422

    step: int = Field(ge=0, description="Hour since the start of the simulation")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0)
    nameOrig: str
    oldbalanceOrg: float = Field(ge=0)
    nameDest: str
    oldbalanceDest: float = Field(ge=0)


class ScoreResponse(BaseModel):
    scored: bool
    fraud_probability: float | None = None
    alert: bool = False
    rules_triggered: list[str] = []
    reason_codes: list[str] = []
    model_version: str = MODEL_VERSION
    latency_ms: float


class _Model:
    def __init__(self, model_dir: Path):
        self.booster = lgb.Booster(model_file=str(model_dir / "fraud_lgbm.txt"))
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.features: list[str] = self.meta["features"]
        self.threshold = self.meta["threshold_raw"]
        self.cal_x = np.asarray(self.meta["calibration"]["x"])
        self.cal_y = np.asarray(self.meta["calibration"]["y"])

    def score(self, feats: dict) -> tuple[float, float, list[str]]:
        x = np.array([[feats[f] for f in self.features]], dtype=float)
        # one row per request: a thread pool costs more than it saves
        raw = float(self.booster.predict(x, num_threads=1)[0])
        prob = float(np.interp(raw, self.cal_x, self.cal_y))
        contrib = self.booster.predict(x, pred_contrib=True, num_threads=1)[0][:-1]
        top = np.argsort(-contrib)[:3]
        reasons = [f"{self.features[i]}={feats[self.features[i]]:.3g}" for i in top if contrib[i] > 0]
        return raw, prob, reasons


app = FastAPI(title="Mobile money fraud scoring", version=MODEL_VERSION)
_model = _Model(MODEL_DIR)
_store = FeatureStore()
_lock = threading.Lock()  # feature-store reads and writes must be atomic per request


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION, "destinations_tracked": len(_store)}


@app.get("/model-info")
def model_info() -> dict:
    keep = ("model", "feature_tier", "features", "rules", "alert_budget", "train_steps", "test_metrics")
    return {k: _model.meta[k] for k in keep}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    t0 = time.perf_counter()
    payload = txn.model_dump()
    with _lock:
        if not FeatureStore.is_scored(payload):
            _store.update(payload)
            return ScoreResponse(scored=False, latency_ms=(time.perf_counter() - t0) * 1e3)
        feats = _store.features(payload)
        _store.update(payload)
    raw, prob, reasons = _model.score(feats)
    rules = ["drains_account"] if feats["drains_account"] else []
    return ScoreResponse(scored=True, fraud_probability=prob, alert=bool(rules) or raw >= _model.threshold,
                         rules_triggered=rules, reason_codes=reasons, latency_ms=(time.perf_counter() - t0) * 1e3)
