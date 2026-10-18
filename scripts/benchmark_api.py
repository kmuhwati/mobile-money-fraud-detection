"""Replay real test-period transactions through the API in arrival order and time it.

Uses FastAPI's in-process TestClient, so the numbers include request parsing,
validation, feature-store lookups, model inference, SHAP reason codes and
response serialisation, but not network time.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from fraud_detection.config import PARQUET, REPORTS, VALID_END  # noqa: E402
from service.app import app  # noqa: E402

N_WARM, N_REQ = 20_000, 5_000
FIELDS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"]


def main():
    df = pl.read_parquet(PARQUET).with_row_index("i")
    start = df.filter(pl.col("step") > VALID_END)["i"].min()
    stream = df.slice(start - N_WARM, N_WARM + N_REQ).select(FIELDS + ["isFraud"])
    client = TestClient(app)

    rows = stream.iter_rows(named=True)
    for _ in range(N_WARM):            # warm the feature store with recent history
        r = next(rows)
        client.post("/score", json={k: r[k] for k in FIELDS})

    wall, server, server_scored, scored, alerts, frauds_alerted = [], [], [], 0, 0, 0
    for r in rows:
        t0 = time.perf_counter()
        resp = client.post("/score", json={k: r[k] for k in FIELDS}).json()
        wall.append((time.perf_counter() - t0) * 1e3)
        server.append(resp["latency_ms"])
        if resp["scored"]:
            server_scored.append(resp["latency_ms"])
            scored += 1
            alerts += resp["alert"]
            frauds_alerted += resp["alert"] and r["isFraud"] == 1

    q = lambda xs, p: statistics.quantiles(xs, n=100)[p - 1]  # noqa: E731
    out = {
        "requests": len(wall), "scored": scored, "alerts": alerts, "frauds_alerted": frauds_alerted,
        "client_ms": {"p50": q(wall, 50), "p95": q(wall, 95), "p99": q(wall, 99)},
        "handler_ms_all": {"p50": q(server, 50), "p95": q(server, 95), "p99": q(server, 99)},
        "handler_ms_scored_transfers_cashouts": {"p50": q(server_scored, 50), "p95": q(server_scored, 95), "p99": q(server_scored, 99)},
        "throughput_rps_single_worker": len(wall) / (sum(wall) / 1e3),
    }
    (REPORTS / "api_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
