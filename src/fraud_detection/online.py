"""Online feature computation for real-time scoring.

``FeatureStore`` keeps per-destination running state in memory. In production
this would be Redis or a managed feature store keyed by account id; the
interface (read state -> compute features -> score -> update state) stays the
same. The arithmetic mirrors ``features.py`` exactly; the parity test enforces it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SCORED_TYPES
from .features import CAP


@dataclass
class _DestState:
    n: int = 0
    amount_sum: float = 0.0
    last_step: int | None = None


class FeatureStore:
    def __init__(self):
        self._dest: dict[str, _DestState] = {}

    def __len__(self) -> int:
        return len(self._dest)

    def features(self, txn: dict) -> dict:
        """Features for one transaction using only state from earlier transactions."""
        s = self._dest.get(txn["nameDest"], _DestState())
        amount, old_o, old_d, step = txn["amount"], txn["oldbalanceOrg"], txn["oldbalanceDest"], txn["step"]
        return {
            "is_transfer": int(txn["type"] == "TRANSFER"),
            "log_amount": math.log1p(amount),
            "hour": step % 24,
            "amount_round_1000": int(amount % 1000 == 0),
            "amount_at_cap": int(amount >= CAP),
            "dest_n_prev": s.n,
            "dest_log_amt_prev_mean": math.log1p(s.amount_sum / max(s.n, 1)),
            "dest_steps_since_last": float(step - s.last_step) if s.last_step is not None else float("nan"),
            "dest_is_new": int(s.n == 0),
            "log_oldbalance_orig": math.log1p(old_o),
            "amount_to_balance": min(amount / (old_o + 1.0), 1e4),
            "drains_account": int(old_o > 0 and abs(old_o - amount) < 0.01),
            "orig_balance_zero": int(old_o == 0),
            "log_oldbalance_dest": math.log1p(old_d),
            "dest_balance_zero": int(old_d == 0),
        }

    def update(self, txn: dict) -> None:
        """Record a transaction (of any type) once it has been processed."""
        s = self._dest.setdefault(txn["nameDest"], _DestState())
        s.n += 1
        s.amount_sum += txn["amount"]
        s.last_step = txn["step"]

    @staticmethod
    def is_scored(txn: dict) -> bool:
        return txn["type"] in SCORED_TYPES
