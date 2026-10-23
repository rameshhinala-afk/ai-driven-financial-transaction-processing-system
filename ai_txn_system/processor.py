"""Transaction processing pipeline:
validate -> build features -> AI fraud score -> decide -> settle -> record.
"""
from datetime import datetime, timedelta

import database as db

HOME_COUNTRY = "IN"
BLOCK_THRESHOLD = 0.80    # score >= this  -> BLOCKED
REVIEW_THRESHOLD = 0.40   # score >= this  -> REVIEW (manual check)


class TransactionProcessor:
    def __init__(self, conn, fraud_model):
        self.conn = conn
        self.model = fraud_model

    # ---------- feature engineering ----------
    def _build_features(self, sender, amount, country, ts):
        history = db.sender_history(self.conn, sender)

        if history:
            avg_amount = sum(h["amount"] for h in history) / len(history)
        else:
            avg_amount = amount  # no history -> ratio of 1
        amount_ratio = amount / avg_amount if avg_amount > 0 else 1.0

        one_hour_ago = ts - timedelta(hours=1)
        txn_count_1h = sum(
            1 for h in history
            if one_hour_ago <= datetime.fromisoformat(h["timestamp"]) <= ts
        )

        return {
            "amount": amount,
            "hour": ts.hour,
            "is_foreign": int(country != HOME_COUNTRY),
            "txn_count_1h": txn_count_1h,
            "amount_ratio": amount_ratio,
        }

    # ---------- main entry point ----------
    def process(self, sender, receiver, amount, country=HOME_COUNTRY, timestamp=None):
        ts = timestamp or datetime.now()
        ts_str = ts.isoformat(timespec="seconds")

        def finish(status, reason, score=None):
            txn_id = db.save_transaction(
                self.conn, sender, receiver, amount, country,
                ts_str, score, status, reason,
            )
            return {"txn_id": txn_id, "status": status,
                    "fraud_score": score, "reason": reason}

        # 1. Validation
        if amount <= 0:
            return finish("REJECTED", "Amount must be positive")
        if sender == receiver:
            return finish("REJECTED", "Sender and receiver are the same")

        src = db.get_account(self.conn, sender)
        dst = db.get_account(self.conn, receiver)
        if src is None or dst is None:
            return finish("REJECTED", "Unknown account")
        if src["balance"] < amount:
            return finish("REJECTED", "Insufficient funds")

        # 2. AI fraud scoring
        features = self._build_features(sender, amount, country, ts)
        score = round(self.model.predict_proba(**features), 3)

        # 3. Decision
        if score >= BLOCK_THRESHOLD:
            return finish("BLOCKED", "High fraud risk", score)
        if score >= REVIEW_THRESHOLD:
            return finish("REVIEW", "Suspicious - needs manual review", score)

        # 4. Settlement (atomic)
        try:
            db.update_balance(self.conn, sender, -amount)
            db.update_balance(self.conn, receiver, +amount)
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            return finish("REJECTED", f"Settlement failed: {exc}", score)

        return finish("APPROVED", "Low fraud risk", score)
