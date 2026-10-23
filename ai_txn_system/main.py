"""Demo runner for the AI-Driven Financial Transaction Processing System."""
import os
from datetime import datetime

import database as db
from fraud_model import FraudModel
from processor import TransactionProcessor


def main():
    # Fresh database for each demo run
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    conn = db.get_connection()
    db.init_db(conn)

    # Sample accounts
    db.create_account(conn, "ACC001", "Priya", 50000)
    db.create_account(conn, "ACC002", "Rahul", 20000)
    db.create_account(conn, "ACC003", "Anita", 10000)

    # Train AI model
    model = FraudModel().train()
    processor = TransactionProcessor(conn, model)

    today = datetime.now().replace(minute=0, second=0, microsecond=0)

    # (sender, receiver, amount, country, hour)
    sample = [
        ("ACC001", "ACC002", 60,    "IN", 10),   # normal small payment
        ("ACC001", "ACC003", 75,    "IN", 13),   # normal
        ("ACC002", "ACC003", 40,    "IN", 18),   # normal
        ("ACC001", "ACC002", 120,   "IN", 14),   # slightly bigger, still plausible
        ("ACC001", "ACC003", 30000, "US", 3),    # huge, foreign, 3 AM -> fraud
        ("ACC003", "ACC001", 90,    "IN", 9),    # normal
        ("ACC002", "ACC001", 99999, "IN", 11),   # insufficient funds
        ("ACC001", "ACC001", 100,   "IN", 12),   # same account
        ("ACC002", "ACC003", -50,   "IN", 12),   # invalid amount
        ("ACC001", "ACC002", 8000,  "RU", 2),    # foreign, night, big -> suspicious
    ]

    print("\n=== Processing transactions ===")
    print(f"{'ID':<4}{'From':<8}{'To':<8}{'Amount':>9}  {'Ctry':<5}"
          f"{'Score':>6}  {'Status':<9} Reason")
    print("-" * 80)
    for sender, receiver, amount, country, hour in sample:
        r = processor.process(sender, receiver, amount, country,
                              timestamp=today.replace(hour=hour))
        score = "-" if r["fraud_score"] is None else f"{r['fraud_score']:.2f}"
        print(f"{r['txn_id']:<4}{sender:<8}{receiver:<8}{amount:>9}  {country:<5}"
              f"{score:>6}  {r['status']:<9} {r['reason']}")

    print("\n=== Final balances ===")
    for acc in conn.execute("SELECT * FROM accounts"):
        print(f"{acc['account_id']}  {acc['owner']:<6} {acc['balance']:>10.2f}")

    print("\n=== Summary ===")
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM transactions GROUP BY status ORDER BY status"
    ):
        print(f"{row['status']:<9} {row['n']}")

    conn.close()


if __name__ == "__main__":
    main()
