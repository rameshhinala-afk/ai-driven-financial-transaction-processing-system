# AI-Driven Financial Transaction Processing System

A simple base project that processes money transfers and uses a machine-learning
model to detect fraud before a transaction is settled.

## How it works

```
Transaction -> Validation -> Feature building -> AI fraud score -> Decision -> Settlement -> Database
```

| Fraud score | Decision | Meaning                          |
|-------------|----------|----------------------------------|
| < 0.40      | APPROVED | Money is moved between accounts  |
| 0.40 - 0.79 | REVIEW   | Held for manual review           |
| >= 0.80     | BLOCKED  | Rejected as likely fraud         |

Validation failures (bad amount, unknown account, insufficient funds) are `REJECTED`.

## Files

| File             | Purpose                                                   |
|------------------|-----------------------------------------------------------|
| `database.py`    | SQLite tables and queries (accounts, transactions)        |
| `fraud_model.py` | RandomForest fraud model trained on synthetic data        |
| `processor.py`   | Pipeline: validate, score, decide, settle, record         |
| `main.py`        | Demo that runs sample transactions                        |

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Model features

`amount`, `hour` of day, `is_foreign`, `txn_count_1h` (velocity), and
`amount_ratio` (amount vs. the sender's average).

## Ideas for extending

- Train on a real dataset (e.g. the Kaggle credit card fraud dataset)
- Add a REST API with FastAPI / Flask
- Add a web dashboard for the REVIEW queue
- Persist the trained model with `joblib`
- Add unit tests and logging
