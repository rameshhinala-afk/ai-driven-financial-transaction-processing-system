"""AI fraud detection model.

Trains a RandomForest on synthetic transaction data and returns a
fraud probability (0-1) for new transactions.

Features:
  amount          - transaction amount
  hour            - hour of day (0-23)
  is_foreign      - 1 if the transaction country differs from the home country
  txn_count_1h    - sender's transactions in the previous hour
  amount_ratio    - amount / sender's average past amount
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = ["amount", "hour", "is_foreign", "txn_count_1h", "amount_ratio"]


def generate_training_data(n_normal=4000, n_fraud=400, seed=42):
    rng = np.random.default_rng(seed)

    # Normal behaviour: modest amounts, daytime, rarely foreign
    normal = np.column_stack([
        rng.lognormal(mean=4.0, sigma=0.7, size=n_normal),         # amount
        rng.choice(np.arange(7, 23), size=n_normal),               # hour
        (rng.random(n_normal) < 0.05).astype(int),                 # is_foreign
        rng.poisson(0.7, size=n_normal),                           # txn_count_1h
        np.clip(rng.normal(1.0, 0.4, size=n_normal), 0.1, None),   # amount_ratio
    ])

    # Fraud behaviour: large amounts, night, foreign, bursts, spikes
    fraud = np.column_stack([
        rng.lognormal(mean=6.5, sigma=0.8, size=n_fraud),
        rng.choice([0, 1, 2, 3, 4, 5, 23], size=n_fraud),
        (rng.random(n_fraud) < 0.65).astype(int),
        rng.poisson(5, size=n_fraud),
        np.clip(rng.normal(6.0, 2.5, size=n_fraud), 0.5, None),
    ])

    X = np.vstack([normal, fraud])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    return X, y


class FraudModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced", random_state=42
        )

    def train(self, verbose=True):
        X, y = generate_training_data()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_tr, y_tr)
        if verbose:
            print("Fraud model trained. Hold-out performance:")
            print(classification_report(
                y_te, self.model.predict(X_te),
                target_names=["legit", "fraud"], digits=3,
            ))
        return self

    def predict_proba(self, amount, hour, is_foreign, txn_count_1h, amount_ratio):
        x = np.array([[amount, hour, is_foreign, txn_count_1h, amount_ratio]])
        return float(self.model.predict_proba(x)[0][1])
