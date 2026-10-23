"""SQLite storage layer: accounts and transactions."""
import sqlite3

DB_PATH = "transactions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    owner      TEXT NOT NULL,
    balance    REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    sender      TEXT NOT NULL,
    receiver    TEXT NOT NULL,
    amount      REAL NOT NULL,
    country     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    fraud_score REAL,
    status      TEXT NOT NULL,   -- APPROVED / REVIEW / BLOCKED / REJECTED
    reason      TEXT
);
"""


def get_connection(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def create_account(conn, account_id, owner, balance=0.0):
    conn.execute(
        "INSERT OR REPLACE INTO accounts (account_id, owner, balance) VALUES (?, ?, ?)",
        (account_id, owner, balance),
    )
    conn.commit()


def get_account(conn, account_id):
    return conn.execute(
        "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
    ).fetchone()


def update_balance(conn, account_id, delta):
    conn.execute(
        "UPDATE accounts SET balance = balance + ? WHERE account_id = ?",
        (delta, account_id),
    )


def save_transaction(conn, sender, receiver, amount, country, timestamp,
                     fraud_score, status, reason):
    cur = conn.execute(
        """INSERT INTO transactions
           (sender, receiver, amount, country, timestamp, fraud_score, status, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (sender, receiver, amount, country, timestamp, fraud_score, status, reason),
    )
    conn.commit()
    return cur.lastrowid


def sender_history(conn, sender):
    """Past APPROVED transactions for a sender (oldest first)."""
    return conn.execute(
        """SELECT amount, timestamp FROM transactions
           WHERE sender = ? AND status = 'APPROVED' ORDER BY timestamp""",
        (sender,),
    ).fetchall()


def all_transactions(conn):
    return conn.execute("SELECT * FROM transactions ORDER BY txn_id").fetchall()
