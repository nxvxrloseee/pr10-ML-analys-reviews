"""SQLite layer for reviews."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import pandas as pd

DB_PATH = Path(__file__).parent / "reviews.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    sentiment INTEGER,
    date TEXT NOT NULL,
    product TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_name TEXT,
    phone TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(date);
"""


@contextmanager
def connect(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def populate_from_df(df: pd.DataFrame, db_path: Path | str = DB_PATH,
                     replace: bool = True) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        if replace:
            conn.execute("DELETE FROM reviews;")
        df[["text", "rating", "sentiment", "date", "product",
            "user_id", "user_name", "phone"]].to_sql(
            "reviews", conn, if_exists="append", index=False
        )


def insert_review(text: str, rating: int, date: str, product: str,
                  user_id: str, sentiment: int | None = None,
                  user_name: str | None = None, phone: str | None = None,
                  db_path: Path | str = DB_PATH) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO reviews(text, rating, sentiment, date, product, "
            "user_id, user_name, phone) VALUES (?,?,?,?,?,?,?,?)",
            (text, rating, sentiment, date, product, user_id, user_name, phone),
        )
        return int(cur.lastrowid or 0)


def load_all(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM reviews ORDER BY date", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def user_history(user_id: str, db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM reviews WHERE user_id = ? ORDER BY date DESC",
            conn, params=(user_id,))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
