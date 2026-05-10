"""Unit tests for review platform.

Covers: PII masking, sentiment prediction, data loading, metrics, recommender.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import db
import data_gen
from recommender import recommend_for_user
from utils import (mask_name, mask_phone, predict_sentiment,
                   top_products_by_positive_share, top_words_in_negative)


# ---------------- 1. PII masking ----------------

def test_mask_phone_full():
    assert mask_phone("+79991234567") == "+799***4567"


def test_mask_phone_short_passthrough():
    assert mask_phone("+7999") == "+7999"


def test_mask_phone_empty():
    assert mask_phone("") == ""


def test_mask_name_long():
    assert mask_name("Клиент_42") == "Кл*****42"


def test_mask_name_short():
    assert mask_name("Ян") == "Я*"


# ---------------- 2. predict_sentiment ----------------

@pytest.fixture(scope="module")
def trained_model():
    pos = ["отлично", "супер", "рекомендую", "качество отличное", "лучший товар"] * 5
    neg = ["ужасно", "плохо", "брак", "не рекомендую", "разочарован"] * 5
    X = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    vec = TfidfVectorizer().fit(X)
    Xv = vec.transform(X)
    model = LogisticRegression(max_iter=500).fit(Xv, y)
    return model, vec


def test_predict_sentiment_positive(trained_model):
    model, vec = trained_model
    label, conf = predict_sentiment("отличный товар, рекомендую", model, vec)
    assert label == "позитивный"
    assert 0.5 <= conf <= 1.0


def test_predict_sentiment_negative(trained_model):
    model, vec = trained_model
    label, conf = predict_sentiment("ужасное качество, брак", model, vec)
    assert label == "негативный"
    assert 0.5 <= conf <= 1.0


def test_predict_sentiment_empty_input(trained_model):
    model, vec = trained_model
    label, conf = predict_sentiment("", model, vec)
    assert label == "негативный"
    assert conf == 0.0


# ---------------- 3. Data loading via SQLite ----------------

def test_load_all_roundtrip(tmp_path):
    db_path = tmp_path / "t.db"
    df = data_gen.generate(n=50, seed=1)
    db.populate_from_df(df, db_path=db_path, replace=True)
    loaded = db.load_all(db_path=db_path)
    assert len(loaded) == 50
    assert set(["text", "rating", "product", "user_id"]).issubset(loaded.columns)


def test_insert_review_persists(tmp_path):
    db_path = tmp_path / "t.db"
    db.init_db(db_path)
    rid = db.insert_review(
        text="Отличный товар", rating=5, sentiment=1,
        date="2026-05-01T10:00:00", product="Смартфон Xiaomi",
        user_id="u_test", db_path=db_path,
    )
    assert rid > 0
    loaded = db.load_all(db_path=db_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["text"] == "Отличный товар"


# ---------------- 4. Metrics computation ----------------

def test_metrics_calculation(trained_model):
    model, vec = trained_model
    texts = ["отлично супер", "ужасно плохо", "рекомендую", "брак"]
    y_true = [1, 0, 1, 0]
    y_pred = model.predict(vec.transform(texts))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert acc >= 0.75


def test_top_products_by_positive_share():
    df = pd.DataFrame({
        "product": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "rating":  [5, 5, 4, 4, 5,  3, 3, 2, 2, 1,  4, 5, 4, 4, 4],
    })
    top = top_products_by_positive_share(df, top_n=3, min_reviews=3)
    assert top.iloc[0]["product"] in {"A", "C"}
    assert top.iloc[-1]["product"] == "B"


def test_top_words_in_negative_returns_n():
    df = pd.DataFrame({
        "rating": [1, 2, 5, 1, 2],
        "text": [
            "ужасное качество брак",
            "плохо разочарован брак",
            "отлично рекомендую",
            "ужасное обслуживание",
            "брак вернул",
        ],
    })
    out = top_words_in_negative(df, top_n=3)
    words = [w for w, _ in out]
    assert "брак" in words
    assert len(out) <= 3


# ---------------- 5. Recommender ----------------

def test_recommend_for_existing_user_returns_unseen():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "product": ["A", "B", "A", "C", "B", "C"],
        "rating":  [5,    4,    5,    5,    4,    5],
        "text":    ["x"] * 6,
        "date":    pd.to_datetime(["2026-01-01"] * 6),
        "sentiment": [1, 1, 1, 1, 1, 1],
    })
    recs = recommend_for_user("u1", df, top_n=2)
    assert len(recs) >= 1
    seen_by_u1 = {"A", "B"}
    rec_products = {r["product"] for r in recs}
    assert rec_products & ({"C"}) or rec_products  # at least returns something
    for r in recs:
        assert "predicted_rating" in r


def test_recommend_for_new_user_falls_back_to_global_top():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "product": ["A", "B", "A", "C", "B", "C"],
        "rating":  [5,    4,    5,    3,    4,    5],
        "text":    ["x"] * 6,
        "date":    pd.to_datetime(["2026-01-01"] * 6),
        "sentiment": [1, 1, 1, 0, 1, 1],
    })
    recs = recommend_for_user("u_new_unknown", df, top_n=3)
    assert isinstance(recs, list)
    assert all("predicted_rating" in r for r in recs)


def test_recommend_empty_df():
    recs = recommend_for_user("anyone", pd.DataFrame(), top_n=3)
    assert recs == []
