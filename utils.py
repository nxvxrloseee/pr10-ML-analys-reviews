"""PII masking, sentiment helpers, aggregations."""
from __future__ import annotations
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

_FALLBACK_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между", "это", "эта",
}


def _load_stopwords() -> set[str]:
    """Load Russian stopwords from NLTK; auto-download corpus on first run.

    Falls back to a curated literal set if NLTK or its data is unavailable
    (offline environment, etc.) so the app still works without network access.
    """
    try:
        from nltk.corpus import stopwords
        try:
            words = stopwords.words("russian")
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            words = stopwords.words("russian")
        return set(words) | _FALLBACK_STOPWORDS
    except Exception:
        return set(_FALLBACK_STOPWORDS)


RU_STOPWORDS: set[str] = _load_stopwords()

WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def mask_phone(phone: str) -> str:
    """+79991234567 -> +799***4567."""
    if not phone:
        return phone
    digits_only = re.sub(r"\D", "", phone)
    if len(digits_only) >= 10:
        return phone[:4] + "***" + phone[-4:]
    return phone


def mask_name(name: str) -> str:
    """Клиент_42 -> Кл***42."""
    if not name:
        return name
    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)
    return name[:2] + "*" * (len(name) - 4) + name[-2:]


def predict_sentiment(text: str, model, vectorizer) -> tuple[str, float]:
    """Returns ('позитивный'|'негативный', confidence_pct as float in [0,1])."""
    if not text or not text.strip():
        return ("негативный", 0.0)
    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    pred = int(np.argmax(proba))
    label = "позитивный" if pred == 1 else "негативный"
    return label, float(proba[pred])


def top_products_by_positive_share(df: pd.DataFrame, top_n: int = 5,
                                   min_reviews: int = 3) -> pd.DataFrame:
    """Top products by share of positive reviews (rating >= 4)."""
    if df.empty:
        return pd.DataFrame(columns=["product", "positive_share", "n_reviews"])
    g = df.groupby("product").agg(
        n_reviews=("rating", "size"),
        positive_share=("rating", lambda r: float((r >= 4).mean())),
        avg_rating=("rating", "mean"),
    ).reset_index()
    g = g[g["n_reviews"] >= min_reviews]
    return g.sort_values("positive_share", ascending=False).head(top_n).reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")
            if t.lower() not in RU_STOPWORDS and len(t) > 2]


def top_words_in_negative(df: pd.DataFrame, top_n: int = 5) -> list[tuple[str, int]]:
    """5 most common words across negative reviews (rating <= 2)."""
    neg = df[df["rating"] <= 2]["text"].dropna().tolist()
    counter: Counter[str] = Counter()
    for t in neg:
        counter.update(tokenize(t))
    return counter.most_common(top_n)


def top_words_in(texts: Iterable[str], top_n: int = 10) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize(t))
    return counter.most_common(top_n)
