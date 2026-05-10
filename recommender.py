"""User-based collaborative filtering for product recommendations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mean rating per (user, product). Missing -> NaN."""
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="user_id", columns="product",
                          values="rating", aggfunc="mean")


def recommend_for_user(user_id: str, df: pd.DataFrame,
                       top_n: int = 5, k_neighbors: int = 5) -> list[dict]:
    """User-based CF: find k similar users, predict ratings for unseen products.

    Falls back to global top-rated products when user is new or has no peers.
    """
    if df.empty:
        return []
    matrix = build_user_item_matrix(df)
    if matrix.empty:
        return []

    if user_id not in matrix.index:
        return _global_top(df, top_n)

    user_vec = matrix.loc[user_id]
    seen = set(user_vec.dropna().index)
    candidates = [p for p in matrix.columns if p not in seen]
    if not candidates:
        candidates = list(matrix.columns)

    filled = matrix.fillna(matrix.mean(axis=0)).fillna(0)
    sims = cosine_similarity(filled.values)
    sim_df = pd.DataFrame(sims, index=matrix.index, columns=matrix.index)
    peer_sims = sim_df.loc[user_id].drop(user_id).sort_values(ascending=False)
    peers = peer_sims.head(k_neighbors)
    if peers.empty or peers.sum() == 0:
        return _global_top(df, top_n)

    peer_ratings = matrix.loc[peers.index]
    weights = peers.values.reshape(-1, 1)
    preds = []
    for product in candidates:
        col = peer_ratings[product]
        mask = col.notna()
        if not mask.any():
            continue
        w = weights[mask.values]
        r = col[mask].values.reshape(-1, 1)
        score = float((w * r).sum() / w.sum()) if w.sum() else float(r.mean())
        preds.append({"product": product, "predicted_rating": round(score, 2)})

    preds.sort(key=lambda x: x["predicted_rating"], reverse=True)
    return preds[:top_n]


def _global_top(df: pd.DataFrame, top_n: int) -> list[dict]:
    g = df.groupby("product")["rating"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 3].sort_values("mean", ascending=False)
    return [{"product": r["product"], "predicted_rating": round(float(r["mean"]), 2)}
            for _, r in g.head(top_n).iterrows()]
