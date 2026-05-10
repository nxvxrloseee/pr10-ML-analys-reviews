"""Training pipeline.

Trains 5 sentiment classifiers (LR, RF, XGBoost, MLP, CatBoost) with TF-IDF.
Runs GridSearchCV (>=3 hyperparams per model). Stores baseline + tuned metrics.

Builds SentenceTransformer embeddings for clustering (KMeans + DBSCAN).
Visualises clusters via PCA. Computes per-cluster top-words and mean sentiment.

Saves artifacts under ./artifacts and produces student_report.png.

Run:  python train.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

import data_gen
import db
from utils import (RU_STOPWORDS, mask_name, mask_phone, tokenize,
                   top_products_by_positive_share, top_words_in,
                   top_words_in_negative)

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)


def _try_imports():
    """Soft-import optional libs. Return dict of available libs."""
    avail = {}
    try:
        import xgboost as xgb  # noqa
        avail["xgboost"] = True
    except Exception as e:
        print(f"[warn] xgboost unavailable: {e}")
        avail["xgboost"] = False
    try:
        from catboost import CatBoostClassifier  # noqa
        avail["catboost"] = True
    except Exception as e:
        print(f"[warn] catboost unavailable: {e}")
        avail["catboost"] = False
    try:
        from sentence_transformers import SentenceTransformer  # noqa
        avail["sbert"] = True
    except Exception as e:
        print(f"[warn] sentence-transformers unavailable: {e}")
        avail["sbert"] = False
    return avail


def load_or_seed_data() -> pd.DataFrame:
    """Generate synthetic data if DB empty, else read from SQLite."""
    db.init_db()
    df = db.load_all()
    if df.empty or len(df) < 200:
        print("[data] seeding synthetic dataset...")
        seed_df = data_gen.generate(n=600, seed=42)
        db.populate_from_df(seed_df, replace=True)
        df = db.load_all()
    print(f"[data] {len(df)} reviews, {df['product'].nunique()} products, "
          f"{df['user_id'].nunique()} users")
    return df


def split_binary(df: pd.DataFrame):
    binary = df[df["sentiment"].notna()].copy()
    binary["sentiment"] = binary["sentiment"].astype(int)
    X = binary["text"].values
    y = binary["sentiment"].values
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def metrics_dict(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None
    return out


def build_estimators(avail: dict):
    """Returns list of (name, estimator, baseline_kw, grid)."""
    items = []

    items.append((
        "LogisticRegression",
        LogisticRegression(max_iter=1000),
        dict(),
        {"C": [0.1, 1.0, 10.0],
         "solver": ["liblinear", "lbfgs"],
         "penalty": ["l2"]},
    ))

    items.append((
        "RandomForest",
        RandomForestClassifier(random_state=42),
        dict(n_estimators=100),
        {"n_estimators": [100, 300],
         "max_depth": [None, 8, 16],
         "min_samples_split": [2, 5]},
    ))

    if avail["xgboost"]:
        from xgboost import XGBClassifier
        items.append((
            "XGBoost",
            XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                          random_state=42, verbosity=0),
            dict(n_estimators=100),
            {"n_estimators": [100, 300],
             "max_depth": [3, 6],
             "learning_rate": [0.05, 0.1]},
        ))

    items.append((
        "MLPClassifier",
        MLPClassifier(random_state=42, max_iter=400),
        dict(hidden_layer_sizes=(64,)),
        {"hidden_layer_sizes": [(32,), (64,), (64, 32)],
         "alpha": [1e-4, 1e-3],
         "learning_rate_init": [1e-3, 5e-3]},
    ))

    if avail["catboost"]:
        from catboost import CatBoostClassifier
        items.append((
            "CatBoost",
            CatBoostClassifier(verbose=0, random_seed=42),
            dict(iterations=200),
            {"iterations": [200, 400],
             "depth": [4, 6],
             "learning_rate": [0.03, 0.1]},
        ))

    return items


def train_classifiers(X_train, X_test, y_train, y_test, vectorizer):
    """Train baseline + GridSearchCV-tuned variants. Save best model + metrics."""
    avail = _try_imports()
    estimators = build_estimators(avail)

    Xtr = vectorizer.transform(X_train)
    Xte = vectorizer.transform(X_test)

    rows = []
    best_models = {}
    roc_data = {}

    for name, est, base_kw, grid in estimators:
        print(f"[model] {name} baseline...")
        base_est = est.__class__(**{**est.get_params(), **base_kw})
        base_est.fit(Xtr, y_train)
        y_pred_b = base_est.predict(Xte)
        try:
            y_proba_b = base_est.predict_proba(Xte)[:, 1]
        except Exception:
            y_proba_b = None
        m_b = metrics_dict(y_test, y_pred_b, y_proba_b)

        print(f"[model] {name} GridSearchCV...")
        gs = GridSearchCV(est, grid, cv=3, scoring="f1", n_jobs=-1)
        gs.fit(Xtr, y_train)
        best = gs.best_estimator_
        y_pred_t = best.predict(Xte)
        try:
            y_proba_t = best.predict_proba(Xte)[:, 1]
        except Exception:
            y_proba_t = None
        m_t = metrics_dict(y_test, y_pred_t, y_proba_t)

        rows.append({
            "model": name,
            "best_params": gs.best_params_,
            **{f"baseline_{k}": v for k, v in m_b.items()},
            **{f"tuned_{k}": v for k, v in m_t.items()},
        })

        best_models[name] = best
        if y_proba_t is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba_t)
            roc_data[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": m_t.get("roc_auc"),
            }

    return rows, best_models, roc_data


def cluster_reviews(df: pd.DataFrame, avail: dict):
    """SentenceTransformer embeddings -> KMeans + DBSCAN. PCA for 2D plot."""
    texts = df["text"].fillna("").tolist()
    if avail["sbert"]:
        from sentence_transformers import SentenceTransformer
        print("[cluster] loading SentenceTransformer...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        emb = model.encode(texts, show_progress_bar=False, batch_size=32)
        emb = np.asarray(emb)
    else:
        print("[cluster] SBERT unavailable - falling back to TF-IDF embeddings")
        v = TfidfVectorizer(max_features=200, stop_words=list(RU_STOPWORDS))
        emb = v.fit_transform(texts).toarray()

    k = max(3, min(6, df["product"].nunique() // 2))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb)
    km_labels = km.labels_

    db_alg = DBSCAN(eps=0.6, min_samples=5, metric="cosine").fit(emb)
    db_labels = db_alg.labels_

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb)

    df_out = df.copy().reset_index(drop=True)
    df_out["kmeans_cluster"] = km_labels
    df_out["dbscan_cluster"] = db_labels
    df_out["pca_x"] = coords[:, 0]
    df_out["pca_y"] = coords[:, 1]

    cluster_summary = []
    for cid in sorted(set(km_labels)):
        sub = df_out[df_out["kmeans_cluster"] == cid]
        words = top_words_in(sub["text"].tolist(), top_n=10)
        sent_known = sub[sub["sentiment"].notna()]
        share_pos = float(sent_known["sentiment"].astype(int).mean()) \
            if not sent_known.empty else None
        cluster_summary.append({
            "cluster": int(cid),
            "size": int(len(sub)),
            "top_words": [w for w, _ in words],
            "positive_share": share_pos,
            "examples": sub["text"].head(3).tolist(),
        })

    return df_out, cluster_summary, k


def make_student_report(df: pd.DataFrame, metrics_rows: list[dict],
                        roc_data: dict, cluster_df: pd.DataFrame,
                        cluster_summary: list[dict]) -> Path:
    """Single PNG with all key visualisations."""
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("ML-анализ отзывов — student_report", fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    df["rating"].plot(kind="hist", bins=5, ax=ax, color="royalblue",
                      edgecolor="black", alpha=0.75)
    ax.set_title("Распределение оценок")
    ax.set_xlabel("rating")

    ax = fig.add_subplot(gs[0, 1])
    daily = df.groupby(df["date"].dt.date).size()
    ax.plot(daily.index, daily.values, "o-", color="teal")
    ax.set_title("Динамика отзывов по дням")
    ax.tick_params(axis="x", rotation=45)

    ax = fig.add_subplot(gs[0, 2])
    pc = df["product"].value_counts()
    ax.barh(pc.index, pc.values, color="seagreen", alpha=0.8)
    ax.set_title("Количество отзывов по товарам")
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[0, 3])
    binary = df[df["sentiment"].notna()]
    if not binary.empty:
        sc = binary["sentiment"].astype(int).value_counts().sort_index()
        ax.pie(sc.values, labels=["Негатив", "Позитив"], autopct="%1.1f%%",
               colors=["coral", "lightgreen"])
    ax.set_title("Соотношение тональности")

    ax = fig.add_subplot(gs[1, 0:2])
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        plot_df = mdf.melt(
            id_vars="model",
            value_vars=["baseline_f1", "tuned_f1"],
            var_name="variant", value_name="f1",
        )
        sns.barplot(data=plot_df, x="model", y="f1", hue="variant", ax=ax)
        ax.set_title("F1-score: baseline vs GridSearchCV-tuned")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15)

    ax = fig.add_subplot(gs[1, 2])
    for name, rd in roc_data.items():
        ax.plot(rd["fpr"], rd["tpr"],
                label=f"{name} (AUC={rd.get('auc') or 0:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_title("ROC-кривые (tuned)")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    all_text = " ".join(df["text"].dropna().tolist()) or "нет данных"
    wc = WordCloud(width=420, height=300, background_color="white",
                   stopwords=RU_STOPWORDS, colormap="viridis").generate(all_text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title("Облако слов: все отзывы")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 0:2])
    palette = sns.color_palette("tab10", n_colors=cluster_df["kmeans_cluster"].nunique())
    for cid, color in zip(sorted(cluster_df["kmeans_cluster"].unique()), palette):
        sub = cluster_df[cluster_df["kmeans_cluster"] == cid]
        ax.scatter(sub["pca_x"], sub["pca_y"], s=18, alpha=0.7,
                   color=color, label=f"K{cid}")
    ax.set_title("Кластеры (KMeans, PCA-2D)")
    ax.legend(fontsize=8, ncol=2)

    ax = fig.add_subplot(gs[2, 2:4])
    top_pos = top_products_by_positive_share(df, top_n=5)
    if not top_pos.empty:
        ax.barh(top_pos["product"], top_pos["positive_share"],
                color="mediumseagreen")
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()
    ax.set_title("Топ товаров по доле позитива")

    out = BASE / "student_report.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] saved -> {out}")
    return out


def forecast_daily(df: pd.DataFrame) -> dict:
    """Simple time-series forecast (kept from lecture baseline)."""
    daily = df.groupby(df["date"].dt.date).size().reset_index(name="reviews_count")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["dow"] = daily["date"].dt.dayofweek
    for lag in (1, 2, 3):
        daily[f"lag_{lag}"] = daily["reviews_count"].shift(lag)
    daily = daily.dropna()
    if len(daily) < 10:
        return {"available": False}
    feats = ["dow", "lag_1", "lag_2", "lag_3"]
    X = daily[feats].values
    y = daily["reviews_count"].values
    Xtr, Xte = X[:-3], X[-3:]
    ytr, yte = y[:-3], y[-3:]
    model = LinearRegression().fit(Xtr, ytr)
    pred = np.maximum(0, model.predict(Xte)).round().astype(int).tolist()
    return {"available": True, "actual": yte.tolist(), "predicted": pred,
            "dates": daily["date"].iloc[-3:].dt.strftime("%Y-%m-%d").tolist()}


def main():
    df = load_or_seed_data()

    print("[pii] masking demo:")
    print(f"  {df['phone'].iloc[0]} -> {mask_phone(df['phone'].iloc[0])}")
    print(f"  {df['user_name'].iloc[0]} -> {mask_name(df['user_name'].iloc[0])}")

    X_train, X_test, y_train, y_test = split_binary(df)
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                 stop_words=list(RU_STOPWORDS))
    vectorizer.fit(np.concatenate([X_train, X_test]))

    metrics_rows, best_models, roc_data = train_classifiers(
        X_train, X_test, y_train, y_test, vectorizer
    )

    primary_name = max(metrics_rows, key=lambda r: r["tuned_f1"])["model"]
    primary_model = best_models[primary_name]
    print(f"[primary] best model = {primary_name}")

    avail = _try_imports()
    cluster_df, cluster_summary, k_used = cluster_reviews(df, avail)
    print(f"[cluster] kmeans k={k_used}, "
          f"dbscan_unique={len(set(cluster_df['dbscan_cluster']))}")

    forecast = forecast_daily(df)
    top_neg_words = top_words_in_negative(df, 5)
    top_pos_products = top_products_by_positive_share(df, 5).to_dict(orient="records")

    joblib.dump(vectorizer, ART / "vectorizer.joblib")
    joblib.dump(primary_model, ART / "primary_model.joblib")
    joblib.dump({n: m for n, m in best_models.items()}, ART / "all_models.joblib")
    cluster_df.to_csv(ART / "clusters.csv", index=False)

    sanitized_metrics = []
    for r in metrics_rows:
        rr = {**r}
        rr["best_params"] = {k: (str(v) if not isinstance(v, (int, float, str, list, dict, type(None))) else v)
                             for k, v in r["best_params"].items()}
        sanitized_metrics.append(rr)

    summary = {
        "primary_model": primary_name,
        "metrics": sanitized_metrics,
        "roc": roc_data,
        "clusters": cluster_summary,
        "forecast": forecast,
        "top_negative_words": top_neg_words,
        "top_positive_products": top_pos_products,
        "n_reviews": int(len(df)),
        "n_products": int(df["product"].nunique()),
        "n_users": int(df["user_id"].nunique()),
    }
    with (ART / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    make_student_report(df, metrics_rows, roc_data, cluster_df, cluster_summary)

    print("[done] artifacts in", ART)


if __name__ == "__main__":
    main()
