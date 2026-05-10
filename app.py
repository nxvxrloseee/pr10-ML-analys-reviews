"""Flask web app for review analytics platform."""
from __future__ import annotations

import json
import logging
import os
import uuid
from base64 import b64encode
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from flask import (Flask, abort, g, jsonify, make_response, redirect,
                   render_template, request, session, url_for)
from wordcloud import WordCloud

import db
from recommender import recommend_for_user
from utils import (RU_STOPWORDS, mask_name, mask_phone, predict_sentiment,
                   top_products_by_positive_share, top_words_in_negative)

BASE = Path(__file__).parent
ART = BASE / "artifacts"
LOG_PATH = BASE / "app.log"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

logger = logging.getLogger("reviews_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


def log_event(action: str, result: str = "ok", **extra):
    user_id = getattr(g, "user_id", "-")
    parts = [f"user_id={user_id}", f"action={action}", f"result={result}"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))


app = Flask(__name__, template_folder=str(BASE / "templates"),
            static_folder=str(BASE / "static"))
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")


def _load_artifact(name: str, default=None):
    path = ART / name
    if not path.exists():
        return default
    if path.suffix == ".joblib":
        return joblib.load(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return None


def get_artifacts():
    if "artifacts" not in app.config:
        app.config["artifacts"] = {
            "vectorizer": _load_artifact("vectorizer.joblib"),
            "primary_model": _load_artifact("primary_model.joblib"),
            "all_models": _load_artifact("all_models.joblib", {}) or {},
            "summary": _load_artifact("summary.json", {}) or {},
            "clusters_df": _load_artifact("clusters.csv"),
        }
    return app.config["artifacts"]


@app.before_request
def assign_user_id():
    uid = request.cookies.get("uid")
    if not uid:
        uid = "u_" + uuid.uuid4().hex[:10]
        g.new_user = True
    else:
        g.new_user = False
    g.user_id = uid


@app.after_request
def persist_user_id(response):
    if getattr(g, "new_user", False):
        response.set_cookie("uid", g.user_id, max_age=60 * 60 * 24 * 365)
    return response


@app.context_processor
def inject_globals():
    return {"current_user": g.get("user_id", "-")}


@app.route("/")
def home():
    art = get_artifacts()
    df = db.load_all()
    products = sorted(df["product"].unique().tolist()) if not df.empty else []
    recs = recommend_for_user(g.user_id, df, top_n=5) if not df.empty else []
    log_event("home", "ok", n_recs=len(recs))
    return render_template("home.html", products=products, recommendations=recs,
                           summary=art["summary"])


@app.post("/review")
def submit_review():
    text = (request.form.get("text") or "").strip()
    rating = request.form.get("rating", type=int)
    product = (request.form.get("product") or "").strip()
    if not text or not product or rating is None or not (1 <= rating <= 5):
        log_event("submit_review", "error_validation")
        return render_template("home.html",
                               products=sorted(db.load_all()["product"].unique()),
                               recommendations=[],
                               error="Заполните все поля. Оценка от 1 до 5."), 400

    art = get_artifacts()
    label, conf = ("негативный", 0.5)
    sentiment = None
    if art["primary_model"] is not None and art["vectorizer"] is not None:
        label, conf = predict_sentiment(text, art["primary_model"], art["vectorizer"])
        sentiment = 1 if label == "позитивный" else 0

    db.insert_review(
        text=text, rating=rating, sentiment=sentiment,
        date=datetime.now().isoformat(timespec="seconds"),
        product=product, user_id=g.user_id,
        user_name=f"User_{g.user_id[-4:]}",
    )
    log_event("submit_review", "ok", product=product, rating=rating,
              sentiment=label, conf=f"{conf:.2f}")
    return render_template("review_result.html", text=text, rating=rating,
                           product=product, label=label, confidence=conf)


@app.get("/predict")
def predict_endpoint():
    """Lightweight JSON sentiment endpoint."""
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "missing 'text'"}), 400
    art = get_artifacts()
    if art["primary_model"] is None or art["vectorizer"] is None:
        return jsonify({"error": "model not trained, run train.py"}), 503
    label, conf = predict_sentiment(text, art["primary_model"], art["vectorizer"])
    log_event("predict", "ok", label=label, conf=f"{conf:.2f}")
    return jsonify({"label": label, "confidence": round(conf, 4)})


@app.get("/stats")
def stats():
    df = db.load_all()
    if df.empty:
        return render_template("stats.html", charts={}, top_neg=[],
                               n=0, summary={})
    rating_dist = _fig_to_b64(_plot_rating_dist(df))
    daily_dyn = _fig_to_b64(_plot_daily(df))
    wc_img = _fig_to_b64(_plot_wordcloud(df))
    top_neg = top_words_in_negative(df, 10)
    log_event("stats", "ok", n=len(df))
    return render_template("stats.html",
                           charts={"rating": rating_dist, "daily": daily_dyn,
                                   "wordcloud": wc_img},
                           top_neg=top_neg, n=len(df),
                           summary=get_artifacts()["summary"])


@app.get("/products")
def products_view():
    df = db.load_all()
    if df.empty:
        return render_template("products.html", rows=[])
    g_df = df.groupby("product").agg(
        n_reviews=("rating", "size"),
        avg_rating=("rating", "mean"),
        positive_share=("rating", lambda r: float((r >= 4).mean())),
    ).reset_index().sort_values("avg_rating", ascending=False)
    g_df["avg_rating"] = g_df["avg_rating"].round(2)
    g_df["positive_share"] = (g_df["positive_share"] * 100).round(1)
    log_event("products", "ok", n=len(g_df))
    return render_template("products.html", rows=g_df.to_dict(orient="records"))


@app.get("/models")
def models_view():
    summary = get_artifacts()["summary"]
    metrics = summary.get("metrics", [])
    roc = summary.get("roc", {})
    roc_img = _fig_to_b64(_plot_roc(roc)) if roc else None
    log_event("models", "ok", n_models=len(metrics))
    return render_template("models.html", metrics=metrics, roc_img=roc_img,
                           primary=summary.get("primary_model"))


@app.get("/clusters")
def clusters_view():
    art = get_artifacts()
    cluster_df = art["clusters_df"]
    summary = art["summary"]
    if cluster_df is None or cluster_df.empty:
        return render_template("clusters.html", plotly_html=None,
                               clusters=summary.get("clusters", []))
    cdf = cluster_df.copy()
    cdf["kmeans_cluster"] = cdf["kmeans_cluster"].astype(str)
    cdf["text_short"] = cdf["text"].str.slice(0, 80)
    fig = px.scatter(cdf, x="pca_x", y="pca_y", color="kmeans_cluster",
                     hover_data=["product", "rating", "text_short"],
                     title="Кластеры отзывов (KMeans, PCA-2D)")
    fig.update_layout(height=560)
    plot_html = plotly.io.to_html(fig, include_plotlyjs="cdn", full_html=False)
    log_event("clusters", "ok")
    return render_template("clusters.html", plotly_html=plot_html,
                           clusters=summary.get("clusters", []))


@app.get("/clusters/<int:cid>")
def cluster_examples(cid: int):
    art = get_artifacts()
    cdf = art["clusters_df"]
    if cdf is None:
        return jsonify({"error": "no clusters"}), 404
    sub = cdf[cdf["kmeans_cluster"] == cid].head(20)
    return jsonify(sub[["text", "rating", "product"]].to_dict(orient="records"))


@app.get("/admin/login")
def admin_login_form():
    return render_template("admin_login.html", error=None)


@app.post("/admin/login")
def admin_login_submit():
    pwd = request.form.get("password", "")
    if pwd == ADMIN_PASSWORD:
        session["admin"] = True
        log_event("admin_login", "ok")
        return redirect(url_for("admin_logs"))
    log_event("admin_login", "fail")
    return render_template("admin_login.html", error="Неверный пароль"), 401


@app.get("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("home"))


@app.get("/admin/logs")
def admin_logs():
    if not session.get("admin"):
        return redirect(url_for("admin_login_form"))
    lines: list[str] = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    last50 = [ln.rstrip("\n") for ln in lines[-50:]]
    return render_template("admin_logs.html", lines=last50)


def _fig_to_b64(fig) -> str:
    if fig is None:
        return ""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return b64encode(buf.getvalue()).decode("ascii")


def _plot_rating_dist(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    df["rating"].plot(kind="hist", bins=5, ax=ax, color="royalblue",
                      edgecolor="black", alpha=0.8)
    ax.set_title("Распределение оценок")
    ax.set_xlabel("rating")
    return fig


def _plot_daily(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    daily = df.groupby(df["date"].dt.date).size()
    ax.plot(daily.index, daily.values, "o-", color="teal")
    ax.set_title("Динамика отзывов по дням")
    fig.autofmt_xdate()
    return fig


def _plot_wordcloud(df: pd.DataFrame):
    text = " ".join(df["text"].tolist()) or "нет данных"
    wc = WordCloud(width=720, height=380, background_color="white",
                   stopwords=RU_STOPWORDS, colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Облако слов")
    return fig


def _plot_roc(roc: dict):
    if not roc:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, rd in roc.items():
        auc = rd.get("auc")
        ax.plot(rd["fpr"], rd["tpr"],
                label=f"{name} (AUC={auc:.2f})" if auc else name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC-кривые (tuned)")
    ax.legend(fontsize=8)
    return fig


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404,
                           message="Страница не найдена"), 404


@app.errorhandler(500)
def server_error(e):
    log_event("server_error", "error", err=str(e)[:120])
    return render_template("error.html", code=500,
                           message="Внутренняя ошибка"), 500


if __name__ == "__main__":
    db.init_db()
    if not (ART / "primary_model.joblib").exists():
        print("[startup] artifacts missing - running train.py first")
        import train
        train.main()
    app.run(host="127.0.0.1", port=5000, debug=False)
