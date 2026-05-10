import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PRODUCTS = [
    "Смартфон Xiaomi", "Ноутбук Lenovo", "Наушники Sony", "Умные часы Apple",
    "Планшет Samsung", "Электрочайник Bosch", "Робот-пылесос Xiaomi",
    "Кофемашина DeLonghi", "Фитнес-браслет Honor", "Колонка JBL",
]

POSITIVE_TEMPLATES = [
    "Отличный {p}, очень доволен покупкой!",
    "{p} превзошёл ожидания, рекомендую всем.",
    "Замечательное качество, {p} работает безупречно.",
    "Супер {p}! За эти деньги — лучшее, что есть.",
    "Купил {p} жене, она в восторге.",
    "Великолепный {p}, упаковка целая, доставка быстрая.",
    "Прекрасная вещь, {p} оправдал каждый рубль.",
    "Идеальное соотношение цены и качества, {p} топ.",
    "Пользуюсь {p} месяц, нареканий ноль.",
    "Лучший {p} из всех, что были у меня.",
]

NEGATIVE_TEMPLATES = [
    "Ужасное качество, {p} сломался через неделю.",
    "Полное разочарование, {p} не работает как описано.",
    "Брак с завода, {p} вернул обратно.",
    "Не рекомендую {p}, деньги на ветер.",
    "Кошмар, {p} пришёл с дефектом.",
    "Очень плохо, {p} перегревается и тормозит.",
    "Отвратительная сборка, {p} развалился сразу.",
    "Хуже {p} не видел, обманули с характеристиками.",
    "Не покупайте {p}, ужас.",
    "Зря потратил деньги на {p}, верну по гарантии.",
]

NEUTRAL_TEMPLATES = [
    "Нормальный {p}, ничего особенного.",
    "Средненько, {p} работает, но есть нюансы.",
    "{p} как {p}, ожидал большего.",
    "Так себе покупка, {p} на троечку.",
    "Обычный {p}, без восторга и без жалоб.",
]


def fill(template: str, product: str) -> str:
    return template.format(p=product.split()[0].lower())


def generate(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2026, 2, 1)
    for i in range(n):
        rating = int(rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.10, 0.18, 0.30, 0.34]))
        product = rng.choice(PRODUCTS)
        if rating >= 4:
            text = fill(rng.choice(POSITIVE_TEMPLATES), product)
            sentiment = 1
        elif rating <= 2:
            text = fill(rng.choice(NEGATIVE_TEMPLATES), product)
            sentiment = 0
        else:
            text = fill(rng.choice(NEUTRAL_TEMPLATES), product)
            sentiment = None
        date = start + timedelta(days=int(rng.integers(0, 90)),
                                 hours=int(rng.integers(0, 24)))
        uid = int(rng.integers(1, 41))
        rows.append({
            "text": text,
            "rating": rating,
            "sentiment": sentiment,
            "date": date.isoformat(timespec="seconds"),
            "product": product,
            "user_id": f"user_{uid:03d}",
            "user_name": f"Клиент_{uid}",
            "phone": f"+7{rng.integers(900, 999)}{rng.integers(1000000, 9999999)}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"Total: {len(df)}, products: {df['product'].nunique()}, "
          f"users: {df['user_id'].nunique()}")
