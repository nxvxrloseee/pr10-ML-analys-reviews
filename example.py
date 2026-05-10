import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


print("\nЗагружаем и подготавливаем данные...")
positive_reviews = ["отлично", "замечательно", "супер", "рекомендую", "лучший",
                    "качество отличное", "доволен", "прекрасно", "великолепно", "нравится"]
negative_reviews = ["ужасно", "плохо", "разочарован", "не рекомендую", "брак",
                    "качество плохое", "ужасное качество", "не работает", "деньги на ветер", "кошмар"]
products = ["Смартфон", "Ноутбук", "Наушники", "Часы", "Планшет"]

data = []
for i in range(200):
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])

    if rating >= 4:
        text = np.random.choice(positive_reviews)
        sentiment = 1
    elif rating <= 2:
        text = np.random.choice(negative_reviews)
        sentiment = 0
    else:
        text = "нормальный товар"
        sentiment = None

    data.append({
        'id': i,
        'text': text,
        'rating': rating,
        'sentiment': sentiment,
        'date': datetime.now() - timedelta(days=np.random.randint(0, 60)),
        'product': np.random.choice(products),
        'user_id': np.random.randint(1, 31),
        'user_name': f"Клиент_{np.random.randint(1, 31)}",
        'phone': f"+7{np.random.randint(900, 999)}{np.random.randint(1000000, 9999999)}"
    })

df = pd.DataFrame(data)
print(f" Создано {len(df)} записей")


print("КЛАССИФИКАЦИЯ ТЕКСТОВ")

df_binary = df[df['sentiment'].notna()].copy()
X = df_binary['text'].values
y = df_binary['sentiment'].values

print(f"Обучающих примеров: {len(X)} (позитив: {sum(y)}, негатив: {len(y)-sum(y)})")

vectorizer = TfidfVectorizer(max_features=100)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nМЕТРИКИ:")
print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
print(f"   Precision: {precision_score(y_test, y_pred):.2%}")
print(f"   Recall:    {recall_score(y_test, y_pred):.2%}")
print(f"   F1-score:  {f1_score(y_test, y_pred):.2%}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nМатрица ошибок:")
print(f"   Негатив → предсказано негатив: {cm[0,0]}, позитив: {cm[0,1]}")
print(f"   Позитив → предсказано негатив: {cm[1,0]}, позитив: {cm[1,1]}")

print(f"\nПРОВЕРКА НА НОВЫХ ПРИМЕРАХ:")
test_phrases = [
    "отличный товар, всем рекомендую",
    "ужасное качество, разочарован",
    "нормально, но ничего особенного"
]

for phrase in test_phrases:
    X_new = vectorizer.transform([phrase])
    pred = model.predict(X_new)[0]
    pred_int = int(pred)
    proba = model.predict_proba(X_new)[0][pred_int]
    result = "ПОЗИТИВ" if pred_int == 1 else "НЕГАТИВ"
    print(f"   \"{phrase}\" → {result} (уверенность: {proba:.1%})")


print("ЗАЩИТА ПЕРСОНАЛЬНЫХ ДАННЫХ")
def mask_phone(phone):
    if len(phone) >= 10:
        return phone[:4] + "***" + phone[-4:]
    return phone

def mask_name(name):
    if len(name) <= 3:
        return name[0] + "*" * (len(name) - 1)
    return name[:2] + "*" * (len(name) - 4) + name[-2:]

print(f"Исходный телефон: {df['phone'].iloc[0]} → {mask_phone(df['phone'].iloc[0])}")
print(f"Исходное имя: {df['user_name'].iloc[0]} → {mask_name(df['user_name'].iloc[0])}")

df_protected = df.copy()
df_protected['user_name'] = df_protected['user_name'].apply(mask_name)
df_protected['phone'] = df_protected['phone'].apply(mask_phone)
print(f"Создана защищённая версия ({len(df_protected)} записей)")


print("ВРЕМЕННЫЕ РЯДЫ")
daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='reviews_count')
daily_counts['date'] = pd.to_datetime(daily_counts['date'])
daily_counts = daily_counts.sort_values('date')

daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
for lag in [1, 2, 3]:
    daily_counts[f'lag_{lag}'] = daily_counts['reviews_count'].shift(lag)

daily_counts = daily_counts.dropna()

if len(daily_counts) >= 10:
    feature_cols = ['day_of_week', 'lag_1', 'lag_2', 'lag_3']
    X_train_ts = daily_counts[feature_cols].values[:-3]
    y_train_ts = daily_counts['reviews_count'].values[:-3]
    X_test_ts = daily_counts[feature_cols].values[-3:]
    y_test_ts = daily_counts['reviews_count'].values[-3:]

    ts_model = LinearRegression()
    ts_model.fit(X_train_ts, y_train_ts)
    predictions = ts_model.predict(X_test_ts)

    print(f"\nПРОГНОЗ НА 3 ДНЯ:")
    for i, (actual, pred) in enumerate(zip(y_test_ts, predictions)):
        print(f"   День {i+1}: Факт = {int(actual):2d} | Прогноз = {int(max(0, pred)):2d}")
else:
    print("Недостаточно данных")

print("РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
product_texts = df.groupby('product').agg({'text': lambda x: ' '.join(x)}).reset_index()
vec = TfidfVectorizer(max_features=50)
vectors = vec.fit_transform(product_texts['text'])
similarity_matrix = cosine_similarity(vectors)

def recommend(product_name, top_n=3):
    idx = product_texts[product_texts['product'] == product_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    return [product_texts.iloc[i]['product'] for i, _ in scores]

for p in products[:2]:
    print(f"\nРекомендации для '{p}':")
    for rec in recommend(p):
        print(f"   → {rec}")

print("ВИЗУАЛИЗАЦИЯ")
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
fig.suptitle('Анализ данных', fontsize=14)

axes[0, 0].hist(df['rating'], bins=5, edgecolor='black', color='royalblue', alpha=0.7)
axes[0, 0].set_title('Распределение оценок')

if len(daily_counts) > 0:
    axes[0, 1].plot(daily_counts['date'], daily_counts['reviews_count'], 'b-o', linewidth=2)
    axes[0, 1].set_title('Динамика отзывов')
    axes[0, 1].tick_params(axis='x', rotation=45)

if len(daily_counts) >= 10:
    axes[0, 2].bar(range(3), y_test_ts, alpha=0.6, label='Факт', color='blue')
    axes[0, 2].bar(range(3), predictions, alpha=0.6, label='Прогноз', color='orange')
    axes[0, 2].set_title('Прогноз')
    axes[0, 2].legend()

im = axes[1, 0].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, cm[i, j], ha='center', va='center')
axes[1, 0].set_title('Матрица ошибок')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Негатив', 'Позитив'])
axes[1, 0].set_yticklabels(['Негатив', 'Позитив'])

product_counts = df['product'].value_counts()
axes[1, 1].bar(product_counts.index, product_counts.values, color='seagreen', alpha=0.7)
axes[1, 1].set_title('Популярность товаров')
axes[1, 1].tick_params(axis='x', rotation=45)

sentiment_counts = df_binary['sentiment'].value_counts()
axes[1, 2].pie(sentiment_counts.values, labels=['Негатив', 'Позитив'],
               colors=['coral', 'lightgreen'], autopct='%1.1f%%')
axes[1, 2].set_title('Соотношение')

plt.tight_layout()
plt.savefig('lecture_ml_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
