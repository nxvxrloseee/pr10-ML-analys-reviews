# RUN.md — ReviewLab

Гайд по запуску ML-платформы анализа отзывов.

## 0. Требования

- Python 3.10+ (проект протестирован на 3.13)
- Windows / Linux / macOS
- ~1 GB свободного места (SBERT-модель скачается в кеш HuggingFace при первом запуске)
- Интернет на первом запуске (для NLTK stopwords + SBERT)

## 1. Установка зависимостей

```powershell
cd C:\Users\Qwerty\PycharmProjects\Chekan\pr10
pip install -r requirements.txt
```

Список ключевых пакетов:

| Пакет | Назначение |
|-------|-----------|
| `flask` | веб-приложение |
| `pandas`, `numpy`, `scikit-learn` | основа ML |
| `xgboost`, `catboost` | модели градиентного бустинга |
| `sentence-transformers` | мультиязычные эмбеддинги для кластеризации |
| `wordcloud` | облако слов |
| `plotly` | интерактивный скаттер кластеров |
| `nltk` | русские стоп-слова |
| `pytest` | тесты |
| `joblib` | сериализация моделей |

Один раз скачать корпус NLTK (`train.py` сделает это автоматически, но можно заранее):

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

## 2. Обучение моделей и генерация отчёта

```powershell
python train.py
```

Что происходит:

1. Если БД пустая, генерируется **600 синтетических отзывов** (10 товаров, 40 пользователей) и пишется в `reviews.db`.
2. Обучаются **5 классификаторов** тональности:
   - LogisticRegression
   - RandomForest
   - XGBoost
   - MLPClassifier (нейросеть)
   - CatBoost
3. Для каждой модели — **baseline + GridSearchCV** по 3+ параметрам, сохраняются метрики (Accuracy, Precision, Recall, F1, ROC-AUC).
4. **SentenceTransformer** (`paraphrase-multilingual-MiniLM-L12-v2`) считает эмбеддинги; **KMeans + DBSCAN** кластеризуют отзывы; **PCA** даёт 2D-координаты.
5. Считается **прогноз** дневного количества отзывов на 3 дня (LinearRegression на лагах).
6. Создаются артефакты в `artifacts/`:
   - `vectorizer.joblib` — TF-IDF
   - `primary_model.joblib` — лучшая по F1 модель
   - `all_models.joblib` — словарь со всеми пятью
   - `summary.json` — метрики, ROC, кластеры, прогноз
   - `clusters.csv` — отзывы с метками KMeans/DBSCAN и PCA-координатами
7. Сохраняется **`student_report.png`** — единый отчёт со всеми ключевыми графиками.

Время выполнения: ~2-5 мин. Первый запуск дольше — SBERT качает ~500 MB модель в `~/.cache/huggingface/`.

## 3. Запуск тестов

```powershell
python -m pytest tests/ -v
```

Должно быть `16 passed`. Тесты покрывают:

- маскировку телефона (`mask_phone`) и имени (`mask_name`)
- предсказание тональности (`predict_sentiment`)
- запись/чтение отзывов в SQLite (`db.insert_review`, `db.load_all`)
- расчёт метрик
- топ товаров по доле позитива и топ слов в негативе
- рекомендательную систему (существующий пользователь, новый, пустой датасет)

## 4. Запуск веб-приложения

```powershell
python app.py
```

Откройте в браузере: <http://127.0.0.1:5000>

Если артефакты ещё не сгенерированы, `app.py` запустит `train.py` автоматически.

### Маршруты

| URL | Описание |
|-----|----------|
| `/` | главная: форма ввода отзыва, показ предсказанной тональности, персональные рекомендации |
| `POST /review` | обработка формы, сохранение в SQLite, классификация |
| `GET /predict?text=...` | JSON-эндпоинт тональности |
| `/stats` | распределение оценок, динамика по дням, облако слов, топ слов в негативе, прогноз |
| `/products` | таблица товаров: средний рейтинг, количество отзывов, доля позитива |
| `/models` | сравнение моделей: baseline vs tuned, ROC-кривые, лучшие гиперпараметры |
| `/clusters` | интерактивный Plotly-скаттер кластеров, топ-10 слов и доля позитива по каждому кластеру |
| `/clusters/<id>` | JSON: до 20 отзывов из кластера |
| `/admin/login` | вход для администратора |
| `/admin/logs` | последние 50 строк `app.log` (требуется авторизация) |
| `/admin/logout` | выход |

### Идентификация пользователя

При первом заходе генерируется UUID-cookie `uid` (срок 1 год). По этому ID пишутся отзывы и считаются персональные рекомендации (user-based collaborative filtering).

### Админка

Пароль по умолчанию: `admin123`. Сменить через переменную окружения:

```powershell
$env:ADMIN_PASSWORD="my_secret"
python app.py
```

## 5. Логирование

Все действия пишутся в `app.log` (создаётся при первом запросе). Формат строки:

```
2026-05-08 12:34:56 | INFO | user_id=u_a1b2c3d4ef | action=submit_review | result=ok | product=Смартфон Xiaomi | rating=5 | sentiment=позитивный | conf=0.97
```

События: `home`, `submit_review`, `predict`, `stats`, `products`, `models`, `clusters`, `admin_login`, `server_error`.

## 6. Структура проекта

```
pr10/
├── app.py                  # Flask-приложение
├── train.py                # обучение, кластеризация, отчёт
├── data_gen.py             # генератор синтетических отзывов
├── db.py                   # слой SQLite
├── recommender.py          # user-based collaborative filtering
├── utils.py                # маски PII, predict_sentiment, агрегации, NLTK stopwords
├── requirements.txt
├── reviews.db              # SQLite (создаётся train.py)
├── app.log                 # лог запросов (создаётся app.py)
├── student_report.png      # сводный отчёт (создаётся train.py)
├── artifacts/              # модели и метрики
│   ├── vectorizer.joblib
│   ├── primary_model.joblib
│   ├── all_models.joblib
│   ├── summary.json
│   └── clusters.csv
├── templates/              # Jinja2 шаблоны
│   ├── base.html
│   ├── home.html
│   ├── review_result.html
│   ├── stats.html
│   ├── products.html
│   ├── models.html
│   ├── clusters.html
│   ├── admin_login.html
│   ├── admin_logs.html
│   └── error.html
├── static/
│   └── style.css
└── tests/
    ├── conftest.py
    └── test_app.py
```

## 7. Сброс и повторный запуск

Полностью пересобрать данные и модели:

```powershell
Remove-Item reviews.db, app.log -ErrorAction SilentlyContinue
Remove-Item artifacts\*.joblib, artifacts\summary.json, artifacts\clusters.csv -ErrorAction SilentlyContinue
Remove-Item student_report.png -ErrorAction SilentlyContinue
python train.py
```

## 8. Типичные ошибки

| Ошибка | Решение |
|--------|---------|
| `ModuleNotFoundError: xgboost` (или `catboost`, `sentence_transformers`, `wordcloud`, `nltk`) | `pip install -r requirements.txt` |
| `LookupError: stopwords not found` | `python -c "import nltk; nltk.download('stopwords')"` |
| SBERT качается медленно | Норма на первом запуске. Кеш в `~/.cache/huggingface/`. На повторных запусках — мгновенно. |
| `artifacts not found` при старте `app.py` | Сам вызовет `train.py`, дождитесь окончания. |
| Кириллица крякозябрами в консоли | Косметика Windows-консоли. В БД, веб-интерфейсе и `student_report.png` всё корректно UTF-8. |
| Порт 5000 занят | В `app.py`, в самом низу: `app.run(host="127.0.0.1", port=5000, debug=False)` — заменить порт. |
| `OSError: [WinError 10013]` | Запустить терминал от администратора либо взять другой порт. |
| Пустые рекомендации для нового пользователя | Норма: для нового UID нет истории, отдаётся глобальный топ. После 1-2 отзывов CF начинает работать. |

## 9. Команды одной строкой

Полный сценарий с нуля:

```powershell
pip install -r requirements.txt
python -m pytest tests/
python train.py
python app.py
```

После этого:
- `student_report.png` лежит в корне проекта
- веб-интерфейс открыт на <http://127.0.0.1:5000>
- логи пишутся в `app.log`
- SQLite — в `reviews.db`
