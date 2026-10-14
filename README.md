# 🚗 Предсказание цены подержанных авто

ML-проект (регрессия) с Telegram-ботом-оценщиком. По характеристикам автомобиля
модель предсказывает его рыночную цену.

**Авторы:** Огей Анна, Долгополова Анна

Проект состоит из двух частей:

- **Рантайм** (`app.py` + пакет `used_cars/`) — приложение, которое само готовит
  модель и запускает Telegram-бота. Только `.py`.
- **Исследование** (`research/`) — Jupyter-ноутбуки с EDA и подбором лучшей модели.

Данные: [Craigslist Used Cars](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
(Kaggle, ~426 тыс. объявлений).

---

##  Стек технологий

| Назначение | Технологии |
|------------|------------|
| Язык | Python 3.12 |
| Пакеты и окружение | [uv](https://docs.astral.sh/uv/) |
| Данные и ML | pandas, numpy, scikit-learn **1.6.1**, joblib |
| Telegram-бот | python-telegram-bot 22.x, python-dotenv |
| Загрузка данных | kaggle |
| Исследование | Jupyter (ipykernel), matplotlib, seaborn |

---

## Структура проекта

```
used_cars_analyzer/
├── app.py                     # точка входа: provide_model() → CarPriceBot(model).run()
├── pyproject.toml             # зависимости и метаданные проекта (uv)
├── uv.lock                    # зафиксированные версии зависимостей
├── .python-version            # версия Python (3.12)
├── .env                       # BOT_TOKEN=... (создаёте сами, не в git)
├── kaggle.json                # ключ Kaggle (создаёте сами, не в git)
│
├── used_cars/                 # рантайм-пакет
│   ├── config.py              # пути, признаки, пороги чистки, гиперпараметры
│   ├── logging_setup.py       # логирование в консоль и в logs/<дата>.log
│   ├── loader.py              # этап 1: скачать датасет с Kaggle
│   ├── cleaner.py             # этап 2: очистить данные → vehicles_clean.csv
│   ├── builder.py             # этап 3: обучить модель → car_price_model.pkl
│   ├── provider.py            # оркестратор loader → cleaner → builder
│   ├── bot.py                 # CarPriceBot: диалог с пользователем и предсказание
│   └── __main__.py            # `python -m used_cars` — трассировка создания и use модели
│
├── research/                  # исследование (Jupyter)
│   ├── 01_eda_data_prep.ipynb # EDA + предобработка
│   └── 02_models.ipynb        # сравнение моделей, кросс-валидация, подбор гиперпараметров
│
├── resources/                 # артефакты (не в git, генерируются на 1-м запуске)
│   ├── vehicles.csv           # сырьё с Kaggle (~1.4 ГБ)
│   ├── vehicles_clean.csv     # очищенные данные
│   └── car_price_model.pkl    # обученный pipeline
│
├── logs/                      # логи по датам: logs/YYYY-MM-DD.log (не в git)
│
├── build_deck.py              # генератор презентации (.pptx)
└── speaker_notes.md           # заметки докладчика
```

### Как устроен рантайм

`app.py` вызывает `provider.provide_model()` — ленивую цепочку, где каждый этап
пропускается, если его артефакт уже лежит в `resources/`:

```
builder.ensure_model()            нет .pkl  ↓
  └─ cleaner.ensure_clean_csv()   нет clean.csv ↓
       └─ loader.ensure_raw_csv() нет raw.csv → скачать с Kaggle
```

Готовая модель передаётся в конструктор `CarPriceBot`, и бот запускается на
long-polling. Если `.pkl` уже есть — ни данные, ни Kaggle не трогаются.

---

##  Установка

### 1. Python и uv

Весь проект работает через **uv** — он сам поставит нужную версию Python и зависимости.

**Установка uv:**

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Python 3.12** uv подтянет автоматически (по файлу `.python-version`) при первом
`uv sync`. При желании можно поставить заранее:

```bash
uv python install 3.12
```

### 2. Зависимости

```bash
# только для бота
uv sync

# вместе с зависимостями для ноутбуков (matplotlib, seaborn, jupyter)
uv sync --all-groups
```

Команда создаёт виртуальное окружение `.venv/` и скачивает все пакеты из `uv.lock`.

### 3. Учётные данные Kaggle

Нужны, чтобы приложение скачало датасет на первом запуске.

1. kaggle.com → **Settings** → **API** → **Create New Token** → скачается `kaggle.json`.
2. Положите `kaggle.json` **в корень проекта** (либо в `~/.kaggle/`).

### 4. Токен Telegram-бота

1. Получите токен у [@BotFather](https://t.me/BotFather) в Telegram.
2. Создайте файл `.env` в корне проекта:

```env
BOT_TOKEN=сюда_токен_от_BotFather
```

---

## Запуск

### Бот

```bash
uv run python app.py
```

- **Первый запуск** скачивает датасет (~1.4 ГБ), чистит данные и обучает модель —
  несколько минут. Всё кэшируется в `resources/`.
- **Последующие запуски** мгновенно загружают готовую модель.
- **Переобучить заново** — удалите файл в `resources/` (модель / clean / raw) и
  запустите снова.

После старта откройте Telegram и напишите боту `/start`. Бот задаёт параметры авто
и возвращает оценку. Допустимые пределы ввода:

| Параметр | Пределы / варианты |
|----------|--------------------|
| Год выпуска | 1990–2022 |
| Пробег (мили) | 0–400 000 |
| Марка | латиницей (ford, toyota, bmw …) |
| Остальные (состояние, топливо, КПП, кузов и т.д.) | выбор на клавиатуре |

### Трассировка модели (без бота)

Прогнать всю цепочку «создание → использование» и увидеть демо-предсказание:

```bash
uv run python -m used_cars
```

---

##  Логи

Все запуски (`app.py` и `python -m used_cars`) пишут логи одновременно в консоль и
в файл `logs/<YYYY-MM-DD>.log` — по одному файлу на дату. Папка `logs/` создаётся
автоматически и в git не попадает.

---

## Исследования (`research/`)

Ноутбуки запускаются в Google Colab или локально (`uv run jupyter lab research/`
после `uv sync --all-groups`).

- **`01_eda_data_prep.ipynb`** — разведочный анализ: распределение цены (скошено →
  обучаемся на `log(price)`), зависимость цены от года и пробега, пропуски,
  корреляции; чистка данных и сохранение `vehicles_clean.csv`.
- **`02_models.ipynb`** — сравнение трёх моделей (LinearRegression, RandomForest,
  GradientBoosting) на кросс-валидации, подбор гиперпараметров через
  `RandomizedSearchCV`. Победитель — **RandomForest**.

Предобработка (One-Hot + масштабирование) вынесена **внутрь sklearn-pipeline** —
так нет утечки данных при кросс-валидации. Выбранная в ноутбуках модель
воспроизводится в проде модулем `builder.py`.

**Качество (holdout, RandomForest):** R² ≈ 0.80, MAE ≈ $3 400.
