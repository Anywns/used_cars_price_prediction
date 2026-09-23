"""Трассировка полного жизненного цикла модели: создание → использование.

Запуск:  uv run python -m used_cars

Прогоняет цепочку loader → cleaner → builder (создание модели), затем делает
демонстрационное предсказание (использование модели), логируя каждый шаг.
Telegram-бота НЕ запускает — это инструмент проверки конвейера.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .bot import CarPriceBot
from .logging_setup import setup_logging
from .provider import provide_model

log = logging.getLogger("used_cars.trace")

SAMPLE_CAR = {
    "manufacturer": "toyota",
    "year": 2017,
    "odometer": 60000,
    "fuel": "gas",
    "transmission": "automatic",
    "type": "sedan",
    "condition": "good",
}


def main() -> None:
    setup_logging()
    log.info("================ СОЗДАНИЕ модели ================")
    model = provide_model()

    log.info("================ ИСПОЛЬЗОВАНИЕ модели ================")
    bot = CarPriceBot(model, token="trace")

    row = dict(bot.defaults)
    row.update(SAMPLE_CAR)
    X = pd.DataFrame([row])[bot.features]
    price = float(np.expm1(model.predict(X)[0]))

    log.info("Вход: %s", SAMPLE_CAR)
    log.info("Предсказанная цена: $%s", f"{price:,.0f}")
    log.info("================ ГОТОВО ================")


if __name__ == "__main__":
    main()
