"""Точка входа приложения «Оценщик цены подержанного авто».

Поток работы:
  1. provider готовит модель по цепочке loader → cleaner → builder
     (каждый этап пропускается, если артефакт уже лежит в resources/);
  2. готовая модель передаётся в конструктор CarPriceBot;
  3. бот запускается и обслуживает диалог с пользователем.

Запуск:
    export BOT_TOKEN="токен_от_@BotFather"   # PowerShell: $env:BOT_TOKEN="..."
    uv run python app.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from used_cars.bot import CarPriceBot
from used_cars.logging_setup import setup_logging
from used_cars.provider import provide_model

log = logging.getLogger("app")


def main() -> None:
    setup_logging()
    load_dotenv(Path(__file__).resolve().parent / ".env")

    token = os.environ.get("BOT_TOKEN")
    if not token:
        log.error("Не задан BOT_TOKEN. Получите токен у @BotFather и задайте "
                  'переменную окружения: export BOT_TOKEN="..."')
        sys.exit(1)

    model = provide_model()

    bot = CarPriceBot(model, token)
    bot.run()


if __name__ == "__main__":
    main()
