"""Этап 1 цепочки — loader.

Задача: гарантировать, что сырой датасет `resources/vehicles.csv` существует.
Если файла нет — скачивает его с Kaggle и распаковывает в resources/.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

_KAGGLE_HELP = (
    "Не удалось скачать датасет с Kaggle — нужны учётные данные Kaggle. Варианты:\n"
    "  • положить kaggle.json в ~/.kaggle/ (kaggle.com → Settings → API → Create New Token); или\n"
    "  • задать переменные окружения KAGGLE_USERNAME и KAGGLE_KEY; или\n"
    "  • выполнить `uv run kaggle auth login`.\n"
    "Либо обойти загрузку: положить готовый car_price_model.pkl (или vehicles_clean.csv) "
    "в папку resources/ — тогда Kaggle не понадобится."
)


def ensure_raw_csv() -> Path:
    """Вернуть путь к сырому vehicles.csv, скачав его при отсутствии.

    Returns:
        Путь к `resources/vehicles.csv`.
    """
    config.ensure_resources_dir()

    if config.RAW_CSV.exists():
        log.info("Сырые данные на месте: %s", config.RAW_CSV)
        return config.RAW_CSV

    log.info("Сырых данных нет — скачиваю с Kaggle: %s", config.KAGGLE_DATASET)
    _download_from_kaggle()

    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"После скачивания не найден {config.RAW_CSV}. "
            "Проверьте содержимое архива датасета."
        )
    log.info("Скачано: %s", config.RAW_CSV)
    return config.RAW_CSV


def _configure_kaggle_env() -> None:
    """Если kaggle.json лежит в корне проекта — указать его SDK через KAGGLE_CONFIG_DIR.

    SDK Kaggle ищет kaggle.json в каталоге KAGGLE_CONFIG_DIR (по умолчанию ~/.kaggle).
    Здесь разрешаем держать файл прямо в корне проекта.
    """
    root_json = config.BASE_DIR / "kaggle.json"
    if root_json.exists() and not os.environ.get("KAGGLE_CONFIG_DIR"):
        os.environ["KAGGLE_CONFIG_DIR"] = str(config.BASE_DIR)
        log.info("Использую kaggle.json из корня проекта")


def _has_kaggle_credentials() -> bool:
    """Есть ли где-нибудь учётные данные Kaggle (не дёргая SDK)."""
    if (config.BASE_DIR / "kaggle.json").exists():
        return True
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    if (Path.home() / ".kaggle" / "access_token").exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    return False


def _download_from_kaggle() -> None:
    """Скачать и распаковать датасет через Kaggle API.

    Учётные данные проверяем заранее: SDK Kaggle при их отсутствии печатает свою
    справку и завершает процесс (SystemExit), поэтому до него не доходим.
    """
    _configure_kaggle_env()
    if not _has_kaggle_credentials():
        raise RuntimeError(_KAGGLE_HELP)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            config.KAGGLE_DATASET,
            path=str(config.RESOURCES_DIR),
            unzip=True,
        )
    except BaseException as exc:
        raise RuntimeError(_KAGGLE_HELP) from exc
