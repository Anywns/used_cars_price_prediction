"""One lightweight Anthropic agent that explains model price predictions.

The ML model predicts the numeric price. This module only adds a short
human-readable explanation for the Telegram bot. If Anthropic is not configured
or the API call fails, the bot returns a local fallback comment and still works.
"""
from __future__ import annotations

import os
from typing import Any


UNSUPPORTED_LUXURY_BRANDS = {
    "ferrari",
    "lamborghini",
    "bentley",
    "rolls-royce",
    "rolls royce",
    "mclaren",
    "aston-martin",
    "aston martin",
    "bugatti",
    "maybach",
}

MIDDLE_SEGMENT_BRANDS = {
    "acura",
    "audi",
    "bmw",
    "cadillac",
    "genesis",
    "infiniti",
    "lexus",
    "lincoln",
    "mercedes-benz",
    "mercedes",
    "mini",
    "porsche",
    "tesla",
    "volvo",
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def is_unsupported_luxury(car: dict[str, Any]) -> bool:
    """Cars outside this project scope: rare luxury/supercars."""
    return _norm(car.get("manufacturer")) in UNSUPPORTED_LUXURY_BRANDS


def detect_market_segment(car: dict[str, Any]) -> str:
    """Coarse supported segment used only for display/commentary."""
    manufacturer = _norm(car.get("manufacturer"))
    return "middle" if manufacturer in MIDDLE_SEGMENT_BRANDS else "low"


def fallback_comment(car: dict[str, Any], price: float, segment: str) -> str:
    """Local explanation when Anthropic is unavailable."""
    year = int(car.get("year", 0) or 0)
    odometer = int(car.get("odometer", 0) or 0)
    condition = str(car.get("condition", "unknown"))
    manufacturer = str(car.get("manufacturer", "unknown"))
    car_type = str(car.get("type", "unknown"))

    reasons: list[str] = []

    if year >= 2018:
        reasons.append("свежий год выпуска поддерживает цену")
    elif year <= 2005:
        reasons.append("возраст автомобиля снижает оценку")
    else:
        reasons.append("год выпуска находится в обычном диапазоне для б/у рынка")

    if odometer <= 50_000:
        reasons.append("невысокий пробег повышает стоимость")
    elif odometer >= 180_000:
        reasons.append("большой пробег снижает стоимость")
    else:
        reasons.append("пробег выглядит типичным для подержанного автомобиля")

    if condition in {"excellent", "like new", "new"}:
        reasons.append("хорошее состояние повышает оценку")
    elif condition in {"fair", "salvage"}:
        reasons.append("состояние может снижать цену")

    segment_ru = "средний" if segment == "middle" else "низкий/массовый"
    return (
        f"Комментарий: автомобиль отнесён к сегменту «{segment_ru}». "
        f"Для {manufacturer} {car_type} оценка около ${price:,.0f} получилась потому, что "
        + "; ".join(reasons[:3])
        + ". Это приблизительная ML-оценка, а не финальная рыночная стоимость."
    )


def _call_anthropic(car: dict[str, Any], price: float, segment: str) -> str | None:
    """Call Anthropic API. Return None if key/dependency/API is unavailable."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        from anthropic import Anthropic
    except Exception:
        return None

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    if model.startswith("anthropic/"):
        model = model.split("/", 1)[1]

    system = (
        "You are a concise assistant inside a Telegram used-car price bot. "
        "The project scope is low and middle segment cars from a US used-car marketplace. "
        "Explain the model prediction in Russian. Do not invent exact market facts. "
        "Mention that this is an approximate ML estimate. Keep it under 5 short sentences."
    )
    user = (
        f"Predicted price: ${price:,.0f}\n"
        f"Supported segment: {segment}\n"
        f"Car features: {car}\n"
        "Write a short Russian explanation of why the predicted price may be like this."
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=220,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [getattr(block, "text", "").strip() for block in response.content]
        text = "\n".join(part for part in parts if part).strip()
        return text or None
    except Exception:
        return None


def explain_price(car: dict[str, Any], price: float) -> str:
    """Telegram-ready explanation for the predicted price."""
    segment = detect_market_segment(car)
    return _call_anthropic(car, price, segment) or fallback_comment(car, price, segment)
