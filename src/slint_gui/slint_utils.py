"""Slint ↔ Python helpers."""
from __future__ import annotations

from slint import ListModel


def slint_int(value: object, default: int = 0) -> int:
    """Coerce Slint SpinBox/ComboBox values (often float) to int."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def bind_string_list_model(window: object, prop: str, items: list[str]) -> ListModel:
    """
    Assign a ListModel to a Slint `[string]` property and keep a strong ref.

    Prevents 'Model implementation is lacking self object' warnings on some
    Slint Python builds when the model is collected too early.
    """
    model = ListModel(list(items))
    cache = getattr(window, "_list_model_cache", None)
    if cache is None:
        cache = {}
        setattr(window, "_list_model_cache", cache)
    cache[prop] = model
    setattr(window, prop, model)
    return model
