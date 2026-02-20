"""Утилиты для загрузки и валидации конфигурации проекта."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_TOP_LEVEL_SECTIONS = (
    "task",
    "sequence",
    "data",
    "compute_budget",
    "training",
)


class ConfigError(ValueError):
    """Исключение для ошибок контрактов конфигурации."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Загружает YAML-конфигурацию и валидирует её."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    if not isinstance(config, dict):
        raise ConfigError("Корневой элемент конфигурации должен быть словарем")

    validate_config(config)
    return config


def _require_positive_int(value: Any, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{field_name}' должно быть положительным целым числом")


def validate_config(config: Mapping[str, Any]) -> None:
    """Проверяет разделы и поля конфигурации на соответствие контракту проекта."""
    missing_sections = [name for name in REQUIRED_TOP_LEVEL_SECTIONS if name not in config]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ConfigError(f"Отсутствуют обязательные разделы верхнего уровня: {missing}")

    task = config["task"]
    if task.get("type") != "multiclass":
        raise ConfigError("На текущем этапе task.type должен быть равен 'multiclass'")
    if set(task.get("labels", [])) != {0, 1, 2}:
        raise ConfigError("task.labels должен быть строго равен [0, 1, 2]")
    if not isinstance(task.get("target_col"), str) or not task["target_col"].strip():
        raise ConfigError("task.target_col должен быть непустой строкой")

    sequence = config["sequence"]
    if sequence.get("mode") != "time_windows":
        raise ConfigError("sequence.mode должен быть равен 'time_windows'")
    _require_positive_int(sequence.get("T"), "sequence.T")
    _require_positive_int(sequence.get("stride"), "sequence.stride")
    overlap = sequence.get("overlap")
    if not isinstance(overlap, (int, float)) or not 0 <= overlap < 1:
        raise ConfigError("sequence.overlap должен быть в диапазоне [0, 1)")

    data = config["data"]
    if data.get("format") != "csv":
        raise ConfigError("data.format должен быть равен 'csv'")
    for text_field in ("path", "sep", "encoding"):
        if not isinstance(data.get(text_field), str) or not data[text_field]:
            raise ConfigError(f"data.{text_field} должен быть непустой строкой")
    null_tokens = data.get("null_tokens")
    if not isinstance(null_tokens, list) or not null_tokens:
        raise ConfigError("data.null_tokens должен быть непустым списком")

    compute_budget = config["compute_budget"]
    _require_positive_int(compute_budget.get("population_size"), "compute_budget.population_size")
    _require_positive_int(compute_budget.get("generations"), "compute_budget.generations")
    _require_positive_int(compute_budget.get("max_epochs_fitness"), "compute_budget.max_epochs_fitness")
    _require_positive_int(compute_budget.get("max_epochs_final"), "compute_budget.max_epochs_final")

    gain = compute_budget.get("target_macro_f1_gain_vs_baseline")
    if not isinstance(gain, (int, float)) or gain <= 0:
        raise ConfigError("compute_budget.target_macro_f1_gain_vs_baseline должен быть больше 0")

    training = config["training"]
    _require_positive_int(training.get("early_stopping_patience"), "training.early_stopping_patience")
    _require_positive_int(training.get("reduce_lr_patience"), "training.reduce_lr_patience")

    factor = training.get("reduce_lr_factor")
    if not isinstance(factor, (int, float)) or not 0 < factor < 1:
        raise ConfigError("training.reduce_lr_factor должен быть в диапазоне (0, 1)")
