"""Аугментация временных окон для повышения устойчивости обучения."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


class AugmentationError(ValueError):
    """Исключение для ошибок аугментации временных окон."""


SUPPORTED_AUGMENTATION_METHODS = {"noise", "scale", "shift"}


def _validate_window(window: np.ndarray) -> np.ndarray:
    if not isinstance(window, np.ndarray):
        raise AugmentationError("window должен быть numpy-массивом")
    if window.ndim != 2:
        raise AugmentationError("window должен иметь форму (T, n_features)")
    if window.shape[0] == 0 or window.shape[1] == 0:
        raise AugmentationError("window не должен быть пустым")
    return np.asarray(window, dtype=np.float32)


def augment_window(
    window: np.ndarray,
    method: str = "noise",
    params: Mapping[str, float | int] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Аугментирует одно временное окно заданным методом."""
    x_window = _validate_window(window)
    parameters = dict(params or {})
    generator = rng or np.random.default_rng()

    if method not in SUPPORTED_AUGMENTATION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_AUGMENTATION_METHODS))
        raise AugmentationError(f"method должен быть одним из: {supported}")

    if method == "noise":
        std = float(parameters.get("std", 0.01))
        if std < 0:
            raise AugmentationError("Параметр std должен быть >= 0")
        noise = generator.normal(0.0, std, size=x_window.shape).astype(np.float32)
        return (x_window + noise).astype(np.float32)

    if method == "scale":
        min_scale = float(parameters.get("min_scale", 0.9))
        max_scale = float(parameters.get("max_scale", 1.1))
        if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
            raise AugmentationError("Для scale требуется 0 < min_scale <= max_scale")
        factor = float(generator.uniform(min_scale, max_scale))
        return (x_window * factor).astype(np.float32)

    shift_min = int(parameters.get("shift_min", -5))
    shift_max = int(parameters.get("shift_max", 5))
    if shift_min > shift_max:
        raise AugmentationError("Для shift требуется shift_min <= shift_max")
    shift_value = int(generator.integers(shift_min, shift_max + 1))
    return np.roll(x_window, shift=shift_value, axis=0).astype(np.float32)


def augment_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    aug_factor: int = 2,
    methods: Iterable[str] | None = None,
    params: Mapping[str, float | int] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Увеличивает train-набор временных окон за счёт аугментации."""
    if not isinstance(x_data, np.ndarray) or x_data.ndim != 3:
        raise AugmentationError("x_data должен иметь форму (n_samples, T, n_features)")
    if not isinstance(y_data, np.ndarray) or y_data.ndim != 1:
        raise AugmentationError("y_data должен быть одномерным массивом")
    if x_data.shape[0] == 0:
        raise AugmentationError("x_data не должен быть пустым")
    if x_data.shape[0] != y_data.shape[0]:
        raise AugmentationError("Количество окон в x_data должно совпадать с длиной y_data")
    if not isinstance(aug_factor, int) or aug_factor <= 0:
        raise AugmentationError("aug_factor должен быть положительным целым числом")

    methods_tuple = tuple(str(method) for method in methods) if methods is not None else ("noise", "scale")
    if not methods_tuple:
        raise AugmentationError("methods не должен быть пустым")
    for method in methods_tuple:
        if method not in SUPPORTED_AUGMENTATION_METHODS:
            supported = ", ".join(sorted(SUPPORTED_AUGMENTATION_METHODS))
            raise AugmentationError(f"methods должен содержать только: {supported}")

    generator = rng or np.random.default_rng()
    x_batches = [np.asarray(x_data, dtype=np.float32)]
    y_batches = [np.asarray(y_data, dtype=np.int64)]

    for _ in range(aug_factor - 1):
        generated = np.empty_like(x_batches[0], dtype=np.float32)
        for idx, window in enumerate(x_data):
            method = str(generator.choice(methods_tuple))
            generated[idx] = augment_window(window, method=method, params=params, rng=generator)
        x_batches.append(generated)
        y_batches.append(np.asarray(y_data, dtype=np.int64))

    return np.vstack(x_batches).astype(np.float32), np.hstack(y_batches).astype(np.int64)
