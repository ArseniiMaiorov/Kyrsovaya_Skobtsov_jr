from __future__ import annotations

import numpy as np
import pytest

from src.data.augmentation import AugmentationError, augment_dataset, augment_window


def _make_window() -> np.ndarray:
    return np.arange(12, dtype=np.float32).reshape(4, 3)


def test_augment_window_noise_scale_shift():
    rng = np.random.default_rng(42)
    window = _make_window()

    noisy = augment_window(window, method="noise", params={"std": 0.01}, rng=rng)
    scaled = augment_window(window, method="scale", params={"min_scale": 1.0, "max_scale": 1.0}, rng=rng)
    shifted = augment_window(window, method="shift", params={"shift_min": 1, "shift_max": 1}, rng=rng)

    assert noisy.shape == window.shape
    assert not np.allclose(noisy, window)
    assert np.allclose(scaled, window)
    assert np.allclose(shifted, np.roll(window, 1, axis=0))


def test_augment_window_errors():
    with pytest.raises(AugmentationError, match="numpy-массивом"):
        augment_window("bad")  # type: ignore[arg-type]

    with pytest.raises(AugmentationError, match="форму"):
        augment_window(np.array([1.0, 2.0]))

    with pytest.raises(AugmentationError, match="не должен быть пустым"):
        augment_window(np.empty((0, 2), dtype=np.float32))

    with pytest.raises(AugmentationError, match="method"):
        augment_window(_make_window(), method="flip")

    with pytest.raises(AugmentationError, match="std"):
        augment_window(_make_window(), method="noise", params={"std": -1.0})

    with pytest.raises(AugmentationError, match="min_scale"):
        augment_window(_make_window(), method="scale", params={"min_scale": 1.2, "max_scale": 1.0})

    with pytest.raises(AugmentationError, match="shift_min"):
        augment_window(_make_window(), method="shift", params={"shift_min": 3, "shift_max": 1})


def test_augment_dataset_success():
    x_data = np.stack([_make_window(), _make_window() + 1], axis=0)
    y_data = np.array([0, 1], dtype=np.int64)

    x_aug, y_aug = augment_dataset(
        x_data,
        y_data,
        aug_factor=3,
        methods=("scale",),
        params={"min_scale": 1.0, "max_scale": 1.0},
        rng=np.random.default_rng(42),
    )

    assert x_aug.shape == (6, 4, 3)
    assert y_aug.tolist() == [0, 1, 0, 1, 0, 1]


def test_augment_dataset_errors():
    x_data = np.stack([_make_window()], axis=0)
    y_data = np.array([0], dtype=np.int64)

    with pytest.raises(AugmentationError, match="форму"):
        augment_dataset(np.array([1.0, 2.0]), y_data)

    with pytest.raises(AugmentationError, match="одномерным"):
        augment_dataset(x_data, np.array([[0]]))

    with pytest.raises(AugmentationError, match="не должен быть пустым"):
        augment_dataset(np.empty((0, 4, 3), dtype=np.float32), np.empty((0,), dtype=np.int64))

    with pytest.raises(AugmentationError, match="должно совпадать"):
        augment_dataset(x_data, np.array([0, 1], dtype=np.int64))

    with pytest.raises(AugmentationError, match="aug_factor"):
        augment_dataset(x_data, y_data, aug_factor=0)

    with pytest.raises(AugmentationError, match="methods не должен быть пустым"):
        augment_dataset(x_data, y_data, methods=())

    with pytest.raises(AugmentationError, match="methods должен содержать только"):
        augment_dataset(x_data, y_data, methods=("flip",))
