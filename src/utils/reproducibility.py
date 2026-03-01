"""Утилиты для воспроизводимости экспериментов и сохранения артефактов запуска."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import random
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

DEFAULT_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "PyYAML",
    "pytest",
    "pytest-cov",
    "matplotlib",
    "seaborn",
    "tqdm",
    "jupyter",
    "tensorflow",
    "xlrd",
)

SUPPORTED_CHECKSUM_ALGORITHMS = {"sha256"}


class ReproducibilityError(ValueError):
    """Исключение для ошибок воспроизводимости и артефактов запуска."""


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ReproducibilityError(f"{field_name} должен быть неотрицательным целым числом")
    return int(value)


def _normalize_project_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Корень проекта не найден: {root}")
    return root


def _sanitize_stage_name(stage_name: str) -> str:
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ReproducibilityError("stage_name должен быть непустой строкой")

    normalized = "".join(char if char.isalnum() else "_" for char in stage_name.strip().lower())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    normalized = normalized.strip("_")
    return normalized or "runtime"


def _set_tensorflow_seed_if_available(seed: int) -> bool:
    try:
        tensorflow = importlib.import_module("tensorflow")
    except Exception:
        return False

    tf_random = getattr(tensorflow, "random", None)
    if tf_random is None or not hasattr(tf_random, "set_seed"):
        return False

    tf_random.set_seed(seed)
    return True


def set_global_seed(seed: int, enable_tensorflow: bool = False) -> bool:
    """Фиксирует seed для стандартного генератора, NumPy и, при необходимости, TensorFlow."""
    seed_value = _require_non_negative_int(seed, "seed")

    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if not enable_tensorflow:
        return False

    return _set_tensorflow_seed_if_available(seed_value)


def compute_file_checksum(
    path: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    """Считает checksum файла потоково, без загрузки файла целиком в память."""
    if algorithm not in SUPPORTED_CHECKSUM_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_CHECKSUM_ALGORITHMS))
        raise ReproducibilityError(f"algorithm должен быть одним из: {supported}")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ReproducibilityError("chunk_size должен быть положительным целым числом")

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Файл для checksum не найден: {file_path}")

    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def build_file_fingerprint(path: str | Path, algorithm: str = "sha256") -> dict[str, Any]:
    """Формирует fingerprint файла: путь, размер и checksum."""
    file_path = Path(path)
    checksum = compute_file_checksum(file_path, algorithm=algorithm)
    return {
        "path": str(file_path),
        "size_bytes": int(file_path.stat().st_size),
        "checksum_algorithm": algorithm,
        "checksum": checksum,
    }


def get_package_version(package_name: str) -> str | None:
    """Возвращает версию пакета или `None`, если пакет не установлен."""
    if not isinstance(package_name, str) or not package_name.strip():
        raise ReproducibilityError("package_name должен быть непустой строкой")

    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_library_versions(packages: Iterable[str] = DEFAULT_TRACKED_PACKAGES) -> dict[str, str | None]:
    """Собирает версии выбранных библиотек из текущего окружения."""
    package_names = tuple(packages)
    if not package_names:
        raise ReproducibilityError("Список пакетов для сбора версий не должен быть пустым")

    return {name: get_package_version(name) for name in package_names}


def snapshot_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Сохраняет YAML-снимок конфигурации с сохранением порядка ключей."""
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(dict(config), file_obj, allow_unicode=True, sort_keys=False)
    return snapshot_path


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_reproducibility_settings(config: Mapping[str, Any]) -> tuple[int, str]:
    repro_cfg = config.get("reproducibility")
    if not isinstance(repro_cfg, Mapping):
        raise ReproducibilityError("В конфигурации отсутствует раздел 'reproducibility'")

    seed = _require_non_negative_int(repro_cfg.get("global_seed"), "reproducibility.global_seed")
    algorithm = repro_cfg.get("checksum_algorithm")
    if algorithm not in SUPPORTED_CHECKSUM_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_CHECKSUM_ALGORITHMS))
        raise ReproducibilityError(f"reproducibility.checksum_algorithm должен быть одним из: {supported}")

    return seed, str(algorithm)


def build_runtime_metadata(
    config: Mapping[str, Any],
    project_root: str | Path,
    stage_name: str,
) -> dict[str, Any]:
    """Собирает runtime-метаданные: seed, окружение и fingerprint датасетов."""
    root = _normalize_project_root(project_root)
    seed, algorithm = _extract_reproducibility_settings(config)
    stage_slug = _sanitize_stage_name(stage_name)

    data_cfg = config.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ReproducibilityError("В конфигурации отсутствует раздел 'data'")

    labeled_cfg = data_cfg.get("labeled")
    unlabeled_cfg = data_cfg.get("unlabeled")
    if not isinstance(labeled_cfg, Mapping) or not isinstance(unlabeled_cfg, Mapping):
        raise ReproducibilityError("Разделы data.labeled и data.unlabeled должны быть словарями")

    labeled_path = (root / str(labeled_cfg["path"])).resolve()
    unlabeled_path = (root / str(unlabeled_cfg["path"])).resolve()

    labeled_fingerprint = build_file_fingerprint(labeled_path, algorithm=algorithm)
    unlabeled_fingerprint = build_file_fingerprint(unlabeled_path, algorithm=algorithm)

    labeled_fingerprint["path"] = _relative_or_absolute(labeled_path, root)
    unlabeled_fingerprint["path"] = _relative_or_absolute(unlabeled_path, root)

    return {
        "stage_name": stage_slug,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_seed": seed,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "library_versions": collect_library_versions(),
        "datasets": {
            "labeled": labeled_fingerprint,
            "unlabeled": unlabeled_fingerprint,
        },
    }


def initialize_reproducibility(
    config: Mapping[str, Any],
    project_root: str | Path,
    stage_name: str,
    use_tensorflow: bool = False,
) -> dict[str, Any]:
    """Фиксирует seed и сохраняет артефакты воспроизводимости в `output/artifacts`."""
    root = _normalize_project_root(project_root)
    seed, _ = _extract_reproducibility_settings(config)
    stage_slug = _sanitize_stage_name(stage_name)

    tensorflow_seed_applied = set_global_seed(seed, enable_tensorflow=use_tensorflow)

    artifacts_dir = root / "output" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = snapshot_config(config, artifacts_dir / "config_snapshot.yaml")
    metadata_path = artifacts_dir / f"reproducibility_{stage_slug}.json"

    metadata = build_runtime_metadata(config, root, stage_name=stage_name)
    metadata["config_snapshot_path"] = _relative_or_absolute(config_snapshot_path, root)
    metadata["tensorflow_seed_applied"] = tensorflow_seed_applied

    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return {
        "global_seed": seed,
        "config_snapshot_path": _relative_or_absolute(config_snapshot_path, root),
        "metadata_path": _relative_or_absolute(metadata_path, root),
        "tensorflow_seed_applied": tensorflow_seed_applied,
        "metadata": metadata,
    }
