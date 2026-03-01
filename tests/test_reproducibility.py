from __future__ import annotations

import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.utils import reproducibility as repro
from src.utils.reproducibility import (
    ReproducibilityError,
    _extract_reproducibility_settings,
    _normalize_project_root,
    _relative_or_absolute,
    _sanitize_stage_name,
    _set_tensorflow_seed_if_available,
    build_file_fingerprint,
    build_runtime_metadata,
    collect_library_versions,
    compute_file_checksum,
    get_package_version,
    initialize_reproducibility,
    set_global_seed,
    snapshot_config,
)


def _make_runtime_config() -> dict:
    return {
        "reproducibility": {"global_seed": 42, "checksum_algorithm": "sha256"},
        "data": {
            "labeled": {"path": "data/labeled.csv"},
            "unlabeled": {"path": "data/unlabeled.csv"},
        },
    }


def test_normalize_project_root_success_and_missing(tmp_path):
    normalized = _normalize_project_root(tmp_path)
    assert normalized == tmp_path.resolve()

    with pytest.raises(FileNotFoundError, match="Корень проекта не найден"):
        _normalize_project_root(tmp_path / "missing")


def test_sanitize_stage_name_success_and_error():
    assert _sanitize_stage_name(" Stage 4: Baseline ") == "stage_4_baseline"

    with pytest.raises(ReproducibilityError, match="stage_name"):
        _sanitize_stage_name("   ")


def test_set_tensorflow_seed_if_available_import_error(monkeypatch: pytest.MonkeyPatch):
    def fake_import_module(name: str):
        raise ImportError("no tensorflow")

    monkeypatch.setattr(repro.importlib, "import_module", fake_import_module)
    assert _set_tensorflow_seed_if_available(42) is False


def test_set_tensorflow_seed_if_available_without_random(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(repro.importlib, "import_module", lambda name: object())
    assert _set_tensorflow_seed_if_available(42) is False


def test_set_tensorflow_seed_if_available_success(monkeypatch: pytest.MonkeyPatch):
    called: dict[str, int] = {}

    def set_seed(seed: int) -> None:
        called["seed"] = seed

    fake_tf = SimpleNamespace(random=SimpleNamespace(set_seed=set_seed))
    monkeypatch.setattr(repro.importlib, "import_module", lambda name: fake_tf)

    assert _set_tensorflow_seed_if_available(7) is True
    assert called["seed"] == 7


def test_set_global_seed_success_and_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(repro, "_set_tensorflow_seed_if_available", lambda seed: False)

    tf_applied = set_global_seed(42)
    python_random_1 = random.random()
    numpy_random_1 = float(np.random.rand())

    set_global_seed(42)
    python_random_2 = random.random()
    numpy_random_2 = float(np.random.rand())

    assert tf_applied is False
    assert repro.os.environ["PYTHONHASHSEED"] == "42"
    assert python_random_1 == python_random_2
    assert numpy_random_1 == numpy_random_2

    assert set_global_seed(42, enable_tensorflow=True) is False

    with pytest.raises(ReproducibilityError, match="seed"):
        set_global_seed(-1)


def test_compute_file_checksum_and_fingerprint_success(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("abc", encoding="utf-8")

    checksum = compute_file_checksum(file_path)
    fingerprint = build_file_fingerprint(file_path)

    assert checksum == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert fingerprint["size_bytes"] == 3
    assert fingerprint["checksum"] == checksum


def test_compute_file_checksum_validation_errors(tmp_path):
    with pytest.raises(ReproducibilityError, match="algorithm"):
        compute_file_checksum(tmp_path / "sample.txt", algorithm="md5")

    with pytest.raises(ReproducibilityError, match="chunk_size"):
        compute_file_checksum(tmp_path / "sample.txt", chunk_size=0)

    with pytest.raises(FileNotFoundError, match="Файл для checksum не найден"):
        compute_file_checksum(tmp_path / "missing.txt")


def test_get_package_version_success_missing_and_error(monkeypatch: pytest.MonkeyPatch):
    def fake_version(name: str) -> str:
        if name == "installed":
            return "1.2.3"
        raise repro.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(repro.importlib_metadata, "version", fake_version)

    assert get_package_version("installed") == "1.2.3"
    assert get_package_version("missing") is None

    with pytest.raises(ReproducibilityError, match="package_name"):
        get_package_version("")


def test_collect_library_versions_success_and_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(repro, "get_package_version", lambda name: f"v:{name}")

    versions = collect_library_versions(("numpy", "pandas"))
    assert versions == {"numpy": "v:numpy", "pandas": "v:pandas"}

    with pytest.raises(ReproducibilityError, match="не должен быть пустым"):
        collect_library_versions(())


def test_snapshot_config_and_relative_path_helpers(tmp_path):
    config = {"task": {"type": "multiclass"}}
    snapshot_path = snapshot_config(config, tmp_path / "output" / "config_snapshot.yaml")
    loaded = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert loaded == config
    assert _relative_or_absolute(snapshot_path, tmp_path.resolve()) == "output/config_snapshot.yaml"

    outside_path = Path("/tmp/outside-file.txt")
    assert _relative_or_absolute(outside_path, tmp_path.resolve()) == str(outside_path)


def test_extract_reproducibility_settings_success_and_errors():
    assert _extract_reproducibility_settings({"reproducibility": {"global_seed": 42, "checksum_algorithm": "sha256"}}) == (
        42,
        "sha256",
    )

    with pytest.raises(ReproducibilityError, match="раздел 'reproducibility'"):
        _extract_reproducibility_settings({})

    with pytest.raises(ReproducibilityError, match="reproducibility.global_seed"):
        _extract_reproducibility_settings({"reproducibility": {"global_seed": -1, "checksum_algorithm": "sha256"}})

    with pytest.raises(ReproducibilityError, match="checksum_algorithm"):
        _extract_reproducibility_settings({"reproducibility": {"global_seed": 42, "checksum_algorithm": "md5"}})


def test_build_runtime_metadata_errors(tmp_path):
    root = tmp_path.resolve()

    with pytest.raises(ReproducibilityError, match="раздел 'data'"):
        build_runtime_metadata({"reproducibility": {"global_seed": 42, "checksum_algorithm": "sha256"}}, root, "stage")

    with pytest.raises(ReproducibilityError, match="data.labeled и data.unlabeled"):
        build_runtime_metadata(
            {
                "reproducibility": {"global_seed": 42, "checksum_algorithm": "sha256"},
                "data": {"labeled": "bad", "unlabeled": "bad"},
            },
            root,
            "stage",
        )


def test_build_runtime_metadata_success(tmp_path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path.resolve()
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "labeled.csv").write_text("f1,Class\n1,0\n", encoding="utf-8")
    (data_dir / "unlabeled.csv").write_text("f1\n1\n", encoding="utf-8")

    monkeypatch.setattr(repro, "collect_library_versions", lambda packages=repro.DEFAULT_TRACKED_PACKAGES: {"numpy": "1.0"})

    metadata = build_runtime_metadata(_make_runtime_config(), root, "Stage 2: EDA")

    assert metadata["stage_name"] == "stage_2_eda"
    assert metadata["global_seed"] == 42
    assert metadata["datasets"]["labeled"]["path"] == "data/labeled.csv"
    assert metadata["datasets"]["unlabeled"]["path"] == "data/unlabeled.csv"
    assert metadata["library_versions"] == {"numpy": "1.0"}


def test_initialize_reproducibility_success(tmp_path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path.resolve()
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "labeled.csv").write_text("f1,Class\n1,0\n", encoding="utf-8")
    (data_dir / "unlabeled.csv").write_text("f1\n1\n", encoding="utf-8")

    monkeypatch.setattr(repro, "set_global_seed", lambda seed, enable_tensorflow=False: True)
    monkeypatch.setattr(repro, "collect_library_versions", lambda packages=repro.DEFAULT_TRACKED_PACKAGES: {"numpy": "1.0"})

    result = initialize_reproducibility(_make_runtime_config(), root, "Stage 4 / Baseline")
    metadata_path = root / result["metadata_path"]
    snapshot_path = root / result["config_snapshot_path"]

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result["global_seed"] == 42
    assert result["tensorflow_seed_applied"] is True
    assert snapshot_path.exists()
    assert metadata_path.exists()
    assert saved_metadata["config_snapshot_path"] == "output/artifacts/config_snapshot.yaml"
    assert saved_metadata["tensorflow_seed_applied"] is True
