from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.io import DataIOError, load_datasets, load_labeled_dataset, load_unlabeled_dataset
from src.data.validation import DataValidationError


def _make_config(labeled: dict, unlabeled: dict, expected_feature_count: int = 2, drop_all_nan_rows: bool = True) -> dict:
    return {
        "task": {"type": "multiclass", "labels": [0, 1, 2], "target_col": "Class"},
        "data": {
            "expected_feature_count": expected_feature_count,
            "drop_all_nan_rows": drop_all_nan_rows,
            "null_tokens": ["", "NA", -999],
            "labeled": labeled,
            "unlabeled": unlabeled,
        },
    }


def test_load_labeled_dataset_csv_success(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    labeled_path.write_text("f1,f2,Class\n1,2,0\n3,4,1\n5,6,2\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    df = load_labeled_dataset(config)
    assert list(df.columns) == ["f1", "f2", "Class"]
    assert len(df) == 3


def test_load_labeled_dataset_drops_all_nan_rows(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    labeled_path.write_text("f1,f2,Class\n1,2,0\n,,\n3,4,1\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    df = load_labeled_dataset(config)
    assert len(df) == 2


def test_load_labeled_dataset_missing_target(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    labeled_path.write_text("f1,f2\n1,2\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataValidationError, match="Отсутствует целевая колонка"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_invalid_labels(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    labeled_path.write_text("f1,f2,Class\n1,2,0\n3,4,9\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataValidationError, match="Обнаружены неизвестные метки"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_non_numeric_feature_error(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    labeled_path.write_text("f1,f2,Class\nabc,2,0\n3,4,1\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataValidationError, match="нечисловые признаки"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_expected_feature_count_error(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    labeled_path.write_text("f1,f2,Class\n1,2,0\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        expected_feature_count=3,
    )

    with pytest.raises(DataValidationError, match="Ожидалось 3 признаков"):
        load_labeled_dataset(config)


def test_load_unlabeled_dataset_has_target_error(tmp_path: Path):
    path = tmp_path / "unlabeled.csv"
    path.write_text("f1,f2,Class\n1,2,0\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataValidationError, match="не должно быть колонки"):
        load_unlabeled_dataset(config, expected_features=("f1", "f2"))


def test_load_unlabeled_dataset_non_numeric_error(tmp_path: Path):
    path = tmp_path / "unlabeled.csv"
    path.write_text("f1,f2\n1,abc\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataValidationError, match="нечисловые признаки"):
        load_unlabeled_dataset(config, expected_features=("f1", "f2"))


def test_load_unlabeled_dataset_feature_mismatch_error(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    unlabeled_path = tmp_path / "unlabeled.csv"

    labeled_path.write_text("f1,f2,Class\n1,2,0\n3,4,1\n", encoding="utf-8")
    unlabeled_path.write_text("f1,f3\n1,2\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(unlabeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataValidationError, match="Несовпадение признаков"):
        load_datasets(config)


def test_load_unlabeled_dataset_success_and_load_datasets_success(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    unlabeled_path = tmp_path / "unlabeled.csv"

    labeled_path.write_text("f1,f2,Class\n1,2,0\n3,4,1\n", encoding="utf-8")
    unlabeled_path.write_text("f1,f2\n5,6\n7,8\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(unlabeled_path), "sep": ",", "encoding": "utf-8"},
    )

    unlabeled_df = load_unlabeled_dataset(config, expected_features=("f1", "f2"))
    labeled_df, unlabeled_df_pair = load_datasets(config)

    assert len(unlabeled_df) == 2
    assert len(labeled_df) == 2
    assert len(unlabeled_df_pair) == 2


def test_load_unlabeled_dataset_expected_feature_count_error(tmp_path: Path):
    unlabeled_path = tmp_path / "unlabeled.csv"
    unlabeled_path.write_text("f1,f2\n1,2\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(unlabeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(unlabeled_path), "sep": ",", "encoding": "utf-8"},
        expected_feature_count=3,
    )

    with pytest.raises(DataValidationError, match="Ожидалось 3 признаков"):
        load_unlabeled_dataset(config, expected_features=("f1", "f2"))


def test_load_labeled_dataset_file_not_found(tmp_path: Path):
    config = _make_config(
        labeled={"format": "csv", "path": str(tmp_path / "missing.csv"), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(tmp_path / "missing.csv"), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(FileNotFoundError, match="Файл датасета не найден"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_empty_dataset_error(tmp_path: Path):
    labeled_path = tmp_path / "empty.csv"
    labeled_path.write_text("", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataIOError, match="Датасет пуст"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_header_only_error(tmp_path: Path):
    labeled_path = tmp_path / "header_only.csv"
    labeled_path.write_text("f1,f2,Class\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataIOError, match="Датасет пуст"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_drop_all_nan_to_empty_error(tmp_path: Path):
    labeled_path = tmp_path / "empty.csv"
    labeled_path.write_text("f1,f2,Class\n,,\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataIOError, match="После удаления полностью пустых строк"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_unsupported_format_error(tmp_path: Path):
    labeled_path = tmp_path / "labeled.csv"
    labeled_path.write_text("f1,f2,Class\n1,2,0\n", encoding="utf-8")

    config = _make_config(
        labeled={"format": "parquet", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
        unlabeled={"format": "csv", "path": str(labeled_path), "sep": ",", "encoding": "utf-8"},
    )

    with pytest.raises(DataIOError, match="Неподдерживаемый формат датасета"):
        load_labeled_dataset(config)


def test_load_labeled_dataset_xls_success_with_monkeypatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    xls_path = tmp_path / "labeled.xls"
    xls_path.write_bytes(b"dummy")

    called = {}

    def fake_read_excel(path, sheet_name, engine, na_values):
        called["path"] = str(path)
        called["sheet_name"] = sheet_name
        called["engine"] = engine
        called["na_values"] = list(na_values)
        return pd.DataFrame({"f1": [1.0], "f2": [2.0], "Class": [0]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    config = _make_config(
        labeled={"format": "xls", "path": str(xls_path), "sheet_name": "Лист1"},
        unlabeled={"format": "csv", "path": str(xls_path), "sep": ",", "encoding": "utf-8"},
    )

    df = load_labeled_dataset(config)

    assert len(df) == 1
    assert called["engine"] == "xlrd"
    assert called["sheet_name"] == "Лист1"
