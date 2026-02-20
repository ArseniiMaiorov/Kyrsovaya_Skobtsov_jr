from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import (
    DataContract,
    DataValidationError,
    normalize_missing_tokens,
    validate_dataframe_schema,
    validate_label_values,
)


def test_normalize_missing_tokens_replaces_values():
    df = pd.DataFrame({"a": ["1", "NA", -999], "Class": [0, 1, 2]})

    normalized = normalize_missing_tokens(df, ["NA", -999])

    assert normalized.isna().sum().to_dict()["a"] == 2


def test_validate_label_values_empty_error():
    with pytest.raises(DataValidationError, match="Целевые метки пусты"):
        validate_label_values([])


def test_validate_label_values_unknown_error():
    with pytest.raises(DataValidationError, match="Обнаружены неизвестные метки"):
        validate_label_values([0, 1, 5])


def test_validate_label_values_success():
    validate_label_values([0, 1, 2, 0])


def test_validate_dataframe_schema_missing_target():
    df = pd.DataFrame({"f1": [1.0, 2.0]})
    contract = DataContract(target_col="Class")

    with pytest.raises(DataValidationError, match="Отсутствует целевая колонка"):
        validate_dataframe_schema(df, contract)


def test_validate_dataframe_schema_missing_features():
    df = pd.DataFrame({"f1": [1.0, 2.0], "Class": [0, 1]})
    contract = DataContract(target_col="Class", required_features=("f1", "f2"))

    with pytest.raises(DataValidationError, match="Отсутствуют обязательные признаки"):
        validate_dataframe_schema(df, contract)


def test_validate_dataframe_schema_invalid_labels():
    df = pd.DataFrame({"f1": [1.0, 2.0], "Class": [0, 4]})
    contract = DataContract(target_col="Class", required_features=("f1",))

    with pytest.raises(DataValidationError, match="Обнаружены неизвестные метки"):
        validate_dataframe_schema(df, contract)


def test_validate_dataframe_schema_success():
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "Class": [0, 2]})
    contract = DataContract(target_col="Class", required_features=("f1", "f2"))

    validate_dataframe_schema(df, contract)
