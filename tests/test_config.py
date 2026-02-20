from __future__ import annotations

from copy import deepcopy

import pytest

from src.utils.config import ConfigError, load_config, validate_config


@pytest.fixture
def valid_config() -> dict:
    return {
        "task": {"type": "multiclass", "labels": [0, 1, 2], "target_col": "Class"},
        "sequence": {"mode": "time_windows", "T": 128, "stride": 32, "overlap": 0.75},
        "data": {
            "format": "csv",
            "path": "data/tmi.csv",
            "sep": ",",
            "encoding": "utf-8",
            "null_tokens": ["", "NA", -999],
        },
        "compute_budget": {
            "population_size": 12,
            "generations": 8,
            "max_epochs_fitness": 10,
            "max_epochs_final": 50,
            "target_macro_f1_gain_vs_baseline": 0.03,
        },
        "training": {
            "early_stopping_patience": 5,
            "reduce_lr_patience": 3,
            "reduce_lr_factor": 0.5,
        },
    }


def test_load_config_success(tmp_path, valid_config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
task:
  type: multiclass
  labels: [0, 1, 2]
  target_col: Class
sequence:
  mode: time_windows
  T: 128
  stride: 32
  overlap: 0.75
data:
  format: csv
  path: data/tmi.csv
  sep: ","
  encoding: utf-8
  null_tokens: ["", "NA", -999]
compute_budget:
  population_size: 12
  generations: 8
  max_epochs_fitness: 10
  max_epochs_final: 50
  target_macro_f1_gain_vs_baseline: 0.03
training:
  early_stopping_patience: 5
  reduce_lr_patience: 3
  reduce_lr_factor: 0.5
        """.strip(),
        encoding="utf-8",
    )

    loaded = load_config(config_path)
    assert loaded["task"]["labels"] == [0, 1, 2]


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("missing.yaml")


def test_load_config_root_not_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- просто\n- список\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Корневой элемент конфигурации должен быть словарем"):
        load_config(config_path)


def test_validate_config_missing_sections(valid_config):
    broken = deepcopy(valid_config)
    del broken["training"]

    with pytest.raises(ConfigError, match="Отсутствуют обязательные разделы верхнего уровня"):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("type", "binary", "task.type"),
        ("labels", [0, 1], "task.labels"),
        ("target_col", "", "task.target_col"),
    ],
)
def test_validate_config_task_errors(valid_config, field, value, error):
    broken = deepcopy(valid_config)
    broken["task"][field] = value

    with pytest.raises(ConfigError, match=error):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("mode", "pseudo", "sequence.mode"),
        ("T", 0, "sequence.T"),
        ("stride", -1, "sequence.stride"),
        ("overlap", 1.0, "sequence.overlap"),
    ],
)
def test_validate_config_sequence_errors(valid_config, field, value, error):
    broken = deepcopy(valid_config)
    broken["sequence"][field] = value

    with pytest.raises(ConfigError, match=error):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("format", "parquet", "data.format"),
        ("path", "", "data.path"),
        ("sep", "", "data.sep"),
        ("encoding", "", "data.encoding"),
    ],
)
def test_validate_config_data_text_field_errors(valid_config, field, value, error):
    broken = deepcopy(valid_config)
    broken["data"][field] = value

    with pytest.raises(ConfigError, match=error):
        validate_config(broken)


def test_validate_config_data_null_tokens_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["null_tokens"] = []

    with pytest.raises(ConfigError, match="data.null_tokens"):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("population_size", 0, "compute_budget.population_size"),
        ("generations", 0, "compute_budget.generations"),
        ("max_epochs_fitness", 0, "compute_budget.max_epochs_fitness"),
        ("max_epochs_final", 0, "compute_budget.max_epochs_final"),
    ],
)
def test_validate_config_compute_positive_int_errors(valid_config, field, value, error):
    broken = deepcopy(valid_config)
    broken["compute_budget"][field] = value

    with pytest.raises(ConfigError, match=error):
        validate_config(broken)


def test_validate_config_compute_gain_error(valid_config):
    broken = deepcopy(valid_config)
    broken["compute_budget"]["target_macro_f1_gain_vs_baseline"] = 0

    with pytest.raises(ConfigError, match="target_macro_f1_gain_vs_baseline"):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("early_stopping_patience", 0, "training.early_stopping_patience"),
        ("reduce_lr_patience", 0, "training.reduce_lr_patience"),
    ],
)
def test_validate_config_training_positive_int_errors(valid_config, field, value, error):
    broken = deepcopy(valid_config)
    broken["training"][field] = value

    with pytest.raises(ConfigError, match=error):
        validate_config(broken)


def test_validate_config_training_factor_error(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["reduce_lr_factor"] = 1.0

    with pytest.raises(ConfigError, match="training.reduce_lr_factor"):
        validate_config(broken)
