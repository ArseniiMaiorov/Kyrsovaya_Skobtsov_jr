from __future__ import annotations

from copy import deepcopy

import pytest

from src.utils.config import ConfigError, load_config, validate_config


@pytest.fixture
def valid_config() -> dict:
    return {
        "task": {"type": "multiclass", "labels": [0, 1, 2], "target_col": "Class"},
        "reproducibility": {"global_seed": 42, "checksum_algorithm": "sha256"},
        "sequence": {"mode": "time_windows", "T": 128, "stride": 32, "overlap": 0.75},
        "data": {
            "expected_feature_count": 49,
            "drop_all_nan_rows": True,
            "null_tokens": ["", "NA", -999],
            "labeled": {
                "format": "xls",
                "path": "data/MKA_TMI_labels.xls",
                "sheet_name": "Лист1",
            },
            "unlabeled": {
                "format": "csv",
                "path": "data/MKA_04.2015_unlabeled.csv",
                "sep": ",",
                "encoding": "utf-8",
            },
        },
        "split": {
            "method": "time_order_windows",
            "random_state": 42,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "preprocessing": {
            "raw": {"impute_strategy": "median", "scaling": "none"},
            "improved": {
                "impute_strategy": "median",
                "scaler": "robust",
                "clip_quantiles": [0.01, 0.99],
            },
        },
        "compute_budget": {
            "population_size": 12,
            "generations": 8,
            "max_epochs_fitness": 10,
            "max_epochs_final": 50,
            "target_macro_f1_gain_vs_baseline": 0.03,
        },
        "augmentation": {
            "enabled": True,
            "aug_factor": 3,
            "methods": ["noise", "scale"],
            "noise_std": 0.01,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "shift_min": -5,
            "shift_max": 5,
        },
        "training": {
            "early_stopping_patience": 5,
            "reduce_lr_patience": 3,
            "reduce_lr_factor": 0.5,
            "hybrid": {
                "n_conv_layers": 1,
                "conv_filters": 64,
                "conv_kernel_size": 5,
                "n_gru_layers": 1,
                "gru_units": 128,
                "n_dense_layers": 1,
                "dense_units": 128,
                "activation": "relu",
                "optimizer": "adam",
                "loss": "sparse_categorical_crossentropy",
                "rnn_type": "gru",
                "use_attention": False,
                "attention_units": 128,
                "batch_size": 8,
                "max_epochs": 30,
                "conv_dropout": 0.2,
                "dense_dropout": 0.3,
                "l2_dense": 0.0001,
            },
            "autoencoder": {
                "batch_size": 8,
                "pretrain_max_epochs": 50,
                "pretrain_val_ratio": 0.15,
                "use_stage6_best_genome": True,
            },
        },
    }


def test_load_config_success(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
task:
  type: multiclass
  labels: [0, 1, 2]
  target_col: Class
reproducibility:
  global_seed: 42
  checksum_algorithm: sha256
sequence:
  mode: time_windows
  T: 128
  stride: 32
  overlap: 0.75
data:
  expected_feature_count: 49
  drop_all_nan_rows: true
  null_tokens: ["", "NA", -999]
  labeled:
    format: xls
    path: data/MKA_TMI_labels.xls
    sheet_name: Лист1
  unlabeled:
    format: csv
    path: data/MKA_04.2015_unlabeled.csv
    sep: ","
    encoding: utf-8
split:
  method: time_order_windows
  random_state: 42
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
preprocessing:
  raw:
    impute_strategy: median
    scaling: none
  improved:
    impute_strategy: median
    scaler: robust
    clip_quantiles: [0.01, 0.99]
compute_budget:
  population_size: 12
  generations: 8
  max_epochs_fitness: 10
  max_epochs_final: 50
  target_macro_f1_gain_vs_baseline: 0.03
augmentation:
  enabled: true
  aug_factor: 3
  methods: [noise, scale]
  noise_std: 0.01
  scale_min: 0.9
  scale_max: 1.1
  shift_min: -5
  shift_max: 5
training:
  early_stopping_patience: 5
  reduce_lr_patience: 3
  reduce_lr_factor: 0.5
  hybrid:
    n_conv_layers: 1
    conv_filters: 64
    conv_kernel_size: 5
    n_gru_layers: 1
    gru_units: 128
    n_dense_layers: 1
    dense_units: 128
    activation: relu
    optimizer: adam
    loss: sparse_categorical_crossentropy
    rnn_type: gru
    use_attention: false
    attention_units: 128
    batch_size: 8
    max_epochs: 30
    conv_dropout: 0.2
    dense_dropout: 0.3
    l2_dense: 0.0001
  autoencoder:
    batch_size: 8
    pretrain_max_epochs: 50
    pretrain_val_ratio: 0.15
    use_stage6_best_genome: true
        """.strip(),
        encoding="utf-8",
    )

    loaded = load_config(config_path)
    assert loaded["split"]["method"] == "time_order_windows"


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


def test_validate_config_reproducibility_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["reproducibility"] = "bad"
    with pytest.raises(ConfigError, match="reproducibility должен быть словарем"):
        validate_config(broken)


def test_validate_config_reproducibility_global_seed_error(valid_config):
    broken = deepcopy(valid_config)
    broken["reproducibility"]["global_seed"] = -1
    with pytest.raises(ConfigError, match="reproducibility.global_seed"):
        validate_config(broken)


def test_validate_config_reproducibility_checksum_algorithm_error(valid_config):
    broken = deepcopy(valid_config)
    broken["reproducibility"]["checksum_algorithm"] = "md5"
    with pytest.raises(ConfigError, match="reproducibility.checksum_algorithm"):
        validate_config(broken)


def test_validate_config_augmentation_errors(valid_config):
    broken = deepcopy(valid_config)
    broken["augmentation"] = "bad"
    with pytest.raises(ConfigError, match="augmentation должен быть словарем"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["enabled"] = "yes"
    with pytest.raises(ConfigError, match="augmentation.enabled"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["methods"] = []
    with pytest.raises(ConfigError, match="augmentation.methods"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["methods"] = ["flip"]
    with pytest.raises(ConfigError, match="augmentation.methods"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["scale_min"] = 2.0
    broken["augmentation"]["scale_max"] = 1.0
    with pytest.raises(ConfigError, match="scale_min"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["noise_std"] = "bad"
    with pytest.raises(ConfigError, match="augmentation.noise_std"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["noise_std"] = -0.1
    with pytest.raises(ConfigError, match="noise_std"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["scale_min"] = 0.0
    with pytest.raises(ConfigError, match="scale_min и augmentation.scale_max"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["augmentation"]["shift_min"] = 10
    broken["augmentation"]["shift_max"] = 5
    with pytest.raises(ConfigError, match="shift_min"):
        validate_config(broken)


def test_validate_config_autoencoder_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["autoencoder"] = "bad"
    with pytest.raises(ConfigError, match="training.autoencoder должен быть словарем"):
        validate_config(broken)


def test_validate_config_autoencoder_field_errors(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["autoencoder"]["batch_size"] = 0
    with pytest.raises(ConfigError, match="training.autoencoder.batch_size"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["training"]["autoencoder"]["pretrain_val_ratio"] = 1.0
    with pytest.raises(ConfigError, match="training.autoencoder.pretrain_val_ratio"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["training"]["autoencoder"]["use_stage6_best_genome"] = "yes"
    with pytest.raises(ConfigError, match="training.autoencoder.use_stage6_best_genome"):
        validate_config(broken)


def test_validate_config_hybrid_new_fields_errors(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"]["rnn_type"] = "rnn"
    with pytest.raises(ConfigError, match="training.hybrid.rnn_type"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["training"]["hybrid"]["use_attention"] = "yes"
    with pytest.raises(ConfigError, match="training.hybrid.use_attention"):
        validate_config(broken)

    broken = deepcopy(valid_config)
    broken["training"]["hybrid"]["attention_units"] = 0
    with pytest.raises(ConfigError, match="training.hybrid.attention_units"):
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


def test_validate_config_data_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["data"] = "bad"
    with pytest.raises(ConfigError, match="data должен быть словарем"):
        validate_config(broken)


def test_validate_config_data_expected_feature_count_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["expected_feature_count"] = 0
    with pytest.raises(ConfigError, match="expected_feature_count"):
        validate_config(broken)


def test_validate_config_data_drop_all_nan_rows_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["drop_all_nan_rows"] = "yes"
    with pytest.raises(ConfigError, match="drop_all_nan_rows"):
        validate_config(broken)


def test_validate_config_data_null_tokens_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["null_tokens"] = []
    with pytest.raises(ConfigError, match="data.null_tokens"):
        validate_config(broken)


def test_validate_config_data_labeled_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["labeled"] = "bad"
    with pytest.raises(ConfigError, match="data.labeled должен быть словарем"):
        validate_config(broken)


def test_validate_config_data_unlabeled_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["unlabeled"] = "bad"
    with pytest.raises(ConfigError, match="data.unlabeled должен быть словарем"):
        validate_config(broken)


def test_validate_config_data_unknown_format(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["labeled"]["format"] = "xlsx"
    with pytest.raises(ConfigError, match="data.labeled.format"):
        validate_config(broken)


def test_validate_config_data_labeled_path_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["labeled"]["path"] = ""
    with pytest.raises(ConfigError, match="data.labeled.path"):
        validate_config(broken)


def test_validate_config_data_unlabeled_sep_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["unlabeled"]["sep"] = ""
    with pytest.raises(ConfigError, match="data.unlabeled.sep"):
        validate_config(broken)


def test_validate_config_data_unlabeled_encoding_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["unlabeled"]["encoding"] = ""
    with pytest.raises(ConfigError, match="data.unlabeled.encoding"):
        validate_config(broken)


def test_validate_config_data_sheet_name_error(valid_config):
    broken = deepcopy(valid_config)
    broken["data"]["labeled"]["sheet_name"] = []
    with pytest.raises(ConfigError, match="data.labeled.sheet_name"):
        validate_config(broken)


def test_validate_config_split_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["split"] = "bad"
    with pytest.raises(ConfigError, match="split должен быть словарем"):
        validate_config(broken)


def test_validate_config_split_method_error(valid_config):
    broken = deepcopy(valid_config)
    broken["split"]["method"] = "stratified_random"
    with pytest.raises(ConfigError, match="split.method"):
        validate_config(broken)


def test_validate_config_split_random_state_error(valid_config):
    broken = deepcopy(valid_config)
    broken["split"]["random_state"] = -1
    with pytest.raises(ConfigError, match="split.random_state"):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("train_ratio", 0, "split.train_ratio"),
        ("val_ratio", 0, "split.val_ratio"),
        ("test_ratio", 0, "split.test_ratio"),
    ],
)
def test_validate_config_split_probability_errors(valid_config, field, value, error):
    broken = deepcopy(valid_config)
    broken["split"][field] = value
    with pytest.raises(ConfigError, match=error):
        validate_config(broken)


def test_validate_config_split_sum_error(valid_config):
    broken = deepcopy(valid_config)
    broken["split"]["train_ratio"] = 0.8
    broken["split"]["val_ratio"] = 0.15
    broken["split"]["test_ratio"] = 0.1
    with pytest.raises(ConfigError, match="Сумма split.train_ratio"):
        validate_config(broken)


def test_validate_config_preprocessing_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"] = "bad"
    with pytest.raises(ConfigError, match="preprocessing должен быть словарем"):
        validate_config(broken)


def test_validate_config_preprocessing_raw_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["raw"] = "bad"
    with pytest.raises(ConfigError, match="preprocessing.raw"):
        validate_config(broken)


def test_validate_config_preprocessing_improved_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["improved"] = "bad"
    with pytest.raises(ConfigError, match="preprocessing.improved"):
        validate_config(broken)


def test_validate_config_preprocessing_raw_impute_error(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["raw"]["impute_strategy"] = "mean"
    with pytest.raises(ConfigError, match="preprocessing.raw.impute_strategy"):
        validate_config(broken)


def test_validate_config_preprocessing_raw_scaling_error(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["raw"]["scaling"] = "standard"
    with pytest.raises(ConfigError, match="preprocessing.raw.scaling"):
        validate_config(broken)


def test_validate_config_preprocessing_impute_strategy_error(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["improved"]["impute_strategy"] = "mean"
    with pytest.raises(ConfigError, match="impute_strategy"):
        validate_config(broken)


def test_validate_config_preprocessing_scaler_error(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["improved"]["scaler"] = "standard"
    with pytest.raises(ConfigError, match="scaler"):
        validate_config(broken)


def test_validate_config_preprocessing_clip_quantiles_len_error(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["improved"]["clip_quantiles"] = [0.1]
    with pytest.raises(ConfigError, match="clip_quantiles"):
        validate_config(broken)


def test_validate_config_preprocessing_clip_quantiles_type_error(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["improved"]["clip_quantiles"] = ["0.1", 0.9]
    with pytest.raises(ConfigError, match="содержать числа"):
        validate_config(broken)


def test_validate_config_preprocessing_clip_quantiles_order_error(valid_config):
    broken = deepcopy(valid_config)
    broken["preprocessing"]["improved"]["clip_quantiles"] = [0.9, 0.1]
    with pytest.raises(ConfigError, match="0 <= lower < upper <= 1"):
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


def test_validate_config_training_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["training"] = "bad"
    with pytest.raises(ConfigError, match="training должен быть словарем"):
        validate_config(broken)


def test_validate_config_training_hybrid_not_mapping(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"] = "bad"
    with pytest.raises(ConfigError, match="training.hybrid"):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("n_conv_layers", 0, "training.hybrid.n_conv_layers"),
        ("conv_filters", 0, "training.hybrid.conv_filters"),
        ("conv_kernel_size", 0, "training.hybrid.conv_kernel_size"),
        ("n_gru_layers", 0, "training.hybrid.n_gru_layers"),
        ("gru_units", 0, "training.hybrid.gru_units"),
        ("n_dense_layers", 0, "training.hybrid.n_dense_layers"),
        ("dense_units", 0, "training.hybrid.dense_units"),
        ("batch_size", 0, "training.hybrid.batch_size"),
        ("max_epochs", 0, "training.hybrid.max_epochs"),
    ],
)
def test_validate_config_training_hybrid_positive_int_errors(valid_config, field, value, error):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"][field] = value
    with pytest.raises(ConfigError, match=error):
        validate_config(broken)


def test_validate_config_training_hybrid_activation_error(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"]["activation"] = "sigmoid"
    with pytest.raises(ConfigError, match="training.hybrid.activation"):
        validate_config(broken)


def test_validate_config_training_hybrid_optimizer_error(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"]["optimizer"] = "sgd"
    with pytest.raises(ConfigError, match="training.hybrid.optimizer"):
        validate_config(broken)


def test_validate_config_training_hybrid_loss_error(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"]["loss"] = "mse"
    with pytest.raises(ConfigError, match="training.hybrid.loss"):
        validate_config(broken)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conv_dropout", 1.0),
        ("dense_dropout", 1.0),
    ],
)
def test_validate_config_training_hybrid_dropout_errors(valid_config, field, value):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"][field] = value
    with pytest.raises(ConfigError, match=f"training.hybrid.{field}"):
        validate_config(broken)


def test_validate_config_training_hybrid_l2_dense_error(valid_config):
    broken = deepcopy(valid_config)
    broken["training"]["hybrid"]["l2_dense"] = -0.1
    with pytest.raises(ConfigError, match="training.hybrid.l2_dense"):
        validate_config(broken)
