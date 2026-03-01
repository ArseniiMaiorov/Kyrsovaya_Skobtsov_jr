"""Утилиты для загрузки и валидации конфигурации проекта."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_TOP_LEVEL_SECTIONS = (
    "task",
    "reproducibility",
    "sequence",
    "data",
    "split",
    "preprocessing",
    "compute_budget",
    "training",
)

SUPPORTED_TABULAR_FORMATS = {"csv", "xls"}
SUPPORTED_CHECKSUM_ALGORITHMS = {"sha256"}
SUPPORTED_HYBRID_ACTIVATIONS = {"relu", "elu", "tanh"}
SUPPORTED_HYBRID_OPTIMIZERS = {"adam", "rmsprop", "nadam"}
SUPPORTED_HYBRID_LOSSES = {"categorical_crossentropy", "sparse_categorical_crossentropy"}
SUPPORTED_HYBRID_RNN_TYPES = {"gru", "lstm", "bi_gru", "bi_lstm"}
SUPPORTED_AUGMENTATION_METHODS = {"noise", "scale", "shift"}


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


def _require_probability(value: Any, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ConfigError(f"{field_name} должен быть числом в диапазоне (0, 1)")


def _validate_data_source(source: Mapping[str, Any] | Any, field_name: str) -> None:
    if not isinstance(source, Mapping):
        raise ConfigError(f"{field_name} должен быть словарем")

    source_format = source.get("format")
    if source_format not in SUPPORTED_TABULAR_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_TABULAR_FORMATS))
        raise ConfigError(f"{field_name}.format должен быть одним из: {supported}")

    path = source.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ConfigError(f"{field_name}.path должен быть непустой строкой")

    if source_format == "csv":
        sep = source.get("sep")
        encoding = source.get("encoding")
        if not isinstance(sep, str) or not sep:
            raise ConfigError(f"{field_name}.sep должен быть непустой строкой")
        if not isinstance(encoding, str) or not encoding:
            raise ConfigError(f"{field_name}.encoding должен быть непустой строкой")

    if source_format == "xls":
        sheet_name = source.get("sheet_name")
        if sheet_name is not None and not isinstance(sheet_name, (str, int)):
            raise ConfigError(f"{field_name}.sheet_name должен быть строкой или целым числом")


def _validate_data_section(data_cfg: Mapping[str, Any] | Any) -> None:
    if not isinstance(data_cfg, Mapping):
        raise ConfigError("data должен быть словарем")

    expected_feature_count = data_cfg.get("expected_feature_count")
    if not isinstance(expected_feature_count, int) or expected_feature_count <= 0:
        raise ConfigError("data.expected_feature_count должен быть положительным целым числом")

    drop_all_nan_rows = data_cfg.get("drop_all_nan_rows")
    if not isinstance(drop_all_nan_rows, bool):
        raise ConfigError("data.drop_all_nan_rows должен быть булевым значением")

    null_tokens = data_cfg.get("null_tokens")
    if not isinstance(null_tokens, list) or not null_tokens:
        raise ConfigError("data.null_tokens должен быть непустым списком")

    _validate_data_source(data_cfg.get("labeled"), "data.labeled")
    _validate_data_source(data_cfg.get("unlabeled"), "data.unlabeled")


def _validate_split_section(split_cfg: Mapping[str, Any] | Any) -> None:
    if not isinstance(split_cfg, Mapping):
        raise ConfigError("split должен быть словарем")

    if split_cfg.get("method") != "time_order_windows":
        raise ConfigError("split.method должен быть равен 'time_order_windows'")

    random_state = split_cfg.get("random_state")
    if not isinstance(random_state, int) or random_state < 0:
        raise ConfigError("split.random_state должен быть неотрицательным целым числом")

    train_ratio = split_cfg.get("train_ratio")
    val_ratio = split_cfg.get("val_ratio")
    test_ratio = split_cfg.get("test_ratio")
    _require_probability(train_ratio, "split.train_ratio")
    _require_probability(val_ratio, "split.val_ratio")
    _require_probability(test_ratio, "split.test_ratio")

    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total - 1.0) > 1e-9:
        raise ConfigError("Сумма split.train_ratio, split.val_ratio и split.test_ratio должна быть равна 1")


def _validate_preprocessing_section(prep_cfg: Mapping[str, Any] | Any) -> None:
    if not isinstance(prep_cfg, Mapping):
        raise ConfigError("preprocessing должен быть словарем")

    raw_cfg = prep_cfg.get("raw")
    improved_cfg = prep_cfg.get("improved")

    if not isinstance(raw_cfg, Mapping):
        raise ConfigError("preprocessing.raw должен быть словарем")
    if not isinstance(improved_cfg, Mapping):
        raise ConfigError("preprocessing.improved должен быть словарем")

    if raw_cfg.get("impute_strategy") != "median":
        raise ConfigError("preprocessing.raw.impute_strategy должен быть равен 'median'")
    if raw_cfg.get("scaling") != "none":
        raise ConfigError("preprocessing.raw.scaling должен быть равен 'none'")

    if improved_cfg.get("impute_strategy") != "median":
        raise ConfigError("preprocessing.improved.impute_strategy должен быть равен 'median'")
    if improved_cfg.get("scaler") != "robust":
        raise ConfigError("preprocessing.improved.scaler должен быть равен 'robust'")

    clip_quantiles = improved_cfg.get("clip_quantiles")
    if not isinstance(clip_quantiles, list) or len(clip_quantiles) != 2:
        raise ConfigError("preprocessing.improved.clip_quantiles должен содержать 2 числа")

    lower_q, upper_q = clip_quantiles
    if not isinstance(lower_q, (int, float)) or not isinstance(upper_q, (int, float)):
        raise ConfigError("preprocessing.improved.clip_quantiles должен содержать числа")
    if not (0 <= lower_q < upper_q <= 1):
        raise ConfigError("preprocessing.improved.clip_quantiles должен удовлетворять 0 <= lower < upper <= 1")


def _validate_reproducibility_section(repro_cfg: Mapping[str, Any] | Any) -> None:
    if not isinstance(repro_cfg, Mapping):
        raise ConfigError("reproducibility должен быть словарем")

    global_seed = repro_cfg.get("global_seed")
    if not isinstance(global_seed, int) or global_seed < 0:
        raise ConfigError("reproducibility.global_seed должен быть неотрицательным целым числом")

    checksum_algorithm = repro_cfg.get("checksum_algorithm")
    if checksum_algorithm not in SUPPORTED_CHECKSUM_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_CHECKSUM_ALGORITHMS))
        raise ConfigError(f"reproducibility.checksum_algorithm должен быть одним из: {supported}")


def _validate_hybrid_training_section(hybrid_cfg: Mapping[str, Any] | Any) -> None:
    if not isinstance(hybrid_cfg, Mapping):
        raise ConfigError("training.hybrid должен быть словарем")

    for field_name in (
        "n_conv_layers",
        "conv_filters",
        "conv_kernel_size",
        "n_gru_layers",
        "gru_units",
        "n_dense_layers",
        "dense_units",
        "batch_size",
        "max_epochs",
    ):
        _require_positive_int(hybrid_cfg.get(field_name), f"training.hybrid.{field_name}")

    activation = hybrid_cfg.get("activation")
    if activation not in SUPPORTED_HYBRID_ACTIVATIONS:
        supported = ", ".join(sorted(SUPPORTED_HYBRID_ACTIVATIONS))
        raise ConfigError(f"training.hybrid.activation должен быть одним из: {supported}")

    optimizer = hybrid_cfg.get("optimizer")
    if optimizer not in SUPPORTED_HYBRID_OPTIMIZERS:
        supported = ", ".join(sorted(SUPPORTED_HYBRID_OPTIMIZERS))
        raise ConfigError(f"training.hybrid.optimizer должен быть одним из: {supported}")

    loss = hybrid_cfg.get("loss")
    if loss not in SUPPORTED_HYBRID_LOSSES:
        supported = ", ".join(sorted(SUPPORTED_HYBRID_LOSSES))
        raise ConfigError(f"training.hybrid.loss должен быть одним из: {supported}")

    for field_name in ("conv_dropout", "dense_dropout"):
        value = hybrid_cfg.get(field_name)
        if not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ConfigError(f"training.hybrid.{field_name} должен быть в диапазоне [0, 1)")

    l2_dense = hybrid_cfg.get("l2_dense")
    if not isinstance(l2_dense, (int, float)) or l2_dense < 0:
        raise ConfigError("training.hybrid.l2_dense должен быть числом >= 0")

    rnn_type = hybrid_cfg.get("rnn_type", "gru")
    if rnn_type not in SUPPORTED_HYBRID_RNN_TYPES:
        supported = ", ".join(sorted(SUPPORTED_HYBRID_RNN_TYPES))
        raise ConfigError(f"training.hybrid.rnn_type должен быть одним из: {supported}")

    use_attention = hybrid_cfg.get("use_attention", False)
    if not isinstance(use_attention, bool):
        raise ConfigError("training.hybrid.use_attention должен быть булевым значением")

    attention_units = hybrid_cfg.get("attention_units", hybrid_cfg.get("gru_units"))
    if not isinstance(attention_units, int) or attention_units <= 0:
        raise ConfigError("training.hybrid.attention_units должен быть положительным целым числом")


def _validate_autoencoder_training_section(autoencoder_cfg: Mapping[str, Any] | Any) -> None:
    if not isinstance(autoencoder_cfg, Mapping):
        raise ConfigError("training.autoencoder должен быть словарем")

    _require_positive_int(autoencoder_cfg.get("batch_size"), "training.autoencoder.batch_size")
    _require_positive_int(autoencoder_cfg.get("pretrain_max_epochs"), "training.autoencoder.pretrain_max_epochs")
    _require_probability(autoencoder_cfg.get("pretrain_val_ratio"), "training.autoencoder.pretrain_val_ratio")

    use_stage6_best_genome = autoencoder_cfg.get("use_stage6_best_genome")
    if not isinstance(use_stage6_best_genome, bool):
        raise ConfigError("training.autoencoder.use_stage6_best_genome должен быть булевым значением")


def _validate_augmentation_section(augmentation_cfg: Mapping[str, Any] | Any) -> None:
    if not isinstance(augmentation_cfg, Mapping):
        raise ConfigError("augmentation должен быть словарем")

    enabled = augmentation_cfg.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigError("augmentation.enabled должен быть булевым значением")

    _require_positive_int(augmentation_cfg.get("aug_factor"), "augmentation.aug_factor")

    methods = augmentation_cfg.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ConfigError("augmentation.methods должен быть непустым списком")
    for method in methods:
        if method not in SUPPORTED_AUGMENTATION_METHODS:
            supported = ", ".join(sorted(SUPPORTED_AUGMENTATION_METHODS))
            raise ConfigError(f"augmentation.methods должен содержать только: {supported}")

    numeric_fields = (
        "noise_std",
        "scale_min",
        "scale_max",
        "shift_min",
        "shift_max",
    )
    for field_name in numeric_fields:
        value = augmentation_cfg.get(field_name)
        if not isinstance(value, (int, float)):
            raise ConfigError(f"augmentation.{field_name} должен быть числом")

    if float(augmentation_cfg["noise_std"]) < 0:
        raise ConfigError("augmentation.noise_std должен быть >= 0")
    if float(augmentation_cfg["scale_min"]) <= 0 or float(augmentation_cfg["scale_max"]) <= 0:
        raise ConfigError("augmentation.scale_min и augmentation.scale_max должны быть > 0")
    if float(augmentation_cfg["scale_min"]) > float(augmentation_cfg["scale_max"]):
        raise ConfigError("augmentation.scale_min не должен быть больше augmentation.scale_max")
    if int(augmentation_cfg["shift_min"]) > int(augmentation_cfg["shift_max"]):
        raise ConfigError("augmentation.shift_min не должен быть больше augmentation.shift_max")


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

    _validate_reproducibility_section(config["reproducibility"])

    sequence = config["sequence"]
    if sequence.get("mode") != "time_windows":
        raise ConfigError("sequence.mode должен быть равен 'time_windows'")
    _require_positive_int(sequence.get("T"), "sequence.T")
    _require_positive_int(sequence.get("stride"), "sequence.stride")
    overlap = sequence.get("overlap")
    if not isinstance(overlap, (int, float)) or not 0 <= overlap < 1:
        raise ConfigError("sequence.overlap должен быть в диапазоне [0, 1)")

    _validate_data_section(config.get("data"))
    _validate_split_section(config.get("split"))
    _validate_preprocessing_section(config.get("preprocessing"))

    compute_budget = config["compute_budget"]
    _require_positive_int(compute_budget.get("population_size"), "compute_budget.population_size")
    _require_positive_int(compute_budget.get("generations"), "compute_budget.generations")
    _require_positive_int(compute_budget.get("max_epochs_fitness"), "compute_budget.max_epochs_fitness")
    _require_positive_int(compute_budget.get("max_epochs_final"), "compute_budget.max_epochs_final")

    gain = compute_budget.get("target_macro_f1_gain_vs_baseline")
    if not isinstance(gain, (int, float)) or gain <= 0:
        raise ConfigError("compute_budget.target_macro_f1_gain_vs_baseline должен быть больше 0")

    training = config["training"]
    if not isinstance(training, Mapping):
        raise ConfigError("training должен быть словарем")

    _require_positive_int(training.get("early_stopping_patience"), "training.early_stopping_patience")
    _require_positive_int(training.get("reduce_lr_patience"), "training.reduce_lr_patience")

    factor = training.get("reduce_lr_factor")
    if not isinstance(factor, (int, float)) or not 0 < factor < 1:
        raise ConfigError("training.reduce_lr_factor должен быть в диапазоне (0, 1)")

    _validate_hybrid_training_section(training.get("hybrid"))
    _validate_autoencoder_training_section(training.get("autoencoder"))

    if "augmentation" in config:
        _validate_augmentation_section(config["augmentation"])
