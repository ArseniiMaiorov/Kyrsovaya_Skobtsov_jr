#!/usr/bin/env python3
"""Этап 6: GA-поиск гиперпараметров для гибридной модели на `improved`."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_datasets
from src.data.preprocessing import fit_improved_preprocessor, transform_improved_labeled
from src.data.splits import build_window_split_plan, materialize_labeled_window_splits
from src.metrics.metrics import plot_multiclass_roc_curves
from src.training.ga_search import GENE_NAMES, GENE_SEARCH_SPACE, run_genetic_search
from src.training.hybrid_training import run_hybrid_experiment
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility, set_global_seed

POPULATION_LOG_PATH = PROJECT_ROOT / "output" / "logs" / "ga_population_log.jsonl"
BEST_MODEL_PATH = PROJECT_ROOT / "output" / "models" / "stage6_ga_best.keras"
BEST_GENOME_PATH = PROJECT_ROOT / "output" / "artifacts" / "stage6_best_genome.json"
SUMMARY_JSON_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage6_ga_search_summary.json"
SUMMARY_MD_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage6_ga_search_summary.md"


def _collect_improved_windows(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labeled_df, _ = load_datasets(config)

    target_col = str(config["task"]["target_col"])
    seq_cfg = config["sequence"]
    split_cfg = config["split"]
    prep_cfg = config["preprocessing"]["improved"]

    plan = build_window_split_plan(
        df=labeled_df,
        target_col=target_col,
        window_size=int(seq_cfg["T"]),
        stride=int(seq_cfg["stride"]),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        random_state=int(split_cfg["random_state"]),
    )
    train_fit_df = labeled_df.loc[plan["train_row_positions_for_fit"]].reset_index(drop=True)

    improved_preprocessor = fit_improved_preprocessor(
        train_fit_df,
        target_col=target_col,
        clip_quantiles=tuple(config["preprocessing"]["improved"]["clip_quantiles"]),
    )
    x_rows_improved, y_rows = transform_improved_labeled(labeled_df, improved_preprocessor, target_col=target_col)
    improved_splits = materialize_labeled_window_splits(x_rows_improved, y_rows, plan)

    x_train, y_train = improved_splits["train"]
    x_val, y_val = improved_splits["val"]
    return x_train, y_train, x_val, y_val


def _merge_genome_with_base_config(base_hybrid_cfg: dict[str, Any], genome: dict[str, Any], max_epochs: int) -> dict[str, Any]:
    hybrid_cfg = dict(base_hybrid_cfg)
    for gene_name in GENE_NAMES:
        hybrid_cfg[gene_name] = genome[gene_name]
    hybrid_cfg["max_epochs"] = int(max_epochs)
    return hybrid_cfg


def _metric_line(metrics: dict[str, Any]) -> str:
    roc_auc = metrics["roc_auc_ovr_macro"]
    roc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
    return (
        f"macro_f1={float(metrics['macro_f1']):.4f}, "
        f"balanced_acc={float(metrics['balanced_accuracy']):.4f}, "
        f"roc_auc={roc_text}"
    )


def _load_reference_metric(path: Path, version_name: str) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    versions = payload.get("versions")
    if not isinstance(versions, dict) or version_name not in versions:
        return None
    metrics = versions[version_name].get("val_metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("macro_f1")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _build_generation_lines(generation_summaries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in generation_summaries:
        lines.append(
            "- Поколение {generation}: best_macro_f1={best_macro_f1:.4f}, "
            "mean_macro_f1={mean_macro_f1:.4f}, ok={ok_individuals}, fail={failed_individuals}".format(**item)
        )
    return lines


def _write_best_genome_artifact(genome: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(genome, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    repro = initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage6_ga_search", use_tensorflow=True)

    labels = tuple(int(label) for label in config["task"]["labels"])
    global_seed = int(repro["global_seed"])
    rng = np.random.default_rng(global_seed)

    x_train, y_train, x_val, y_val = _collect_improved_windows(config)
    base_hybrid_cfg = dict(config["training"]["hybrid"])
    training_cfg = {
        "early_stopping_patience": int(config["training"]["early_stopping_patience"]),
        "reduce_lr_patience": int(config["training"]["reduce_lr_patience"]),
        "reduce_lr_factor": float(config["training"]["reduce_lr_factor"]),
    }
    compute_cfg = config["compute_budget"]

    def fitness_evaluator(generation: int, individual_id: int, genome: dict[str, Any]) -> dict[str, Any]:
        eval_seed = global_seed + generation * 1000 + individual_id
        set_global_seed(eval_seed, enable_tensorflow=True)
        hybrid_cfg = _merge_genome_with_base_config(
            base_hybrid_cfg=base_hybrid_cfg,
            genome=genome,
            max_epochs=int(compute_cfg["max_epochs_fitness"]),
        )
        try:
            result = run_hybrid_experiment(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                labels=labels,
                hybrid_cfg=hybrid_cfg,
                training_cfg=training_cfg,
            )
            parameter_count = int(result["model"].count_params())
            return {
                "seed": eval_seed,
                "status": "OK",
                "metrics": result["metrics"],
                "best_epoch": int(result["history"]["best_epoch"]),
                "parameter_count": parameter_count,
                "saved_model_path": None,
            }
        finally:
            tf.keras.backend.clear_session()

    ga_result = run_genetic_search(
        population_size=int(compute_cfg["population_size"]),
        generations=int(compute_cfg["generations"]),
        rng=rng,
        fitness_evaluator=fitness_evaluator,
        log_path=POPULATION_LOG_PATH,
        search_space=GENE_SEARCH_SPACE,
        tournament_size=3,
        mutation_probability=0.2,
        elite_size=1,
        resume_from_existing_log=POPULATION_LOG_PATH.exists() and not SUMMARY_JSON_PATH.exists(),
    )

    best_record = ga_result["best_record"]
    best_genome = dict(best_record["genome"])
    _write_best_genome_artifact(best_genome, BEST_GENOME_PATH)

    final_seed = global_seed + 999_999
    set_global_seed(final_seed, enable_tensorflow=True)
    final_hybrid_cfg = _merge_genome_with_base_config(
        base_hybrid_cfg=base_hybrid_cfg,
        genome=best_genome,
        max_epochs=int(compute_cfg["max_epochs_final"]),
    )
    final_result = run_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=labels,
        hybrid_cfg=final_hybrid_cfg,
        training_cfg=training_cfg,
        artifacts_dir=PROJECT_ROOT / "output" / "artifacts" / "stage6_final_best",
    )
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_result["model"].save(BEST_MODEL_PATH)
    final_model_params = int(final_result["model"].count_params())
    roc_info = plot_multiclass_roc_curves(
        y_true=y_val,
        y_proba=final_result["y_proba"],
        labels=labels,
        save_path=PROJECT_ROOT / "reports" / "figures" / "stage6_ga_best_roc.png",
    )
    tf.keras.backend.clear_session()

    baseline_reference = _load_reference_metric(
        PROJECT_ROOT / "reports" / "experiments" / "stage4_baseline_summary.json",
        version_name="improved",
    )
    default_hybrid_reference = _load_reference_metric(
        PROJECT_ROOT / "reports" / "experiments" / "stage5_hybrid_summary.json",
        version_name="improved",
    )
    target_gain = float(compute_cfg["target_macro_f1_gain_vs_baseline"])

    final_macro = float(final_result["metrics"]["macro_f1"])
    gain_vs_baseline = None if baseline_reference is None else final_macro - baseline_reference
    gain_vs_default_hybrid = None if default_hybrid_reference is None else final_macro - default_hybrid_reference

    report: dict[str, Any] = {
        "dataset_version": "improved",
        "model_name": "hybrid_cnn_gru_dense",
        "fitness_protocol": "fitness считается только на официальном val; rolling-диагностика не участвует в отборе",
        "ga_config": {
            "population_size": int(compute_cfg["population_size"]),
            "generations": int(compute_cfg["generations"]),
            "max_epochs_fitness": int(compute_cfg["max_epochs_fitness"]),
            "max_epochs_final": int(compute_cfg["max_epochs_final"]),
            "tournament_size": 3,
            "mutation_probability": 0.2,
            "elite_size": 1,
        },
        "search_space": {key: list(values) for key, values in GENE_SEARCH_SPACE.items()},
        "log_path": str(POPULATION_LOG_PATH.relative_to(PROJECT_ROOT)),
        "best_genome_artifact": str(BEST_GENOME_PATH.relative_to(PROJECT_ROOT)),
        "best_fitness_record": best_record,
        "generation_summaries": ga_result["generation_summaries"],
        "final_best_run": {
            "seed": final_seed,
            "hybrid_config": final_hybrid_cfg,
            "history": final_result["history"],
            "history_artifacts": {
                key: str(Path(value).relative_to(PROJECT_ROOT)) for key, value in final_result["artifacts"].items()
            },
            "val_metrics": final_result["metrics"],
            "parameter_count": final_model_params,
            "model_path": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "roc_curve": {
                "path": str(Path(roc_info["path"]).relative_to(PROJECT_ROOT)),
                "present_labels": list(roc_info["present_labels"]),
                "per_class_auc": dict(roc_info["per_class_auc"]),
            },
        },
        "references": {
            "baseline_improved_val_macro_f1": baseline_reference,
            "default_hybrid_improved_val_macro_f1": default_hybrid_reference,
            "target_macro_f1_gain_vs_baseline": target_gain,
            "final_gain_vs_baseline": gain_vs_baseline,
            "final_gain_vs_default_hybrid": gain_vs_default_hybrid,
            "target_reached": None if gain_vs_baseline is None else bool(gain_vs_baseline >= target_gain),
        },
    }

    out_dir = PROJECT_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = SUMMARY_JSON_PATH
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    md_lines = [
        "# Этап 6: GA-поиск гиперпараметров",
        "",
        "- Датасет для поиска: `improved`.",
        "- Модель для поиска: `hybrid_cnn_gru_dense`.",
        "- Fitness считается только на официальном `val`.",
        "- Rolling-диагностика остаётся отдельным вспомогательным контуром и не участвует в отборе.",
        f"- Лог популяции: `{report['log_path']}`",
        f"- Артефакт лучшего генома: `{report['best_genome_artifact']}`",
        "",
        "## Конфиг GA",
        f"- population_size: {report['ga_config']['population_size']}",
        f"- generations: {report['ga_config']['generations']}",
        f"- max_epochs_fitness: {report['ga_config']['max_epochs_fitness']}",
        f"- max_epochs_final: {report['ga_config']['max_epochs_final']}",
        f"- tournament_size: {report['ga_config']['tournament_size']}",
        f"- mutation_probability: {report['ga_config']['mutation_probability']}",
        f"- elite_size: {report['ga_config']['elite_size']}",
        "",
        "## Лучший индивид по fitness",
        f"- Поколение: {best_record['generation']}",
        f"- Индивид: {best_record['individual_id']}",
        f"- Статус: {best_record['status']}",
        f"- Метрики fitness-val: {_metric_line(best_record['val_metrics'])}",
        f"- Лучшая эпоха fitness: {best_record['best_epoch']}",
        f"- Число параметров: {best_record['parameter_count']}",
        "- Геном:",
    ]
    for gene_name in GENE_NAMES:
        md_lines.append(f"  - {gene_name}: {best_genome[gene_name]}")

    md_lines.extend(
        [
            "",
            "## Поколения",
        ]
    )
    md_lines.extend(_build_generation_lines(report["generation_summaries"]))

    final_metrics = report["final_best_run"]["val_metrics"]
    md_lines.extend(
        [
            "",
            "## Финальное дообучение лучшего генома",
            f"- Seed: {report['final_best_run']['seed']}",
            f"- Метрики final-val: {_metric_line(final_metrics)}",
            f"- Лучшая эпоха: {report['final_best_run']['history']['best_epoch']}",
            f"- Эпох выполнено: {report['final_best_run']['history']['epochs_ran']}",
            f"- Число параметров: {report['final_best_run']['parameter_count']}",
            f"- Сохранённая модель: `{report['final_best_run']['model_path']}`",
            f"- ROC-кривые: `{report['final_best_run']['roc_curve']['path']}`",
        ]
    )
    for artifact_name, artifact_path in report["final_best_run"]["history_artifacts"].items():
        md_lines.append(f"- Артефакт {artifact_name}: `{artifact_path}`")
    if "roc_auc_note" in final_metrics:
        md_lines.append(f"- Примечание по ROC AUC: {final_metrics['roc_auc_note']}")

    refs = report["references"]
    md_lines.extend(["", "## Сравнение с предыдущими этапами"])
    if refs["baseline_improved_val_macro_f1"] is not None:
        md_lines.append(f"- Baseline improved (этап 4): {refs['baseline_improved_val_macro_f1']:.4f}")
        md_lines.append(f"- Прирост относительно baseline: {refs['final_gain_vs_baseline']:.4f}")
        md_lines.append(
            f"- Цель по ТЗ (+{refs['target_macro_f1_gain_vs_baseline']:.2f}) "
            f"{'выполнена' if refs['target_reached'] else 'не выполнена'}."
        )
    if refs["default_hybrid_improved_val_macro_f1"] is not None:
        md_lines.append(f"- Базовый hybrid improved (этап 5): {refs['default_hybrid_improved_val_macro_f1']:.4f}")
        md_lines.append(f"- Прирост относительно базового hybrid: {refs['final_gain_vs_default_hybrid']:.4f}")

    md_path = SUMMARY_MD_PATH
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Сохранено: {json_path}")
    print(f"Сохранено: {md_path}")


if __name__ == "__main__":
    main()
