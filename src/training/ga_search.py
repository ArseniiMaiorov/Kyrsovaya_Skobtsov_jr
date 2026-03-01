"""Генетический алгоритм для подбора гиперпараметров гибридной модели."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

GENE_NAMES = (
    "n_conv_layers",
    "conv_filters",
    "conv_kernel_size",
    "n_gru_layers",
    "gru_units",
    "n_dense_layers",
    "dense_units",
    "optimizer",
    "activation",
)

GENE_SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "n_conv_layers": (1, 2, 3),
    "conv_filters": (16, 32, 64, 128),
    "conv_kernel_size": (3, 5, 7, 9, 11),
    "n_gru_layers": (1, 2),
    "gru_units": (32, 64, 128, 256),
    "n_dense_layers": (1, 2),
    "dense_units": (32, 64, 128, 256),
    "optimizer": ("adam", "rmsprop", "nadam"),
    "activation": ("relu", "elu", "tanh"),
}


class GASearchError(ValueError):
    """Исключение для ошибок конфигурации и выполнения генетического алгоритма."""


FitnessEvaluator = Callable[[int, int, dict[str, Any]], Mapping[str, Any]]


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise GASearchError(f"{field_name} должен быть положительным целым числом")
    return int(value)


def _require_probability(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise GASearchError(f"{field_name} должен быть числом в диапазоне [0, 1]")
    return float(value)


def _normalize_search_space(search_space: Mapping[str, tuple[Any, ...]] | None = None) -> dict[str, tuple[Any, ...]]:
    space = GENE_SEARCH_SPACE if search_space is None else dict(search_space)
    for gene_name in GENE_NAMES:
        values = space.get(gene_name)
        if not isinstance(values, tuple) or not values:
            raise GASearchError(f"Пространство поиска для гена '{gene_name}' должно быть непустым кортежем")
    return space


def validate_genome(
    genome: Mapping[str, Any],
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Проверяет геном на допустимые значения и возвращает нормализованную копию."""
    if not isinstance(genome, Mapping):
        raise GASearchError("genome должен быть словарем")

    space = _normalize_search_space(search_space)
    normalized: dict[str, Any] = {}
    for gene_name in GENE_NAMES:
        if gene_name not in genome:
            raise GASearchError(f"В геноме отсутствует обязательный ген '{gene_name}'")
        value = genome[gene_name]
        if value not in space[gene_name]:
            raise GASearchError(
                f"Недопустимое значение для гена '{gene_name}': {value}. "
                f"Допустимо: {list(space[gene_name])}"
            )
        normalized[gene_name] = value
    return normalized


def sample_random_genome(
    rng: np.random.Generator,
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Сэмплирует случайный допустимый геном равномерно по пространству поиска."""
    if not isinstance(rng, np.random.Generator):
        raise GASearchError("rng должен быть экземпляром numpy.random.Generator")

    space = _normalize_search_space(search_space)
    genome = {
        gene_name: space[gene_name][int(rng.integers(0, len(space[gene_name])))]
        for gene_name in GENE_NAMES
    }
    return validate_genome(genome, search_space=space)


def create_initial_population(
    population_size: int,
    rng: np.random.Generator,
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Создаёт начальную популяцию случайных индивидов."""
    size = _require_positive_int(population_size, "population_size")
    return [sample_random_genome(rng, search_space=search_space) for _ in range(size)]


def _genome_to_list(genome: Mapping[str, Any]) -> list[Any]:
    return [genome[gene_name] for gene_name in GENE_NAMES]


def _list_to_genome(values: list[Any]) -> dict[str, Any]:
    return {gene_name: values[idx] for idx, gene_name in enumerate(GENE_NAMES)}


def two_point_crossover(
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    rng: np.random.Generator,
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Выполняет two-point crossover по 9 генам."""
    if not isinstance(rng, np.random.Generator):
        raise GASearchError("rng должен быть экземпляром numpy.random.Generator")

    space = _normalize_search_space(search_space)
    parent_a_norm = validate_genome(parent_a, search_space=space)
    parent_b_norm = validate_genome(parent_b, search_space=space)

    points = sorted(int(x) for x in rng.choice(np.arange(1, len(GENE_NAMES)), size=2, replace=False))
    left, right = points

    values_a = _genome_to_list(parent_a_norm)
    values_b = _genome_to_list(parent_b_norm)

    child_a = values_a[:left] + values_b[left:right] + values_a[right:]
    child_b = values_b[:left] + values_a[left:right] + values_b[right:]

    return (
        validate_genome(_list_to_genome(child_a), search_space=space),
        validate_genome(_list_to_genome(child_b), search_space=space),
    )


def mutate_genome(
    genome: Mapping[str, Any],
    rng: np.random.Generator,
    mutation_probability: float = 0.2,
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Мутирует каждый ген независимо с вероятностью `p`."""
    if not isinstance(rng, np.random.Generator):
        raise GASearchError("rng должен быть экземпляром numpy.random.Generator")

    probability = _require_probability(mutation_probability, "mutation_probability")
    space = _normalize_search_space(search_space)
    mutated = dict(validate_genome(genome, search_space=space))

    for gene_name in GENE_NAMES:
        if float(rng.random()) > probability:
            continue
        choices = list(space[gene_name])
        if len(choices) == 1:
            continue
        current = mutated[gene_name]
        alternatives = [value for value in choices if value != current]
        mutated[gene_name] = alternatives[int(rng.integers(0, len(alternatives)))]

    return validate_genome(mutated, search_space=space)


def _extract_metrics(record: Mapping[str, Any]) -> tuple[float, float, float]:
    if record.get("status") != "OK":
        return 0.0, 0.0, float("inf")

    metrics = record.get("val_metrics")
    if not isinstance(metrics, Mapping):
        raise GASearchError("Для успешной записи val_metrics должен быть словарем")

    macro_f1 = float(metrics["macro_f1"])
    balanced = float(metrics["balanced_accuracy"])
    params = record.get("parameter_count")
    if not isinstance(params, int) or params <= 0:
        raise GASearchError("Для успешной записи parameter_count должен быть положительным целым числом")
    return macro_f1, balanced, float(params)


def rank_record(record: Mapping[str, Any]) -> tuple[float, float, float]:
    """Возвращает кортеж ранжирования: fitness, balanced_acc, компактность."""
    macro_f1, balanced, params = _extract_metrics(record)
    return (round(macro_f1, 3), balanced, -params)


def _json_ready_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "roc_auc_ovr_macro": None if metrics.get("roc_auc_ovr_macro") is None else float(metrics["roc_auc_ovr_macro"]),
    }


def _normalize_evaluation_output(
    evaluation: Mapping[str, Any],
    generation: int,
    individual_id: int,
    genome: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(evaluation, Mapping):
        raise GASearchError("Результат fitness_evaluator должен быть словарем")

    status = evaluation.get("status")
    if status not in {"OK", "FAIL"}:
        raise GASearchError("status должен быть равен 'OK' или 'FAIL'")

    record: dict[str, Any] = {
        "generation": int(generation),
        "individual_id": int(individual_id),
        "genome": dict(genome),
        "seed": int(evaluation.get("seed", 0)),
        "status": status,
        "best_epoch": None if evaluation.get("best_epoch") is None else int(evaluation.get("best_epoch")),
        "parameter_count": None if evaluation.get("parameter_count") is None else int(evaluation.get("parameter_count")),
        "error": None if evaluation.get("error") is None else str(evaluation.get("error")),
        "saved_model_path": None if evaluation.get("saved_model_path") is None else str(evaluation.get("saved_model_path")),
        "val_metrics": _json_ready_metrics(evaluation.get("metrics")),
    }

    if status == "OK":
        if record["val_metrics"] is None:
            raise GASearchError("Для успешной записи требуется metrics")
        if record["best_epoch"] is None:
            raise GASearchError("Для успешной записи требуется best_epoch")
        if record["parameter_count"] is None:
            raise GASearchError("Для успешной записи требуется parameter_count")
        record["error"] = None
    else:
        if not record["error"]:
            raise GASearchError("Для неуспешной записи требуется текст ошибки")
        record["val_metrics"] = {
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "roc_auc_ovr_macro": None,
        }
        record["best_epoch"] = None
        record["parameter_count"] = None

    return record


def append_ga_log(log_path: str | Path, record: Mapping[str, Any]) -> None:
    """Добавляет одну JSONL-запись в лог популяции GA."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _load_existing_records(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise GASearchError("Лог GA содержит некорректную JSONL-запись")
        rows.append(payload)
    return rows


def _group_existing_records(
    records: list[dict[str, Any]],
    population_size: int,
    generations: int,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        generation = int(record["generation"])
        individual_id = int(record["individual_id"])
        if not 1 <= generation <= generations:
            raise GASearchError("Лог GA содержит запись с generation вне допустимого диапазона")
        if not 1 <= individual_id <= population_size:
            raise GASearchError("Лог GA содержит запись с individual_id вне допустимого диапазона")
        grouped.setdefault(generation, []).append(record)

    for generation, items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["individual_id"]))
        expected_ids = list(range(1, len(ordered) + 1))
        actual_ids = [int(item["individual_id"]) for item in ordered]
        if actual_ids != expected_ids:
            raise GASearchError(
                f"Лог GA поврежден: в поколении {generation} ожидались непрерывные individual_id от 1, получено {actual_ids}"
            )
        grouped[generation] = ordered

    return grouped


def tournament_select(
    population_records: list[dict[str, Any]],
    rng: np.random.Generator,
    tournament_size: int = 3,
) -> dict[str, Any]:
    """Отбор индивида турнирным методом."""
    if not population_records:
        raise GASearchError("population_records не должен быть пустым")
    if not isinstance(rng, np.random.Generator):
        raise GASearchError("rng должен быть экземпляром numpy.random.Generator")

    size = _require_positive_int(tournament_size, "tournament_size")
    sample_size = min(size, len(population_records))
    candidate_indexes = rng.choice(np.arange(len(population_records)), size=sample_size, replace=False)
    candidates = [population_records[int(idx)] for idx in candidate_indexes]
    return max(candidates, key=rank_record)


def _evaluate_population(
    population: list[dict[str, Any]],
    generation: int,
    fitness_evaluator: FitnessEvaluator,
    log_path: str | Path,
    start_individual_id: int = 1,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for idx, genome in enumerate(population, start=start_individual_id):
        try:
            raw_result = fitness_evaluator(generation, idx, dict(genome))
            record = _normalize_evaluation_output(raw_result, generation=generation, individual_id=idx, genome=genome)
        except Exception as exc:
            record = {
                "generation": int(generation),
                "individual_id": int(idx),
                "genome": dict(genome),
                "seed": 0,
                "status": "FAIL",
                "best_epoch": None,
                "parameter_count": None,
                "error": str(exc),
                "saved_model_path": None,
                "val_metrics": {
                    "macro_f1": 0.0,
                    "balanced_accuracy": 0.0,
                    "roc_auc_ovr_macro": None,
                },
            }
        append_ga_log(log_path, record)
        records.append(record)

    return records


def _build_generation_summary(
    generation: int,
    population_records: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [record for record in population_records if record["status"] == "OK"]
    best_record = max(population_records, key=rank_record)

    summary: dict[str, Any] = {
        "generation": int(generation),
        "best_individual_id": int(best_record["individual_id"]),
        "best_genome": dict(best_record["genome"]),
        "best_status": str(best_record["status"]),
        "ok_individuals": len(successful),
        "failed_individuals": len(population_records) - len(successful),
    }

    if successful:
        macro_values = np.asarray([float(record["val_metrics"]["macro_f1"]) for record in successful], dtype=np.float64)
        summary["best_macro_f1"] = float(max(macro_values))
        summary["mean_macro_f1"] = float(np.mean(macro_values))
        summary["min_macro_f1"] = float(np.min(macro_values))
    else:
        summary["best_macro_f1"] = 0.0
        summary["mean_macro_f1"] = 0.0
        summary["min_macro_f1"] = 0.0

    return summary


def run_genetic_search(
    population_size: int,
    generations: int,
    rng: np.random.Generator,
    fitness_evaluator: FitnessEvaluator,
    log_path: str | Path,
    search_space: Mapping[str, tuple[Any, ...]] | None = None,
    tournament_size: int = 3,
    mutation_probability: float = 0.2,
    elite_size: int = 1,
    resume_from_existing_log: bool = False,
) -> dict[str, Any]:
    """Полный цикл GA с логированием всех индивидов популяции."""
    pop_size = _require_positive_int(population_size, "population_size")
    gen_count = _require_positive_int(generations, "generations")
    elite = _require_positive_int(elite_size, "elite_size")
    if elite > pop_size:
        raise GASearchError("elite_size не может быть больше population_size")
    if not isinstance(rng, np.random.Generator):
        raise GASearchError("rng должен быть экземпляром numpy.random.Generator")
    if not callable(fitness_evaluator):
        raise GASearchError("fitness_evaluator должен быть вызываемым объектом")

    _require_probability(mutation_probability, "mutation_probability")
    normalized_space = _normalize_search_space(search_space)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_records = _load_existing_records(path) if resume_from_existing_log else []
    if not resume_from_existing_log:
        path.write_text("", encoding="utf-8")
    grouped_existing = _group_existing_records(existing_records, population_size=pop_size, generations=gen_count)

    population = create_initial_population(pop_size, rng=rng, search_space=normalized_space)
    generation_summaries: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for generation in range(1, gen_count + 1):
        existing_for_generation = list(grouped_existing.get(generation, []))
        if len(existing_for_generation) == pop_size:
            population_records = existing_for_generation
        else:
            remaining_population = population[len(existing_for_generation) :]
            evaluated_tail = _evaluate_population(
                population=remaining_population,
                generation=generation,
                fitness_evaluator=fitness_evaluator,
                log_path=path,
                start_individual_id=len(existing_for_generation) + 1,
            )
            population_records = existing_for_generation + evaluated_tail

        generation_summary = _build_generation_summary(generation, population_records)
        generation_summaries.append(generation_summary)

        generation_best = max(population_records, key=rank_record)
        if best_record is None or rank_record(generation_best) > rank_record(best_record):
            best_record = dict(generation_best)

        if generation == gen_count:
            break

        ranked = sorted(population_records, key=rank_record, reverse=True)
        next_population = [dict(record["genome"]) for record in ranked[:elite]]

        while len(next_population) < pop_size:
            parent_a = tournament_select(ranked, rng=rng, tournament_size=tournament_size)
            parent_b = tournament_select(ranked, rng=rng, tournament_size=tournament_size)
            child_a, child_b = two_point_crossover(
                parent_a["genome"],
                parent_b["genome"],
                rng=rng,
                search_space=normalized_space,
            )
            next_population.append(
                mutate_genome(child_a, rng=rng, mutation_probability=mutation_probability, search_space=normalized_space)
            )
            if len(next_population) < pop_size:
                next_population.append(
                    mutate_genome(child_b, rng=rng, mutation_probability=mutation_probability, search_space=normalized_space)
                )

        population = next_population

    assert best_record is not None

    return {
        "population_size": pop_size,
        "generations": gen_count,
        "tournament_size": int(tournament_size),
        "mutation_probability": float(mutation_probability),
        "elite_size": elite,
        "best_record": best_record,
        "generation_summaries": generation_summaries,
        "log_path": str(path),
    }
