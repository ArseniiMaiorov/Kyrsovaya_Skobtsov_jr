from __future__ import annotations

import json

import numpy as np
import pytest

from src.training.ga_search import (
    GENE_NAMES,
    GENE_SEARCH_SPACE,
    GASearchError,
    _group_existing_records,
    _load_existing_records,
    _build_generation_summary,
    _normalize_evaluation_output,
    _normalize_search_space,
    append_ga_log,
    create_initial_population,
    mutate_genome,
    rank_record,
    run_genetic_search,
    sample_random_genome,
    tournament_select,
    two_point_crossover,
    validate_genome,
)


def _valid_genome() -> dict[str, object]:
    return {
        "n_conv_layers": 1,
        "conv_filters": 32,
        "conv_kernel_size": 5,
        "n_gru_layers": 1,
        "gru_units": 64,
        "n_dense_layers": 1,
        "dense_units": 64,
        "optimizer": "adam",
        "activation": "relu",
    }


def _ok_record(macro_f1: float = 0.5, balanced: float = 0.6, params: int = 1234) -> dict[str, object]:
    return {
        "generation": 1,
        "individual_id": 1,
        "genome": _valid_genome(),
        "seed": 42,
        "status": "OK",
        "best_epoch": 3,
        "parameter_count": params,
        "error": None,
        "saved_model_path": None,
        "val_metrics": {
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced,
            "roc_auc_ovr_macro": None,
        },
    }


def test_normalize_search_space_success_and_error():
    normalized = _normalize_search_space()
    assert tuple(normalized.keys()) == GENE_NAMES

    broken = dict(GENE_SEARCH_SPACE)
    broken["activation"] = ()
    with pytest.raises(GASearchError, match="Пространство поиска"):
        _normalize_search_space(broken)


def test_validate_genome_success_and_errors():
    genome = validate_genome(_valid_genome())
    assert genome["optimizer"] == "adam"

    with pytest.raises(GASearchError, match="genome должен быть словарем"):
        validate_genome("bad")  # type: ignore[arg-type]

    broken = _valid_genome()
    del broken["optimizer"]
    with pytest.raises(GASearchError, match="отсутствует обязательный ген"):
        validate_genome(broken)

    broken = _valid_genome()
    broken["optimizer"] = "sgd"
    with pytest.raises(GASearchError, match="Недопустимое значение"):
        validate_genome(broken)


def test_sample_random_genome_and_initial_population():
    rng = np.random.default_rng(42)
    genome = sample_random_genome(rng)
    for gene_name in GENE_NAMES:
        assert genome[gene_name] in GENE_SEARCH_SPACE[gene_name]

    with pytest.raises(GASearchError, match="rng должен быть экземпляром"):
        sample_random_genome("bad")  # type: ignore[arg-type]

    population = create_initial_population(4, rng=np.random.default_rng(0))
    assert len(population) == 4

    with pytest.raises(GASearchError, match="population_size"):
        create_initial_population(0, rng=np.random.default_rng(0))


def test_two_point_crossover_success_and_error():
    rng = np.random.default_rng(1)
    parent_a = _valid_genome()
    parent_b = dict(_valid_genome())
    parent_b["activation"] = "tanh"
    parent_b["dense_units"] = 128

    child_a, child_b = two_point_crossover(parent_a, parent_b, rng=rng)

    assert validate_genome(child_a) == child_a
    assert validate_genome(child_b) == child_b

    with pytest.raises(GASearchError, match="rng должен быть экземпляром"):
        two_point_crossover(parent_a, parent_b, rng="bad")  # type: ignore[arg-type]


def test_mutate_genome_success_and_errors():
    rng = np.random.default_rng(2)
    mutated = mutate_genome(_valid_genome(), rng=rng, mutation_probability=1.0)
    assert validate_genome(mutated) == mutated

    single_choice_space = dict(GENE_SEARCH_SPACE)
    single_choice_space["activation"] = ("relu",)
    mutated_single = mutate_genome(
        _valid_genome(),
        rng=np.random.default_rng(3),
        mutation_probability=1.0,
        search_space=single_choice_space,
    )
    assert mutated_single["activation"] == "relu"

    with pytest.raises(GASearchError, match="mutation_probability"):
        mutate_genome(_valid_genome(), rng=np.random.default_rng(0), mutation_probability=1.1)

    with pytest.raises(GASearchError, match="rng должен быть экземпляром"):
        mutate_genome(_valid_genome(), rng="bad")  # type: ignore[arg-type]


def test_normalize_evaluation_output_success_and_errors():
    ok = _normalize_evaluation_output(
        {
            "seed": 100,
            "status": "OK",
            "metrics": {"macro_f1": 0.4, "balanced_accuracy": 0.5, "roc_auc_ovr_macro": None},
            "best_epoch": 2,
            "parameter_count": 500,
        },
        generation=1,
        individual_id=1,
        genome=_valid_genome(),
    )
    assert ok["status"] == "OK"
    assert ok["val_metrics"]["macro_f1"] == 0.4

    fail = _normalize_evaluation_output(
        {
            "seed": 101,
            "status": "FAIL",
            "error": "ошибка",
        },
        generation=1,
        individual_id=2,
        genome=_valid_genome(),
    )
    assert fail["status"] == "FAIL"
    assert fail["val_metrics"]["macro_f1"] == 0.0

    with pytest.raises(GASearchError, match="должен быть словарем"):
        _normalize_evaluation_output("bad", 1, 1, _valid_genome())  # type: ignore[arg-type]

    with pytest.raises(GASearchError, match="status должен быть"):
        _normalize_evaluation_output({}, 1, 1, _valid_genome())

    with pytest.raises(GASearchError, match="требуется metrics"):
        _normalize_evaluation_output(
            {"status": "OK", "best_epoch": 1, "parameter_count": 100},
            1,
            1,
            _valid_genome(),
        )

    with pytest.raises(GASearchError, match="требуется best_epoch"):
        _normalize_evaluation_output(
            {
                "status": "OK",
                "metrics": {"macro_f1": 0.1, "balanced_accuracy": 0.2, "roc_auc_ovr_macro": None},
                "parameter_count": 100,
            },
            1,
            1,
            _valid_genome(),
        )

    with pytest.raises(GASearchError, match="требуется parameter_count"):
        _normalize_evaluation_output(
            {
                "status": "OK",
                "metrics": {"macro_f1": 0.1, "balanced_accuracy": 0.2, "roc_auc_ovr_macro": None},
                "best_epoch": 1,
            },
            1,
            1,
            _valid_genome(),
        )

    with pytest.raises(GASearchError, match="требуется текст ошибки"):
        _normalize_evaluation_output({"status": "FAIL"}, 1, 1, _valid_genome())


def test_rank_record_success_and_errors():
    stronger = _ok_record(macro_f1=0.4444, balanced=0.7, params=1000)
    weaker = _ok_record(macro_f1=0.4441, balanced=0.6, params=900)
    assert rank_record(stronger) > rank_record(weaker)
    assert rank_record({"status": "FAIL"}) == (0.0, 0.0, float("-inf"))

    with pytest.raises(GASearchError, match="val_metrics"):
        rank_record({"status": "OK"})

    with pytest.raises(GASearchError, match="parameter_count"):
        rank_record(
            {
                "status": "OK",
                "val_metrics": {"macro_f1": 0.1, "balanced_accuracy": 0.2},
                "parameter_count": None,
            }
        )


def test_append_ga_log_and_tournament_select(tmp_path):
    path = tmp_path / "logs" / "ga.jsonl"
    append_ga_log(path, _ok_record())
    append_ga_log(path, {"status": "FAIL"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["status"] == "OK"

    records = [
        _ok_record(macro_f1=0.2, balanced=0.3, params=2000),
        _ok_record(macro_f1=0.6, balanced=0.3, params=2000),
        _ok_record(macro_f1=0.4, balanced=0.7, params=1000),
    ]
    selected = tournament_select(records, rng=np.random.default_rng(0), tournament_size=3)
    assert selected["val_metrics"]["macro_f1"] == 0.6

    with pytest.raises(GASearchError, match="не должен быть пустым"):
        tournament_select([], rng=np.random.default_rng(0))

    with pytest.raises(GASearchError, match="rng должен быть экземпляром"):
        tournament_select(records, rng="bad")  # type: ignore[arg-type]

    with pytest.raises(GASearchError, match="tournament_size"):
        tournament_select(records, rng=np.random.default_rng(0), tournament_size=0)


def test_load_existing_records_handles_missing_blank_and_invalid_jsonl(tmp_path):
    missing_path = tmp_path / "missing.jsonl"
    assert _load_existing_records(missing_path) == []

    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text('\n{"status": "OK"}\n\n{"status": "FAIL"}\n', encoding="utf-8")
    rows = _load_existing_records(valid_path)
    assert [row["status"] for row in rows] == ["OK", "FAIL"]

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text('"строка, а не объект"\n', encoding="utf-8")
    with pytest.raises(GASearchError, match="некорректную JSONL-запись"):
        _load_existing_records(invalid_path)


def test_group_existing_records_validation_errors():
    with pytest.raises(GASearchError, match="generation вне допустимого диапазона"):
        _group_existing_records(
            [{"generation": 3, "individual_id": 1}],
            population_size=2,
            generations=2,
        )

    with pytest.raises(GASearchError, match="individual_id вне допустимого диапазона"):
        _group_existing_records(
            [{"generation": 1, "individual_id": 3}],
            population_size=2,
            generations=2,
        )

    with pytest.raises(GASearchError, match="ожидались непрерывные individual_id"):
        _group_existing_records(
            [
                {"generation": 1, "individual_id": 1},
                {"generation": 1, "individual_id": 3},
            ],
            population_size=3,
            generations=2,
        )


def test_build_generation_summary_success_and_empty():
    summary = _build_generation_summary(
        1,
        [
            _ok_record(macro_f1=0.2),
            _ok_record(macro_f1=0.6),
            {"generation": 1, "individual_id": 3, "genome": _valid_genome(), "status": "FAIL", "val_metrics": {"macro_f1": 0.0, "balanced_accuracy": 0.0}},
        ],
    )
    assert summary["best_macro_f1"] == 0.6
    assert summary["ok_individuals"] == 2

    summary = _build_generation_summary(
        2,
        [
            {"generation": 2, "individual_id": 1, "genome": _valid_genome(), "status": "FAIL", "val_metrics": {"macro_f1": 0.0, "balanced_accuracy": 0.0}},
        ],
    )
    assert summary["best_macro_f1"] == 0.0
    assert summary["ok_individuals"] == 0


def test_run_genetic_search_success_and_failure_logging(tmp_path):
    calls: list[tuple[int, int]] = []
    log_path = tmp_path / "ga_population_log.jsonl"

    def evaluator(generation: int, individual_id: int, genome: dict[str, object]) -> dict[str, object]:
        calls.append((generation, individual_id))
        if generation == 1 and individual_id == 2:
            raise RuntimeError("искусственный сбой")
        return {
            "seed": generation * 100 + individual_id,
            "status": "OK",
            "metrics": {
                "macro_f1": 0.1 * generation + (0.01 * individual_id),
                "balanced_accuracy": 0.2 * generation,
                "roc_auc_ovr_macro": None,
            },
            "best_epoch": 1 + individual_id,
            "parameter_count": 1000 + individual_id,
            "saved_model_path": None,
        }

    result = run_genetic_search(
        population_size=4,
        generations=3,
        rng=np.random.default_rng(42),
        fitness_evaluator=evaluator,
        log_path=log_path,
    )

    assert result["population_size"] == 4
    assert result["generations"] == 3
    assert len(result["generation_summaries"]) == 3
    assert result["best_record"]["status"] == "OK"
    assert len(calls) == 12

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    payloads = [json.loads(line) for line in lines]
    assert any(item["status"] == "FAIL" for item in payloads)
    assert any(item["status"] == "OK" for item in payloads)


def test_run_genetic_search_resume_from_existing_log(tmp_path):
    log_path = tmp_path / "ga_population_log.jsonl"
    rng_for_log = np.random.default_rng(7)
    first_population = create_initial_population(3, rng=rng_for_log)
    first_genome = first_population[0]

    append_ga_log(
        log_path,
        {
            "generation": 1,
            "individual_id": 1,
            "genome": first_genome,
            "seed": 1001,
            "status": "OK",
            "best_epoch": 2,
            "parameter_count": 501,
            "error": None,
            "saved_model_path": None,
            "val_metrics": {
                "macro_f1": 0.11,
                "balanced_accuracy": 0.21,
                "roc_auc_ovr_macro": None,
            },
        },
    )

    calls: list[tuple[int, int]] = []

    def evaluator(generation: int, individual_id: int, genome: dict[str, object]) -> dict[str, object]:
        calls.append((generation, individual_id))
        return {
            "seed": generation * 100 + individual_id,
            "status": "OK",
            "metrics": {"macro_f1": 0.2, "balanced_accuracy": 0.3, "roc_auc_ovr_macro": None},
            "best_epoch": 1,
            "parameter_count": 100,
        }

    result = run_genetic_search(
        population_size=3,
        generations=2,
        rng=np.random.default_rng(7),
        fitness_evaluator=evaluator,
        log_path=log_path,
        resume_from_existing_log=True,
    )

    assert result["generations"] == 2
    assert calls[0] == (1, 2)
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 6


def test_run_genetic_search_resume_reuses_full_generation(tmp_path):
    log_path = tmp_path / "ga_population_log.jsonl"
    rng_for_log = np.random.default_rng(11)
    first_population = create_initial_population(2, rng=rng_for_log)

    for individual_id, genome in enumerate(first_population, start=1):
        append_ga_log(
            log_path,
            {
                "generation": 1,
                "individual_id": individual_id,
                "genome": genome,
                "seed": 2000 + individual_id,
                "status": "OK",
                "best_epoch": 2,
                "parameter_count": 700 + individual_id,
                "error": None,
                "saved_model_path": None,
                "val_metrics": {
                    "macro_f1": 0.2 + (0.01 * individual_id),
                    "balanced_accuracy": 0.3,
                    "roc_auc_ovr_macro": None,
                },
            },
        )

    calls: list[tuple[int, int]] = []

    def evaluator(generation: int, individual_id: int, genome: dict[str, object]) -> dict[str, object]:
        calls.append((generation, individual_id))
        return {
            "seed": generation * 100 + individual_id,
            "status": "OK",
            "metrics": {"macro_f1": 0.4, "balanced_accuracy": 0.5, "roc_auc_ovr_macro": None},
            "best_epoch": 1,
            "parameter_count": 100,
        }

    result = run_genetic_search(
        population_size=2,
        generations=2,
        rng=np.random.default_rng(11),
        fitness_evaluator=evaluator,
        log_path=log_path,
        resume_from_existing_log=True,
    )

    assert result["generation_summaries"][0]["generation"] == 1
    assert result["generation_summaries"][0]["ok_individuals"] == 2
    assert calls == [(2, 1), (2, 2)]
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 4


def test_run_genetic_search_validation_errors(tmp_path):
    def evaluator(generation: int, individual_id: int, genome: dict[str, object]) -> dict[str, object]:
        return {
            "seed": 1,
            "status": "OK",
            "metrics": {"macro_f1": 0.1, "balanced_accuracy": 0.2, "roc_auc_ovr_macro": None},
            "best_epoch": 1,
            "parameter_count": 100,
        }

    with pytest.raises(GASearchError, match="population_size"):
        run_genetic_search(0, 1, np.random.default_rng(0), evaluator, tmp_path / "x.jsonl")

    with pytest.raises(GASearchError, match="generations"):
        run_genetic_search(1, 0, np.random.default_rng(0), evaluator, tmp_path / "x.jsonl")

    with pytest.raises(GASearchError, match="elite_size"):
        run_genetic_search(1, 1, np.random.default_rng(0), evaluator, tmp_path / "x.jsonl", elite_size=2)

    with pytest.raises(GASearchError, match="rng должен быть экземпляром"):
        run_genetic_search(1, 1, "bad", evaluator, tmp_path / "x.jsonl")  # type: ignore[arg-type]

    with pytest.raises(GASearchError, match="fitness_evaluator"):
        run_genetic_search(1, 1, np.random.default_rng(0), "bad", tmp_path / "x.jsonl")  # type: ignore[arg-type]
