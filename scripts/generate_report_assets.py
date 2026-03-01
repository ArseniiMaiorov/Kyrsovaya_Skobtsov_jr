#!/usr/bin/env python3
"""Генерирует иллюстрации и скриншоты для курсового отчёта."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
APPENDIX_SNIPPET_PATH = REPORT_DIR / "generated_code_appendices.tex"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape_tex_text(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _prepare_output_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _plot_class_distribution(eda: dict) -> None:
    labels = list(eda["class_distribution"].keys())
    values = list(eda["class_distribution"].values())

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=["#2A6F97", "#E07A5F", "#3D405B"])
    ax.set_title("Распределение классов в размеченном наборе")
    ax.set_xlabel("Класс состояния")
    ax.set_ylabel("Число наблюдений")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(values) * 0.01,
            str(value),
            ha="center",
            va="bottom",
        )
    _save(fig, "class_distribution.png")


def _plot_model_comparison(stage4: dict, stage5: dict, stage6: dict) -> None:
    names = [
        "Baseline\nraw",
        "Baseline\nimproved",
        "Hybrid\nraw",
        "Hybrid\nimproved",
        "GA best\nimproved",
    ]
    macro_f1 = [
        stage4["versions"]["raw"]["val_metrics"]["macro_f1"],
        stage4["versions"]["improved"]["val_metrics"]["macro_f1"],
        stage5["versions"]["raw"]["val_metrics"]["macro_f1"],
        stage5["versions"]["improved"]["val_metrics"]["macro_f1"],
        stage6["final_best_run"]["val_metrics"]["macro_f1"],
    ]
    bal_acc = [
        stage4["versions"]["raw"]["val_metrics"]["balanced_accuracy"],
        stage4["versions"]["improved"]["val_metrics"]["balanced_accuracy"],
        stage5["versions"]["raw"]["val_metrics"]["balanced_accuracy"],
        stage5["versions"]["improved"]["val_metrics"]["balanced_accuracy"],
        stage6["final_best_run"]["val_metrics"]["balanced_accuracy"],
    ]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x - width / 2, macro_f1, width, label="macro-F1", color="#2A9D8F")
    ax.bar(x + width / 2, bal_acc, width, label="balanced accuracy", color="#E9C46A")
    ax.axhline(0.03 + stage4["versions"]["improved"]["val_metrics"]["macro_f1"], color="#D62828", linestyle="--", linewidth=1.0, label="Цель baseline+0.03")
    ax.set_title("Сравнение качества моделей на официальной validation-выборке")
    ax.set_ylabel("Значение метрики")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    _save(fig, "model_comparison.png")


def _plot_outlier_features(eda: dict) -> None:
    top = eda["top_outlier_features"][:10]
    names = [item[0] for item in top]
    shares = [item[1] for item in top]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    y = np.arange(len(names))
    ax.barh(y, shares, color="#E76F51")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Доля выбросов по IQR")
    ax.set_title("Признаки с наибольшей долей выбросов")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(shares):
        ax.text(value + 0.003, idx, f"{value:.3f}", va="center")
    _save(fig, "top_outliers.png")


def _plot_ga_progress(stage6: dict) -> None:
    gens = [item["generation"] for item in stage6["generation_summaries"]]
    best = [item["best_macro_f1"] for item in stage6["generation_summaries"]]
    mean = [item["mean_macro_f1"] for item in stage6["generation_summaries"]]
    low = [item["min_macro_f1"] for item in stage6["generation_summaries"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(gens, best, marker="o", color="#264653", label="Лучший в поколении")
    ax.plot(gens, mean, marker="s", color="#2A9D8F", label="Средний fitness")
    ax.plot(gens, low, marker="^", color="#E76F51", label="Минимальный fitness")
    ax.set_title("Динамика GA-поиска по поколениям")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("macro-F1 на val")
    ax.set_xticks(gens)
    ax.set_ylim(0.1, 0.75)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, "ga_progress.png")


def _plot_seed_stability(stage8: dict) -> None:
    rows = stage8["stability_check"]["results"]
    seeds = [str(item["seed"]) for item in rows]
    macro_f1 = [item["val_metrics"]["macro_f1"] for item in rows]
    baseline = stage8["stability_check"]["baseline_improved_val_macro_f1"]
    colors = ["#2A9D8F" if v >= baseline else "#D62828" for v in macro_f1]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(seeds, macro_f1, color=colors)
    ax.axhline(baseline, color="#1D3557", linestyle="--", label="Baseline improved")
    ax.set_title("Проверка устойчивости лучшей конфигурации по seed")
    ax.set_xlabel("Значение seed")
    ax.set_ylabel("macro-F1 на val")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, macro_f1):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.4f}", ha="center", va="bottom")
    ax.legend(loc="upper right")
    _save(fig, "seed_stability.png")


def _plot_test_confusion(stage8: dict) -> None:
    matrix = np.array(stage8["final_test_eval"]["test_metrics"]["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Матрица ошибок на финальном test")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#102A43")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "test_confusion_matrix.png")


def _plot_split_scheme(stage3: dict) -> None:
    row_counts = [1875, 402, 402]
    labels = ["Train rows", "Validation rows", "Test rows"]
    colors = ["#2A9D8F", "#E9C46A", "#E76F51"]

    fig, ax = plt.subplots(figsize=(10.2, 2.8))
    start = 0
    total = sum(row_counts)
    for label, value, color in zip(labels, row_counts, colors):
        ax.barh([0], [value], left=[start], height=0.55, color=color)
        ax.text(start + value / 2, 0, f"{label}\n{value} строк", ha="center", va="center", fontsize=10)
        start += value

    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("Положение во временном ряду")
    ax.set_title("Схема временного разбиения исходной последовательности")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    _save(fig, "split_scheme.png")


def _draw_box(ax, xy, width, height, text, fc):
    rect = plt.Rectangle(xy, width, height, facecolor=fc, edgecolor="#1F2937", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1F2937"))


def _plot_hybrid_scheme() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")
    _draw_box(ax, (0.3, 1.0), 2.0, 1.0, "Окно 128×49", "#D8F3DC")
    _draw_box(ax, (2.9, 1.0), 2.2, 1.0, "Conv1D + BN\n+ ReLU", "#95D5B2")
    _draw_box(ax, (5.8, 1.0), 2.0, 1.0, "Dropout", "#B7E4C7")
    _draw_box(ax, (8.3, 1.0), 2.2, 1.0, "GRU", "#FFD166")
    _draw_box(ax, (11.1, 1.0), 2.1, 1.0, "Dense + Softmax", "#F4A261")
    _arrow(ax, 2.3, 1.5, 2.9, 1.5)
    _arrow(ax, 5.1, 1.5, 5.8, 1.5)
    _arrow(ax, 7.8, 1.5, 8.3, 1.5)
    _arrow(ax, 10.5, 1.5, 11.1, 1.5)
    ax.set_title("Схема гибридной модели классификации")
    _save(fig, "hybrid_scheme.png")


def _plot_autoencoder_scheme() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3)
    ax.axis("off")
    _draw_box(ax, (0.3, 1.0), 2.2, 1.0, "Неразмеченное\nокно 128×49", "#D8F3DC")
    _draw_box(ax, (3.0, 1.0), 2.3, 1.0, "Encoder\nConv1D + GRU", "#74C69D")
    _draw_box(ax, (6.0, 1.0), 2.3, 1.0, "Латентное\nпредставление", "#52B788")
    _draw_box(ax, (9.0, 1.0), 2.3, 1.0, "Decoder\nRepeat + GRU", "#FFD166")
    _draw_box(ax, (12.0, 1.0), 2.4, 1.0, "Восстановленное\nокно", "#F4A261")
    _arrow(ax, 2.5, 1.5, 3.0, 1.5)
    _arrow(ax, 5.3, 1.5, 6.0, 1.5)
    _arrow(ax, 8.3, 1.5, 9.0, 1.5)
    _arrow(ax, 11.3, 1.5, 12.0, 1.5)
    ax.set_title("Схема реконструкционного предобучения автоэнкодера")
    _save(fig, "autoencoder_scheme.png")


def _make_terminal_image(name: str, lines: list[str], width: float = 10.5) -> None:
    text = "\n".join(lines)
    height = max(2.8, 0.28 * len(lines))
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=10,
        color="#E6EDF3",
        linespacing=1.35,
        transform=ax.transAxes,
    )
    _save(fig, name)


def _make_terminal_screens(stage4: dict, stage5: dict, stage6: dict, stage8: dict) -> None:
    stage1_lines = [
        "$ .venv/bin/python scripts/check_stage1_data.py",
        "Этап 1: проверка датасетов",
        "Размеченный набор: 2679 строк, 50 колонок",
        "Неразмеченный набор: 15243 строк, 49 колонок",
        "Распределение классов:",
        "  класс 0: 2356",
        "  класс 1: 218",
        "  класс 2: 105",
        "Пропуски (размеченный): 0",
        "Пропуски (неразмеченный): 0",
    ]
    _make_terminal_image("screen_stage1_validation.png", stage1_lines)

    stage5_lines = [
        "$ .venv/bin/python scripts/run_stage5_hybrid.py",
        "Этап 5: обучение гибридной модели",
        "Версия данных: improved",
        "Train shape: [55, 128, 49]",
        "Val shape: [9, 128, 49]",
        f"macro_f1(val): {stage5['versions']['improved']['val_metrics']['macro_f1']:.4f}",
        f"balanced_accuracy(val): {stage5['versions']['improved']['val_metrics']['balanced_accuracy']:.4f}",
        f"best_epoch: {stage5['versions']['improved']['history']['best_epoch']}",
        "ROC AUC: n/a (в validation отсутствует класс 1)",
        "Модель сохранена: output/models/stage5_hybrid_improved.keras",
    ]
    _make_terminal_image("screen_stage5_hybrid.png", stage5_lines)

    stage6_lines = [
        "$ .venv/bin/python scripts/run_stage6_ga_search.py",
        "Этап 6: генетический поиск гиперпараметров",
        "Конфиг: population=12, generations=8, fitness_epochs=10",
        "Всего fitness-оценок: 96",
        "Лучший индивид: generation=5, id=11",
        f"macro_f1(fitness): {stage6['best_fitness_record']['val_metrics']['macro_f1']:.4f}",
        f"balanced_accuracy(fitness): {stage6['best_fitness_record']['val_metrics']['balanced_accuracy']:.4f}",
        "Статус всех особей: FAIL=0",
        "Лучший геном: conv=16, kernel=5, gru=32, dense=64x2, adam/relu",
        "Финальное дообучение завершено: output/models/stage6_ga_best.keras",
    ]
    _make_terminal_image("screen_stage6_ga.png", stage6_lines)

    stage8_lines = [
        "$ .venv/bin/python scripts/run_stage8_final_eval.py",
        "Этап 8: финальная оценка на test",
        "Выбранный пайплайн: stage6_ga_no_ae",
        "Финальный seed: 1",
        f"macro_f1(test): {stage8['final_test_eval']['test_metrics']['macro_f1']:.4f}",
        f"balanced_accuracy(test): {stage8['final_test_eval']['test_metrics']['balanced_accuracy']:.4f}",
        "ROC AUC(test): n/a",
        f"Классы в test: {stage8['final_test_eval']['present_test_labels']}",
        f"Отсутствующий класс в test: {stage8['final_test_eval']['missing_test_labels']}",
        "Финальная модель: output/models/stage8_final_selected.keras",
    ]
    _make_terminal_image("screen_stage8_final.png", stage8_lines)


def _render_code_page(
    name: str,
    title: str,
    lines: list[str],
    page_no: int,
    total_pages: int,
    start_line_number: int,
) -> None:
    padded = [f"{idx:>4} | {line.rstrip()}" for idx, line in enumerate(lines, start=start_line_number)]
    header = [f"{title} (лист {page_no} из {total_pages})", "-" * 110]
    content = header + padded
    text = "\n".join(content)
    height = max(4.5, 0.22 * len(content))
    fig, ax = plt.subplots(figsize=(11.0, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(
        0.015,
        0.985,
        text,
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=7.2,
        color="#111111",
        linespacing=1.25,
        transform=ax.transAxes,
    )
    _save(fig, name)


def _generate_code_appendices() -> None:
    appendix_specs = [
        ("А", "Базовая модель классификации", ROOT / "src" / "models" / "baseline.py", "appendix_baseline"),
        ("Б", "Гибридная модель 1D-CNN -> GRU -> Dense", ROOT / "src" / "models" / "hybrid.py", "appendix_hybrid"),
        ("В", "Генетический алгоритм подбора гиперпараметров", ROOT / "src" / "training" / "ga_search.py", "appendix_ga"),
        ("Г", "Автоэнкодер и перенос весов энкодера", ROOT / "src" / "models" / "autoencoder.py", "appendix_autoencoder"),
    ]
    lines_per_page = 46
    tex_lines = []
    for letter, title, path, stem in appendix_specs:
        rel_path = path.relative_to(ROOT).as_posix()
        rel_path_tex = _escape_tex_text(rel_path)
        file_lines = path.read_text(encoding="utf-8").splitlines()
        pages = [file_lines[i : i + lines_per_page] for i in range(0, len(file_lines), lines_per_page)]
        tex_lines.append(rf"\section*{{Приложение {letter}. {title}}}")
        tex_lines.append(rf"\addcontentsline{{toc}}{{section}}{{Приложение {letter}. {title}}}")
        tex_lines.append(rf"\noindent\textbf{{Файл:}} \path{{{rel_path}}}")
        tex_lines.append(r"\par\vspace{0.4cm}")
        total_pages = len(pages)
        for index, page_lines in enumerate(pages, start=1):
            image_name = f"{stem}_p{index:02d}.png"
            start_line_number = 1 + (index - 1) * lines_per_page
            _render_code_page(
                image_name,
                rel_path,
                page_lines,
                index,
                total_pages,
                start_line_number,
            )
            tex_lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=\textwidth]{{figures/{image_name}}}",
                    rf"\caption{{Листинг файла {rel_path_tex}, лист {index}}}",
                    r"\end{figure}",
                ]
            )
        tex_lines.append(r"\clearpage")
    APPENDIX_SNIPPET_PATH.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> None:
    _prepare_output_dir()
    _setup_style()

    eda = _load_json(ROOT / "reports" / "experiments" / "eda_summary.json")
    stage3 = _load_json(ROOT / "reports" / "experiments" / "stage3_preprocessing_summary.json")
    stage4 = _load_json(ROOT / "reports" / "experiments" / "stage4_baseline_summary.json")
    stage5 = _load_json(ROOT / "reports" / "experiments" / "stage5_hybrid_summary.json")
    stage6 = _load_json(ROOT / "reports" / "experiments" / "stage6_ga_search_summary.json")
    stage8 = _load_json(ROOT / "reports" / "experiments" / "stage8_final_eval_summary.json")

    _plot_class_distribution(eda)
    _plot_outlier_features(eda)
    _plot_split_scheme(stage3)
    _plot_hybrid_scheme()
    _plot_autoencoder_scheme()
    _plot_model_comparison(stage4, stage5, stage6)
    _plot_ga_progress(stage6)
    _plot_seed_stability(stage8)
    _plot_test_confusion(stage8)
    _make_terminal_screens(stage4, stage5, stage6, stage8)
    _generate_code_appendices()

    print("Иллюстрации отчёта сгенерированы:")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(path.relative_to(ROOT))
    print(APPENDIX_SNIPPET_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
