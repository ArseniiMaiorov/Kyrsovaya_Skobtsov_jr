# Kyrsovaya_Skobtsov_jr

Курсовой проект по ИАД/МО: классификация технического состояния МКА по телеметрии (ТМИ) с архитектурой `1D-CNN -> GRU -> Dense`.

## Текущая постановка (зафиксировано)
- Тип задачи: `multiclass` (вариант 1.1)
- Метки: `0 = штатное`, `1 = отказ`, `2 = сбой`
- Формирование последовательности: временные окна
  - `T = 128`, `stride = 32`, `overlap = 0.75`
- Источники данных:
  - размеченный: `data/MKA_TMI_labels.xls` (49 признаков + `Class`)
  - неразмеченный: `data/MKA_04.2015_unlabeled.csv` (49 признаков)
- Токены пропусков: `"", NA, N/A, NaN, nan, null, None, -999, -9999`

## Этап 0 (инициализация)
Сделано:
- создана базовая структура проекта;
- зафиксированы спецификации в `docs/specs/*.md`;
- добавлен рабочий `config.yaml`;
- добавлены базовые модули в `src/`;
- добавлены тесты в `tests/`.

## Этап 1 (получение и валидация датасета)
Сделано:
- реализован загрузчик `src/data/io.py` для `csv/xls`;
- добавлена контрактная проверка размеченного и неразмеченного наборов;
- добавлены тесты ошибок формата и схемы;
- добавлен скрипт быстрой проверки данных `scripts/check_stage1_data.py`.

## Этап 2 (EDA)
Сделано:
- реализован модуль `src/data/eda.py`;
- добавлен скрипт `scripts/run_eda_stage2.py`;
- формируются артефакты в `reports/experiments/eda_summary.*`;
- в сводке сохраняются пропуски, базовая статистика, заглушки `-999/-9999`, константные признаки, корреляции и дисбаланс.

## Этап 3 (предобработка и split)
Сделано:
- реализован модуль `src/data/splits.py` (time-based split по непрерывным участкам, затем окна внутри каждого split);
- реализован модуль `src/data/preprocessing.py` (`raw` и `improved` версии по новому ТЗ);
- добавлен скрипт `scripts/run_stage3_preprocessing.py`;
- формируются артефакты в `reports/experiments/stage3_preprocessing_summary.*`.

## Этап 4 (baseline)
Сделано:
- реализован модуль `src/models/baseline.py`;
- реализован единый модуль метрик `src/metrics/metrics.py`;
- добавлен скрипт `scripts/run_stage4_baseline.py` для `LogisticRegression` на flattened окнах;
- формируются артефакты в `reports/experiments/stage4_baseline_summary.*`.

## Этапы 5-8
Пока не реализованы:
- `scripts/run_stage5_hybrid.py`
- `scripts/run_stage6_ga_search.py`
- `scripts/run_stage7_autoencoder_pretrain.py`
- `scripts/run_stage8_final_eval.py`

## Модуль воспроизводимости
Реализован единый модуль `src/utils/reproducibility.py`, который:
- фиксирует глобальный seed;
- сохраняет `output/artifacts/config_snapshot.yaml`;
- сохраняет stage-specific метаданные `output/artifacts/reproducibility_*.json`;
- собирает версии Python/библиотек;
- считает checksum и размер исходных датасетов.

На этапах 1-4 TensorFlow seed по умолчанию не активируется, чтобы не тянуть TensorFlow в нетренировочных скриптах. Для будущих этапов 5-8 его нужно включать явно.

## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
pytest --cov=src --cov-report=term-missing --cov-fail-under=100
./scripts/run_tests_ru.sh
./scripts/check_stage1_data.py
./scripts/run_eda_stage2.py
./scripts/run_stage3_preprocessing.py
./scripts/run_stage4_baseline.py
```

## Структура
```text
src/
  data/
  metrics/
  models/
  training/
  utils/
tests/
docs/
  specs/
  cheatsheets/
reports/
output/
report/
scripts/
config.yaml
requirements.txt
```

## Политика изменения контрактов
Любое изменение `data/task/sequence/ae/compute`-контрактов фиксируется в `docs/devlog.md` с причиной и влиянием на метрики/поведение.

## Язык проекта
Текстовая документация, комментарии и сообщения об ошибках в коде ведутся на русском языке.
