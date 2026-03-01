# Журнал разработки

## 2026-03-01 - Модуль воспроизводимости

Что сделано:
- Добавлен `src/utils/reproducibility.py`.
- В `config.yaml` добавлен раздел `reproducibility`:
  - `global_seed`
  - `checksum_algorithm`
- Реализованы артефакты:
  - `output/artifacts/config_snapshot.yaml`
  - `output/artifacts/reproducibility_<stage>.json`
- В метаданные сохраняются:
  - seed;
  - версия Python;
  - версии библиотек;
  - размер и checksum размеченного и неразмеченного датасетов.
- Модуль подключен к скриптам:
  - `scripts/check_stage1_data.py`
  - `scripts/run_eda_stage2.py`
  - `scripts/run_stage3_preprocessing.py`
  - `scripts/run_stage4_baseline.py`

Инженерное решение:
- Для этапов 1-4 TensorFlow seed по умолчанию отключён, чтобы не загружать TensorFlow в скриптах, где он не нужен.
- Для этапов 5-8 при обучении нейросетей нужно вызывать инициализацию с включённым TensorFlow seed.

Проверка:
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=100`
- Результат: `167` тестов, покрытие `src = 100%`.

## 2026-03-01 - Приведение этапов 1-4 к обновлённому ТЗ

Что изменено:
- Контракт `config.yaml` обновлён под новое ТЗ: `time_order_windows`, `49` признаков, `raw/improved` по фиксированным правилам.
- `src/data/io.py` и `src/data/validation.py` усилены:
  - строгая проверка числа признаков;
  - проверка числового типа признаков;
  - правило `drop_all_nan_rows`.
- `src/data/splits.py` переписан:
  - разбиение выполняется по непрерывным временным участкам;
  - окна строятся отдельно внутри `train/val/test`;
  - контроль утечки проверяет пересечение строк, а не только целевых позиций.
- `src/data/preprocessing.py` приведён к контракту:
  - `raw = median + float32`;
  - `improved = median + winsorize(q01/q99) + RobustScaler`.
- `src/data/eda.py` расширен:
  - пропуски по признакам;
  - базовая статистика;
  - поиск заглушек `-999/-9999`;
  - константные признаки;
  - дисбаланс классов.
- `scripts/run_stage4_baseline.py` закреплён за `LogisticRegression` на flattened окнах и сравнением только по `val`.
- Документация `README.md`, `docs/specs/*`, `docs/cheatsheets/*` синхронизирована с новым ТЗ.

Что осталось незакрытым по ТЗ:
- Этап 5: гибрид `1D-CNN -> GRU -> Dense`.
- Этап 6: генетический алгоритм.
- Этап 7: AE-предобучение.
- Этап 8: финальная оценка на `test`.
- `report/course_report.tex`.

Проверка:
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=100`
- После правок покрытие `src` должно оставаться `100%`.

## 2026-02-20 - Инициализация этапа 0

Что сделано:
- Инициализирована структура проекта.
- Зафиксированы спецификации:
  - `docs/specs/data_schema.md`
  - `docs/specs/task_spec.md`
  - `docs/specs/sequence_spec.md`
  - `docs/specs/ae_spec.md`
  - `docs/specs/compute_budget.md`
- Добавлен `config.yaml` как единый источник параметров.
- Добавлены базовые модули `src/utils/config.py` и `src/data/validation.py`.
- Добавлены стартовые тесты для проверки контрактов и ошибок.

Правило изменения контрактов:
- Любая правка `data/task/sequence/ae/compute` контрактов фиксируется отдельной записью:
  - причина изменения;
  - какие метрики или поведение изменились;
  - обратная совместимость (если нарушена).

Проверка:
- Команда `pytest --cov=src --cov-report=term-missing --cov-fail-under=100` выполнена успешно.
- Текущий результат: 32 теста пройдено, покрытие `src` = 100%.

## 2026-02-20 - Локализация на русский язык

Что сделано:
- Переведены на русский язык все текстовые артефакты в `README.md` и `docs/`.
- Переведены докстринги и сообщения ошибок в `src/utils/config.py` и `src/data/validation.py`.
- Актуализированы проверки в тестах под русские сообщения ошибок.
- Добавлен `scripts/run_tests_ru.sh` с русским итоговым резюме по тестам и покрытию.

Проверка:
- Команда `pytest --cov=src --cov-report=term-missing --cov-fail-under=100` выполнена успешно.
- Текущий результат: 32 теста пройдено, покрытие `src` = 100%.

## 2026-02-23 - Этап 1: получение и валидация датасета ТМИ

Что сделано:
- Обновлен контракт данных под фактические источники:
  - `data/MKA_TMI_labels.xls` (размеченный)
  - `data/MKA_04.2015_unlabeled.csv` (неразмеченный)
- Реализован модуль `src/data/io.py`:
  - загрузка `csv/xls`;
  - контрактная проверка меток и колонок;
  - согласованная загрузка размеченного и неразмеченного наборов.
- Обновлена валидация `config.yaml` под разделы `data.labeled` и `data.unlabeled`.
- Добавлены тесты Этапа 1:
  - `tests/test_io.py`;
  - обновлен `tests/test_config.py`.
- Добавлен скрипт проверки реальных данных: `scripts/check_stage1_data.py`.

Фактический результат проверки данных:
- Размеченный набор: `2679 x 50`.
- Неразмеченный набор: `15243 x 49`.
- Распределение классов: `0=2356`, `1=218`, `2=105`.
- Пропуски: `0` в обоих наборах.

Проверка:
- Команда `pytest --cov=src --cov-report=term-missing --cov-fail-under=100` выполнена успешно.
- Результат на этом этапе: 46 тестов пройдено, покрытие `src` = 100%.

## 2026-02-23 - Этап 2: EDA и выводы для предобработки

Что сделано:
- Реализован модуль `src/data/eda.py`:
  - пропуски и их доля;
  - распределение классов;
  - доля выбросов (IQR);
  - пары признаков с высокой корреляцией.
- Добавлены тесты `tests/test_eda.py`.
- Добавлен скрипт `scripts/run_eda_stage2.py`.
- Сформированы артефакты:
  - `reports/experiments/eda_summary.md`
  - `reports/experiments/eda_summary.json`

Ключевые выводы:
- Выраженный дисбаланс классов: `majority/minority = 22.4381`.
- Пропуски отсутствуют в обоих наборах.
- Для обучения классификатора требуется стратифицированное разбиение и учет дисбаланса (например, class weights).

Проверка:
- Команда `pytest --cov=src --cov-report=term-missing --cov-fail-under=100` выполнена успешно.
- Текущий результат: 55 тестов пройдено, покрытие `src` = 100%.

## 2026-02-23 - Этап 3: split и две версии данных (raw/improved)

Что сделано:
- Реализован модуль `src/data/splits.py`:
  - детерминированный `stratified_random` split;
  - проверка отсутствия пересечения индексов;
  - отчет по распределению классов по сплитам.
- Реализован модуль `src/data/preprocessing.py`:
  - `raw`-версия данных (без масштабирования);
  - `improved`-версия (медианная импутация, clipping по квантилям, `StandardScaler`);
  - правило `fit только на train`, затем `transform` для `val/test/unlabeled`.
- Обновлен контракт `config.yaml`:
  - добавлены разделы `split` и `preprocessing`.
- Добавлены тесты:
  - `tests/test_splits.py`
  - `tests/test_preprocessing.py`
  - расширен `tests/test_config.py` под новые разделы.
- Добавлен скрипт запуска этапа:
  - `scripts/run_stage3_preprocessing.py`
- Добавлены артефакты:
  - `reports/experiments/stage3_preprocessing_summary.md`
  - `reports/experiments/stage3_preprocessing_summary.json`

Фактический результат (текущий запуск):
- Split:
  - Train: `1607 x 50`
  - Val: `536 x 50`
  - Test: `536 x 50`
- Распределение классов:
  - Train: `{0: 1414, 1: 130, 2: 63}`
  - Val: `{0: 471, 1: 44, 2: 21}`
  - Test: `{0: 471, 1: 44, 2: 21}`
- Проверка improved-преобразования:
  - `max(|mean(train_scaled)|) = 0.000000`
  - `mean(|std(train_scaled)-1|) = 0.000000`

Проверка:
- Команда `pytest --cov=src --cov-report=term-missing --cov-fail-under=100` выполнена успешно.
- Текущий результат: 99 тестов пройдено, покрытие `src` = 100%.

## 2026-02-23 - Этап 4: baseline и единые метрики

Что сделано:
- Реализован модуль `src/metrics/metrics.py` для унифицированного расчета метрик.
- Реализован модуль `src/models/baseline.py`:
  - `logistic_regression`;
  - `random_forest`;
  - единый сценарий обучения и оценки.
- Добавлен скрипт этапа `scripts/run_stage4_baseline.py`.
- Добавлены тесты:
  - `tests/test_metrics.py`;
  - `tests/test_baseline.py`.
- Добавлены спецификация и шпаргалка:
  - `docs/specs/baseline_spec.md`;
  - `docs/cheatsheets/005_baseline_stage4.txt`.
- Артефакты этапа:
  - `reports/experiments/stage4_baseline_summary.md`;
  - `reports/experiments/stage4_baseline_summary.json`.

Принцип этапа:
- Сравнение baseline проводится на двух версиях данных (`raw` и `improved`).
- Выбор лучшей baseline-модели выполняется только по `val macro_f1`.
- `test`-метрики сохраняются как справочные и не используются для выбора.

Фактический результат (текущий запуск):
- `raw`:
  - `logistic_regression (val macro_f1) = 0.8372`
  - `random_forest (val macro_f1) = 0.8936`
  - лучшая по val: `random_forest`
- `improved`:
  - `logistic_regression (val macro_f1) = 0.8251`
  - `random_forest (val macro_f1) = 0.8987`
  - лучшая по val: `random_forest`

Проверка:
- Команда `pytest --cov=src --cov-report=term-missing --cov-fail-under=100` выполнена успешно.
- Текущий результат: 123 теста пройдено, покрытие `src` = 100%.
