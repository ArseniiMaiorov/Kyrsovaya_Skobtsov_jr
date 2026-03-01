# Журнал разработки

## 2026-03-01 - Этап 8: финальная оценка на test

Что сделано:
- Добавлен скрипт `scripts/run_stage8_final_eval.py`.
- Реализован защищённый протокол:
  - лучший пайплайн выбирается только по `val`;
  - перед `test` выполняется проверка стабильности на `seed ∈ {0,1,2}`;
  - `test` используется один раз;
  - повторный запуск блокируется, если уже существует `stage8_final_eval_summary.json`.
- В качестве финального кандидата сравниваются:
  - `stage6_ga_no_ae`
  - `stage7_ae_pretrained`
- Для финальной оценки выбран `stage6_ga_no_ae`, так как он лучший по `val`.

Результат:
- Выбранный пайплайн: `stage6_ga_no_ae`
- Финальный seed: `1`
- Проверка стабильности на `val`:
  - `seed 0`: `macro_f1 = 0.6444`
  - `seed 1`: `macro_f1 = 0.6667`
  - `seed 2`: `macro_f1 = 0.2917`
- Относительно baseline условие стабильности не выполнено:
  - на `seed 2` модель опускается ниже baseline `improved` (`0.3137`).
- Финальная оценка на `test`:
  - `macro_f1 = 0.6667`
  - `balanced_accuracy = 1.0000`
  - `ROC AUC = n/a`
  - на текущем `test` отсутствует класс `2`
  - ошибок по `confusion matrix` не зафиксировано

Сформированы артефакты:
- `reports/experiments/stage8_final_eval_summary.json`
- `reports/experiments/stage8_final_eval_summary.md`
- `output/models/stage8_final_selected.keras`

## 2026-03-01 - Этап 7: AE-предобучение и fine-tuning классификатора

Что сделано:
- Добавлен модуль `src/models/autoencoder.py`:
  - AE строится на encoder-части гибридной модели;
  - decoder восстанавливает вход через `RepeatVector -> GRU(return_sequences=True) -> TimeDistributed(Dense)`;
  - реализованы извлечение и перенос encoder-весов.
- Добавлен модуль `src/training/autoencoder_training.py`:
  - разбиение неразмеченных окон на `pretrain_train/pretrain_val` без перемешивания;
  - обучение AE на задаче реконструкции;
  - оценка reconstruction MSE;
  - fine-tuning классификатора после переноса весов.
- Добавлен скрипт `scripts/run_stage7_autoencoder_pretrain.py`.
- Для Stage 7 используется `improved`-предобработка, обученная на размеченном `train`, и затем применяемая к `unlabeled`.
- В качестве encoder-конфига используется `stage6_best_genome` (если артефакт доступен).
- Добавлены тесты:
  - `tests/test_autoencoder.py`
  - `tests/test_autoencoder_training.py`

Результат:
- Неразмеченные окна для AE: `473`
- Внутренний split для AE:
  - `pretrain_train = 403`
  - `pretrain_val = 70`
- AE:
  - `mean_reconstruction_mse = 141.109283`
  - лучшая эпоха = `6`
  - эпох выполнено = `11`
- Классификатор с нуля:
  - `val macro_f1 = 0.6444`
  - `val balanced_accuracy = 0.9375`
- Классификатор после AE:
  - `val macro_f1 = 0.6444`
  - `val balanced_accuracy = 0.9375`
  - перенесены слои: `conv1d_1`, `batch_norm_1`, `gru_1`
- На текущем официальном `val` AE-предобучение не дало прироста относительно запуска `с нуля`.

Сформированы артефакты:
- `reports/experiments/stage7_autoencoder_pretrain_summary.json`
- `reports/experiments/stage7_autoencoder_pretrain_summary.md`
- `output/models/stage7_autoencoder.keras`
- `output/models/stage7_hybrid_scratch.keras`
- `output/models/stage7_hybrid_ae_finetuned.keras`

Проверка:
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=100`
- Результат: `253` теста, покрытие `src = 100%`.

## 2026-03-01 - Этап 6: генетический алгоритм для подбора гиперпараметров

Что сделано:
- Добавлен модуль `src/training/ga_search.py`:
  - фиксированное пространство поиска по 9 генам;
  - начальная популяция равномерным сэмплированием;
  - `tournament selection` с размером турнира `3`;
  - `two-point crossover`;
  - независимая мутация каждого гена с вероятностью `0.2`;
  - `elitism = 1`;
  - JSONL-лог всех индивидов.
- Добавлен скрипт `scripts/run_stage6_ga_search.py`.
- Для поиска используется только `improved`-версия данных.
- Fitness считается только на официальном `val`.
- Поддержано возобновление из частично заполненного `output/logs/ga_population_log.jsonl`.
- Сформированы артефакты:
  - `reports/experiments/stage6_ga_search_summary.json`
  - `reports/experiments/stage6_ga_search_summary.md`
  - `output/logs/ga_population_log.jsonl`
  - `output/artifacts/stage6_best_genome.json`
  - `output/models/stage6_ga_best.keras`
- Добавлены тесты `tests/test_ga_search.py`.

Результат:
- Полный лог GA: `96` записей (`12 x 8`).
- Лучший индивид по fitness:
  - `generation = 5`
  - `individual_id = 11`
  - `val macro_f1 = 0.6667`
  - `val balanced_accuracy = 1.0000`
- Финальное дообучение лучшего генома:
  - `val macro_f1 = 0.6667`
  - прирост к baseline (`этап 4`, `improved`) = `+0.3529`
  - прирост к базовому hybrid (`этап 5`, `improved`) = `+0.0222`

Проверка:
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=100`
- Результат: `244` теста, покрытие `src = 100%`.

## 2026-03-01 - Дополнительная rolling-диагностика устойчивости

Что сделано:
- Добавлен модуль `src/data/rolling_validation.py`.
- Реализована дополнительная expanding-window валидация внутри официального train-сегмента.
- Добавлен скрипт `scripts/run_rolling_diagnostics.py`.
- Диагностика выполняется на `improved`-версии данных для:
  - `baseline_logistic_regression`
  - `hybrid_cnn_gru_dense`
- Сохраняются артефакты:
  - `reports/experiments/rolling_diagnostics_summary.*`
  - `output/artifacts/rolling_diagnostics_plan.json`
- При неинформативном train-фолде (меньше двух классов) модель помечается как `FAIL`, но общий прогон не останавливается.
- Исправлен baseline-контур:
  - вероятности `predict_proba` теперь выравниваются по полному списку меток `0/1/2`, даже если модель обучена не на всех классах.
- Исправлен общий расчёт macro-метрик:
  - `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1` теперь считаются строго по фиксированному списку меток, а не по случайному набору классов, попавших в `y_true/y_pred`.

Зачем это добавлено:
- текущий официальный `val` слишком мал и не содержит все классы;
- rolling-диагностика нужна как дополнительная проверка устойчивости перед этапом 6 (GA);
- официальный `val` по ТЗ не меняется и не заменяется.

## 2026-03-01 - Этап 5: гибрид 1D-CNN -> GRU -> Dense

Что сделано:
- `config.yaml` расширен разделом `training.hybrid`.
- Добавлен модуль `src/models/hybrid.py`:
  - валидация конфига гибридной модели;
  - сборка `Conv1D -> BatchNorm -> Activation -> Dropout -> GRU -> Dense -> Softmax`;
  - компиляция с поддержкой `adam/rmsprop/nadam`.
- Добавлен модуль `src/training/hybrid_training.py`:
  - обязательные callbacks по ТЗ;
  - расчёт `class_weight=balanced`;
  - обучение без перемешивания;
  - оценка на `val`;
  - краткая сводка истории обучения.
- Добавлен скрипт `scripts/run_stage5_hybrid.py`.
- Добавлены тесты:
  - `tests/test_hybrid.py`
  - `tests/test_hybrid_training.py`

Фактический результат на реальных данных:
- `raw`: `val macro_f1 = 0.3111`
- `improved`: `val macro_f1 = 0.6444`
- прирост относительно baseline на `improved`:
  - baseline `0.4706`
  - hybrid `0.6444`
  - итоговый прирост `+0.1738` по `val macro_f1`

Ограничение текущего split:
- В `val` по-прежнему отсутствует класс `1`, поэтому `ROC AUC` на этапе 5 не вычисляется и сохраняется как `n/a` с пояснением.
- Это не нарушение контракта ТЗ, а свойство фиксированного time-based разбиения на текущем наборе.

Проверка:
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=100`
- Результат: `214` тестов, покрытие `src = 100%`.

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
