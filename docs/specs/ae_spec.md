# Спецификация автокодировщика

## Роль в проекте
- Автокодировщик обязателен по постановке.
- Для `multiclass` используется режим 6.1: предобучение энкодера на неразмеченных окнах, затем перенос весов в классификатор.

## Использование данных
- Основной источник для AE: `data/MKA_04.2015_unlabeled.csv`.
- Предобработка для AE должна совпадать с `improved`.
- По умолчанию статистики `median / q01-q99 / scaler` обучаются на размеченном `train`, затем применяются к `unlabeled`.
- Окна строятся теми же правилами `T=128`, `stride=32`, time-based split по сериям.
- Для самого AE из неразмеченных окон выделяется внутренний time-based `pretrain_val` (по умолчанию последние `15%` окон).
- `test` используется один раз на финальном этапе и в Stage 7 не затрагивается.

## Архитектура
- Encoder:
  - `Conv1D + BatchNorm + Activation + Dropout`
  - `GRU`
- Decoder:
  - `RepeatVector(T)`
  - `GRU(return_sequences=True)`
  - `TimeDistributed(Dense(n_features))`
- Loss: `MSE`
- Оптимизатор берётся из текущего encoder-конфига гибридной модели.

## Transfer learning
- В классификатор переносятся веса слоёв:
  - `conv1d_*`
  - `batch_norm_*`
  - `gru_*`
- Сравнение обязательно выполняется в двух режимах:
  - классификатор `с нуля`
  - классификатор `после AE`

## Обязательная проверка
- Тест на синтетике: AE восстанавливает форму входа `(batch, T, n_features)`.
- Тест на перенос весов: веса энкодера реально переносятся в классификатор.
- Smoke-тест: короткий прогон обучения без падения пайплайна.

## Артефакты этапа
- `reports/experiments/stage7_autoencoder_pretrain_summary.json`
- `reports/experiments/stage7_autoencoder_pretrain_summary.md`
- `output/models/stage7_autoencoder.keras`
- `output/models/stage7_hybrid_scratch.keras`
- `output/models/stage7_hybrid_ae_finetuned.keras`
