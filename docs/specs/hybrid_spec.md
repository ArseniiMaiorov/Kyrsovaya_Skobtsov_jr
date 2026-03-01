# Спецификация гибридной модели (Этап 5)

## Модель
- Архитектура: `1D-CNN -> GRU -> Dense`.
- Вход: батч окон `(batch, T, n_features)`.
- На текущем этапе используется базовый конфиг из `config.yaml`, без GA-поиска.

## Базовый конфиг
- `n_conv_layers = 1`
- `conv_filters = 64`
- `conv_kernel_size = 5`
- `n_gru_layers = 1`
- `gru_units = 128`
- `n_dense_layers = 1`
- `dense_units = 128`
- `activation = relu`
- `optimizer = adam`
- `loss = sparse_categorical_crossentropy`

## Регуляризация
- `Dropout` в Conv-блоке и Dense-блоке.
- `L2` на Dense-слое через `l2_dense`.
- `EarlyStopping(patience=5, restore_best_weights=True)`.
- `ReduceLROnPlateau(patience=3, factor=0.5)`.

## Протокол оценки
- Сравнение выполняется на `raw` и `improved`.
- Выбор лучшей версии данных выполняется только по `val macro_f1`.
- `test` на этапе 5 не используется.

## Артефакты
- `reports/experiments/stage5_hybrid_summary.json`
- `reports/experiments/stage5_hybrid_summary.md`
- `output/models/stage5_hybrid_raw.keras`
- `output/models/stage5_hybrid_improved.keras`
