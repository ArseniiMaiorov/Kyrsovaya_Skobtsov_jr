# Бюджет вычислений

## Политика по железу
- Фактическое железо (CPU/GPU) фиксируется в отчете.

## Бюджет GA
- `population_size = 12`
- `generations = 8`
- `max_epochs_fitness = 10`
- `max_epochs_final = 50`

Итого базовый объем fitness-оценок: `12 * 8 = 96`.

## Ограничения обучения
- `EarlyStopping(patience=5, restore_best_weights=True)`
- `ReduceLROnPlateau(patience=3, factor=0.5)`

## Цель по качеству
- Целевой прирост: `macro-F1 +0.03` относительно baseline на validation.
- Проверка устойчивости: минимум 3 seed в финальной сверке.
