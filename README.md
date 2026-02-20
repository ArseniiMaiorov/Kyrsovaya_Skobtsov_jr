# Kyrsovaya_Skobtsov_jr

Курсовой проект по ИАД/МО: классификация технического состояния МКА по телеметрии (ТМИ) с архитектурой `1D-CNN -> GRU -> Dense`.

## Текущая постановка (зафиксировано)
- Тип задачи: `multiclass` (вариант 1.1)
- Метки: `0 = штатное`, `1 = отказ`, `2 = сбой`
- Формирование последовательности: временные окна
  - `T = 128`, `stride = 32`, `overlap = 0.75`
- Контракт данных по умолчанию:
  - `data/tmi.csv`, `sep=,`, `encoding=utf-8`
  - токены пропусков: `"", NA, N/A, NaN, nan, null, None, -999, -9999`

## Этап 0 (инициализация)
Сделано:
- создана базовая структура проекта;
- зафиксированы спецификации в `docs/specs/*.md`;
- добавлен рабочий `config.yaml`;
- добавлены базовые модули в `src/`:
  - загрузка и валидация конфигурации;
  - валидация схемы данных и меток;
- добавлены тесты в `tests/`.

## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
pytest --cov=src --cov-report=term-missing --cov-fail-under=100
./scripts/run_tests_ru.sh
```

## Структура
```text
notebooks/
src/
  data/
  utils/
tests/
docs/
  specs/
  cheatsheets/
reports/
output/
report/
config.yaml
requirements.txt
```

## Политика изменения контрактов
Любое изменение `data/task/sequence/ae/compute`-контрактов фиксируется в `docs/devlog.md` с причиной и влиянием на метрики/поведение.

## Язык проекта
Текстовая документация, комментарии и сообщения об ошибках в коде ведутся на русском языке.
