#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/pytest" ]]; then
  PYTEST_CMD=".venv/bin/pytest"
else
  PYTEST_CMD="pytest"
fi

UNIT_LOG="/tmp/pytest_unit.log"
COV_LOG="/tmp/pytest_cov.log"

echo "Запуск модульных тестов..."
if ${PYTEST_CMD} -q >"${UNIT_LOG}" 2>&1; then
  echo "Модульные тесты: успешно."
else
  echo "Модульные тесты: ошибка. Ниже вывод pytest:"
  cat "${UNIT_LOG}"
  exit 1
fi

echo "Проверка покрытия..."
if ${PYTEST_CMD} --cov=src --cov-report=term-missing --cov-fail-under=100 >"${COV_LOG}" 2>&1; then
  total_line=$(grep '^TOTAL' "${COV_LOG}" | tail -n 1 || true)
  coverage_value=$(echo "${total_line}" | awk '{print $4}')
  if [[ -z "${coverage_value}" ]]; then
    coverage_value="не определено"
  fi
  echo "Покрытие кода: ${coverage_value} (требование 100% выполнено)."
else
  echo "Проверка покрытия: не пройдена. Ниже вывод pytest:"
  cat "${COV_LOG}"
  exit 1
fi

echo "Все проверки завершены успешно."
