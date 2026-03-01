.PHONY: build test stage4 stage5 diagnostics eval shell

build:
	docker-compose build

test:
	docker-compose run --rm coursework pytest -q

stage4:
	docker-compose run --rm coursework python scripts/run_stage4_baseline.py

stage5:
	docker-compose run --rm coursework python scripts/run_stage5_hybrid.py

diagnostics:
	docker-compose run --rm coursework python scripts/run_posthoc_validation_diagnostics.py

eval:
	docker-compose run --rm coursework python scripts/run_stage8_final_eval.py

shell:
	docker-compose run --rm coursework bash
