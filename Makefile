install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .
	black --check .

test:
	pytest -q

smoke:
	python scripts/smoke_test.py

run-all:
	python scripts/run_sweep.py --config configs/experiments/all_models_mmlu_sweep.yaml

plot:
	python scripts/plot_results.py
