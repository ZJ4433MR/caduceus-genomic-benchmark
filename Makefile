.PHONY: smoke table2 table3 table4 clean

PYTHON ?= python

smoke:
	$(PYTHON) -m compileall -q src caduceus scripts train.py
	bash scripts/test_imports.sh

table2:
	$(PYTHON) scripts/aggregate_results.py --input results/raw/table2_runs.csv --output results/tables/table2_main_results.csv

table3:
	$(PYTHON) scripts/aggregate_results.py --input results/raw/table3_same_budget_runs.csv --output results/tables/table3_same_budget_controls.csv

table4:
	$(PYTHON) scripts/aggregate_results.py --input results/raw/table4_ablation_runs.csv --output results/tables/table4_ablation.csv

clean:
	rm -rf .pytest_cache __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

