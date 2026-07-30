.PHONY: validate smoke archive clean

PYTHON ?= python

validate:
	$(PYTHON) scripts/validate_package.py
	$(PYTHON) scripts/verify_reported_results.py

smoke:
	$(PYTHON) -m compileall -q src caduceus scripts train.py vep_embeddings.py vep_svm_eval.py
	$(PYTHON) scripts/test_window_readout.py

archive: validate
	$(PYTHON) scripts/build_anonymous_archive.py

clean:
	rm -rf .pytest_cache __pycache__ dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
