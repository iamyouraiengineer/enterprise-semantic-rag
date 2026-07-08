.PHONY: install test run-api run-ui lint clean

PYTHON := python3

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

run-api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	$(PYTHON) -m streamlit run ui/app_ui.py

lint:
	$(PYTHON) -m mypy src/ config/ --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/*.log