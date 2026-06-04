.PHONY: install train api ui test lint

install:
	pip install -e .[dev]

train:
	python src/train.py

api:
	uvicorn app.api:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check src/ app/ tests/
