.PHONY: install up down run test lint

install:
	pip install -r requirements.txt

up:
	docker compose up -d --wait

down:
	docker compose down

run:
	uvicorn api.main:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .
