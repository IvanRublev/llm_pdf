.PHONY: deps lint shell server server_headless test

deps:
	poetry install

lint:
	poetry run ruff check . 

shell:
	poetry run python

server:
	poetry run streamlit run app.py

server_headless:
	poetry run streamlit run app.py --browser.serverAddress 0.0.0.0 --server.headless true

test:
	poetry run -- ptw -- -s -vv $(args)

test_once:
	poetry run pytest -s
