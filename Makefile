test:
	uv run pytest --cov=genflow tests/
	uv run rm .coverage

type-check:
	uv run ty check "genflow"

mutation-test:
	uv run mutmut run
	uv run rm -rf mutants