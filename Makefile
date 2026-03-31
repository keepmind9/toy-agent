.PHONY: lint fmt fmtcheck run mcp test

lint:
	uv run ruff check src/ main.py tests/

fmt:
	uv run ruff format src/ main.py tests/

fmtcheck:
	uv run ruff format --check src/ main.py tests/

check: lint fmtcheck

run:
	uv run python main.py

mcp:
	uv run python tests/mcp_sse_server.py

test:
	uv run pytest tests/
