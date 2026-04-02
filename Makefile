.PHONY: lint fmt fmtcheck run mcp test

lint:
	uv run ruff check toy_agent/ main.py tests/

fmt:
	uv run ruff format toy_agent/ main.py tests/

fmtcheck:
	uv run ruff format --check toy_agent/ main.py tests/

check: lint fmtcheck

run:
	uv run python main.py

mcp:
	uv run python tests/mcp_sse_server.py

test:
	uv run pytest tests/
