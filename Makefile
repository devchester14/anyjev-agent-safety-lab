.PHONY: setup check test ci-fixture local-benchmark

MODEL ?= Qwen/Qwen3-1.7B
REVISION ?= 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e # pragma: allowlist secret
DEVICE ?= mps
OUTPUT ?= reports/local/model-run

setup:
	uv sync --extra dev

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest

test:
	uv run pytest

ci-fixture:
	SOURCE_DATE_EPOCH=1792108800 uv run anyjev-decision-lab benchmark --backend fixture --output reports/ci
	uv run anyjev-decision-lab report --decisions reports/ci/decisions.json --output reports/ci
	uv run anyjev-decision-lab verify --decisions reports/ci/decisions.json --report reports/ci/report.json

local-benchmark:
	@if [ "$(strip $(MODEL))" != "Qwen/Qwen3-1.7B" ] && [ "$(strip $(REVISION))" = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ]; then echo "Set REVISION to the immutable revision for $(strip $(MODEL))"; exit 2; fi # pragma: allowlist secret
	uv sync --extra hf --extra dev
	uv run anyjev-decision-lab benchmark --backend hf --model $(strip $(MODEL)) --revision $(strip $(REVISION)) --device $(strip $(DEVICE)) --output $(strip $(OUTPUT))
	uv run anyjev-decision-lab report --decisions $(strip $(OUTPUT))/decisions.json --output $(strip $(OUTPUT)) --site docs/index.html
