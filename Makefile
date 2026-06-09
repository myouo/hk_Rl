# HK-RL developer task runner.
# Most targets operate on the `python/` package; schema codegen feeds both
# Python and the C# mod from schema/hkrl.fbs (the single source of truth).

PY        := python
PKG_DIR   := python
FBS       := schema/hkrl.fbs
PY_SCHEMA := python/hkrl/schema
CS_SCHEMA := mod/HKRLEnvMod/Schema

.DEFAULT_GOAL := help
.PHONY: help gen-schema gen-schema-py gen-schema-cs install install-hooks check lint format-check typecheck test fmt clean smoke phase8-smoke phase8-dashboard

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- Schema codegen (single source of truth -> C# + Python) ----------------
gen-schema: gen-schema-py gen-schema-cs ## Regenerate all FlatBuffers bindings

gen-schema-py: ## Generate Python FlatBuffers bindings
	flatc --python -o $(PY_SCHEMA) $(FBS)

gen-schema-cs: ## Generate C# FlatBuffers bindings
	flatc --csharp -o $(CS_SCHEMA) $(FBS)

# ---- Python dev ------------------------------------------------------------
install: ## Editable install with dev extras
	pip install -e "$(PKG_DIR)[dev]"

install-hooks: ## Configure git to use tracked hooks
	git config core.hooksPath .githooks

check: gen-schema format-check lint typecheck test ## Run local quality gates

lint: ## ruff lint
	cd $(PKG_DIR) && ruff check .

format-check: ## ruff format check
	cd $(PKG_DIR) && ruff format --check .

fmt: ## ruff format
	cd $(PKG_DIR) && ruff format .

typecheck: ## mypy
	cd $(PKG_DIR) && mypy hkrl

test: ## pytest unit suite
	cd $(PKG_DIR) && pytest

smoke: ## Run random-policy smoke against a live env (Phase 2+)
	$(PY) scripts/train.py --config configs/train/ppo_mlp.yaml --smoke

phase8-smoke: ## Run offline distributed wiring smoke (no live game required)
	$(PY) scripts/run_phase8_smoke.py --config configs/train/remote_learner.yaml --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml

phase8-dashboard: ## Render offline Phase 8 smoke dashboard under runs/phase8-smoke
	mkdir -p runs/phase8-smoke
	$(PY) scripts/run_phase8_smoke.py --config configs/train/remote_learner.yaml --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml --work-dir runs/phase8-smoke --output runs/phase8-smoke/summary.json
	$(PY) scripts/render_phase8_dashboard.py --summary runs/phase8-smoke/summary.json --output-html runs/phase8-smoke/dashboard.html --output-json runs/phase8-smoke/dashboard.json

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(PKG_DIR)/.ruff_cache $(PKG_DIR)/.mypy_cache $(PKG_DIR)/.pytest_cache
