# HK-RL developer task runner.
# Most targets operate on the `python/` package; schema codegen feeds both
# Python and the C# mod from schema/hkrl.fbs (the single source of truth).

PY        := python
FLATC     ?= flatc
FLATC_PY  ?= $(FLATC)
FLATC_CS  ?= $(FLATC)
CSHARP_FLATC_VERSION ?= 23.5.26
PKG_DIR   := python
FBS       := schema/hkrl.fbs
PY_SCHEMA := python/hkrl/schema
CS_SCHEMA := mod/HKRLEnvMod/Schema
GIT_SHA  ?= $(shell git rev-parse HEAD 2>/dev/null)
GIT_DIRTY ?= $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo true || echo false)

.DEFAULT_GOAL := help
.PHONY: help gen-schema gen-schema-py gen-schema-cs install install-hooks check lint format-check typecheck test fmt clean smoke phase8-smoke phase8-dashboard phase8-profile phase8-eval-report phase8-release-checklist phase8-release-evidence phase8-verify-release-evidence

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- Schema codegen (single source of truth -> C# + Python) ----------------
gen-schema: gen-schema-py gen-schema-cs ## Regenerate all FlatBuffers bindings

gen-schema-py: ## Generate Python FlatBuffers bindings
	mkdir -p $(PY_SCHEMA)
	rm -rf $(PY_SCHEMA)/HKRL
	$(FLATC_PY) --python -o $(PY_SCHEMA) $(FBS)

gen-schema-cs: ## Generate C# FlatBuffers bindings
	@version="$$($(FLATC_CS) --version)"; \
	case "$$version" in \
	  *"$(CSHARP_FLATC_VERSION)"*) ;; \
	  *) echo "error: C# schema generation requires flatc $(CSHARP_FLATC_VERSION); got: $$version" >&2; \
	     echo "Use environment-mod-build.yml, or pass FLATC_CS=/path/to/flatc-$(CSHARP_FLATC_VERSION)." >&2; \
	     exit 2 ;; \
	esac
	mkdir -p $(CS_SCHEMA)
	rm -rf $(CS_SCHEMA)/HKRL
	$(FLATC_CS) --csharp -o $(CS_SCHEMA) $(FBS)

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
	$(PY) scripts/run_phase8_smoke.py --config configs/train/remote_learner.yaml \
	  --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml \
	  --work-dir runs/phase8-smoke --output runs/phase8-smoke/summary.json \
	  --dashboard-html runs/phase8-smoke/dashboard.html \
	  --dashboard-json runs/phase8-smoke/dashboard.json

phase8-profile: ## Render offline Phase 8 profiling report under runs/phase8-smoke
	mkdir -p runs/phase8-smoke
	$(PY) scripts/run_phase8_smoke.py --config configs/train/remote_learner.yaml \
	  --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml \
	  --work-dir runs/phase8-smoke --output runs/phase8-smoke/summary.json \
	  --profile-json runs/phase8-smoke/profile.json \
	  --profile-md runs/phase8-smoke/profile.md

phase8-eval-report: ## Render fixed-seed eval report from runs/eval.json
	$(PY) scripts/render_eval_report.py --eval-json runs/eval.json \
	  --output-json runs/eval-report.json --output-md runs/eval-report.md \
	  --fail-on-critical

phase8-release-checklist: ## Render Phase 8 release checklist under runs/release
	mkdir -p runs/release
	$(PY) scripts/render_release_checklist.py --version phase8 --git-sha "$(GIT_SHA)" --git-dirty "$(GIT_DIRTY)" --output-json runs/release/checklist.json --output-md runs/release/checklist.md

phase8-release-evidence: ## Render Phase 8 release evidence bundle manifest
	mkdir -p runs/phase8-smoke runs/release
	$(PY) scripts/run_phase8_smoke.py --config configs/train/remote_learner.yaml \
	  --tasks configs/tasks/gruz_mother.yaml configs/tasks/hornet_protector.yaml \
	  --work-dir runs/phase8-smoke --output runs/phase8-smoke/summary.json \
	  --dashboard-html runs/phase8-smoke/dashboard.html \
	  --dashboard-json runs/phase8-smoke/dashboard.json \
	  --profile-json runs/phase8-smoke/profile.json \
	  --profile-md runs/phase8-smoke/profile.md
	$(PY) scripts/render_release_checklist.py --version phase8 --git-sha "$(GIT_SHA)" --git-dirty "$(GIT_DIRTY)" --output-json runs/release/checklist.json --output-md runs/release/checklist.md
	$(PY) scripts/render_release_evidence.py --version phase8 --git-sha "$(GIT_SHA)" --git-dirty "$(GIT_DIRTY)" --output-json runs/release/evidence.json --output-md runs/release/evidence.md
	$(PY) scripts/verify_release_evidence.py --manifest runs/release/evidence.json --git-sha "$(GIT_SHA)" --git-dirty "$(GIT_DIRTY)" --output-json runs/release/evidence-verification.json

phase8-verify-release-evidence: ## Verify Phase 8 release evidence hashes
	$(PY) scripts/verify_release_evidence.py --manifest runs/release/evidence.json --git-sha "$(GIT_SHA)" --git-dirty "$(GIT_DIRTY)" --output-json runs/release/evidence-verification.json

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(PKG_DIR)/.ruff_cache $(PKG_DIR)/.mypy_cache $(PKG_DIR)/.pytest_cache
