.PHONY: install build test lint clean upload upload-test version-bump

# ── Development setup ─────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

lint:
	ruff check dspipeline/
	ruff format --check dspipeline/

format:
	ruff format dspipeline/

# ── Build ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf dist/ build/ *.egg-info/ __pycache__ dspipeline/__pycache__

build: clean
	python -m build
	@echo ""
	@echo "Built packages:"
	@ls -lh dist/

# ── Validate before upload ────────────────────────────────────────────────────
check: build
	twine check dist/*

# ── Publish ───────────────────────────────────────────────────────────────────
# Upload to TestPyPI first to verify everything looks correct:
#   make upload-test
# Then publish to the real PyPI:
#   make upload

upload-test: check
	twine upload --repository testpypi dist/*
	@echo ""
	@echo "Install from TestPyPI with:"
	@echo "  pip install --index-url https://test.pypi.org/simple/ dspipeline"

upload: check
	twine upload dist/*
	@echo ""
	@echo "Install with:  pip install dspipeline"

# ── Version bump helper ───────────────────────────────────────────────────────
# Usage:  make version-bump NEW=0.2.0
version-bump:
	@if [ -z "$(NEW)" ]; then echo "Usage: make version-bump NEW=x.y.z"; exit 1; fi
	sed -i 's/^version = .*/version = "$(NEW)"/' pyproject.toml
	sed -i 's/__version__ = .*/__version__ = "$(NEW)"/' dspipeline/__init__.py
	@echo "Version bumped to $(NEW)"
	@echo "Remember to: git tag v$(NEW) && git push --tags"
