all:
	pytest tests/test_anthropic.py
#pytest

build:
	python -m build
	pip install -e .