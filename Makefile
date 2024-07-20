all:
	pytest

build:
	python -m build
	pip install -e .