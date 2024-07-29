all: test_anthropic

test:
	pytest

test_oai:
	pytest tests/test_oai.py

test_anthropic:
	pytest tests/test_anthropic.py

test_gemini:
	pytest tests/test_gemini.py


nm:
	python -c 'import lite_llm_client; print("\n".join(dir(lite_llm_client)))'
build:
	python -m build
	pip install -e .