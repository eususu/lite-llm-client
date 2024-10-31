all:

test:
	pytest

test_minor:
	pytest tests/test_minor.py

test_oai:
	pytest tests/test_oai.py

test_anthropic:
	pytest tests/test_anthropic.py

test_gemini:
	pytest tests/test_gemini.py



nm:
	@python -c 'import lite_llm_client; exports = list(filter(lambda x: x.find("_") == -1, dir(lite_llm_client))); print("\n".join(exports));'

build:
	python -m build
	pip install -e .

check:
	twine check dist/*

upload_test:
	twine upload -r testpypi dist/*

upload:
	twine upload dist/*