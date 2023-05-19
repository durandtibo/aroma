SHELL=/bin/bash
NAME=aroma
SOURCE=src/$(NAME)
TESTS=tests
UNIT_TESTS=tests/unit
INTEGRATION_TESTS=tests/integration

.PHONY : conda
conda :
	conda env create -f environment.yaml --force

.PHONY : config-poetry
config-poetry :
	poetry config experimental.system-git-client true
	poetry config --list

.PHONY : install-min
install-min :
	poetry install --no-interaction
	pip install --upgrade torch>=2.0.1  # TODO: https://github.com/pytorch/pytorch/issues/100974

.PHONY : install
install :
	poetry install --no-interaction --all-extras
	pip install --upgrade torch>=2.0.1  # TODO: https://github.com/pytorch/pytorch/issues/100974

.PHONY : install-all
install-all : install

.PHONY : update
update :
	-poetry self update
	poetry update
	-pre-commit autoupdate

.PHONY : lint
lint :
	ruff check --format=github .

.PHONY : format
format :
	black --check .

.PHONY : docformat
docformat :
	docformatter --config ./pyproject.toml --in-place $(SOURCE)

.PHONY : unit-test
unit-test :
	python -m pytest --timeout 10 $(UNIT_TESTS)

.PHONY : unit-test-cov
unit-test-cov :
	python -m pytest --timeout 10 --cov-report html --cov-report xml --cov-report term --cov=$(NAME) $(UNIT_TESTS)

.PHONY : integration-test
integration-test :
	python -m pytest --timeout 60 $(INTEGRATION_TESTS)

.PHONY : integration-test-cov
integration-test-cov :
	python -m pytest --timeout 60 --cov-report html --cov-report xml --cov-report term --cov=$(NAME) --cov-append $(INTEGRATION_TESTS)

.PHONY : test
make test : unit-test integration-test

.PHONY : test-cov
make test-cov : unit-test-cov integration-test-cov

.PHONY : publish-pypi
publish-pypi :
	poetry config pypi-token.pypi ${AROMA_PYPI_TOKEN}
	poetry publish --build
