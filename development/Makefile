SYSTEM_PYTHON ?= python3
VENV_DIR ?= .venv
PYTHON ?= $(if $(wildcard $(VENV_DIR)/bin/python),$(VENV_DIR)/bin/python,$(SYSTEM_PYTHON))
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup env-check build test smoke check

setup:
	bash scripts/setup/bootstrap_env.sh

env-check:
	$(PYTHON) scripts/setup/check_environment.py --requirements requirements.txt

build:
	$(PYTHON) -m compileall -q src scripts tests

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) scripts/submission/smoke_contract.py

check: env-check build test smoke
