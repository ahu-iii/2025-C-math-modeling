PYTHON := .venv/bin/python

all:
	$(PYTHON) code/run_all.py

check:
	$(PYTHON) code/run_all.py
	git diff --exit-code -- output/
