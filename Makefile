PYTHON := .venv/bin/python

all:
	$(PYTHON) code/run_all.py

check:
	$(PYTHON) code/run_all.py
	$(PYTHON) code/check_paper_numbers.py
	git diff --exit-code -- output/
