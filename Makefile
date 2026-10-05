.PHONY: install ingest ask eval

install:
	pip install -r requirements.txt

ingest:
	python run.py ingest

ask:
	@if [ -z "$(Q)" ]; then \
		echo "Error: Please provide a question using Q=\"your question\""; \
	else \
		python run.py ask "$(Q)"; \
	fi

eval:
	python run.py eval