.PHONY: install test lint run

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	python -m unittest discover -s tests -p "test_*.py"

run:
	media-auth-forensics infer --input ./samples/image.jpg --out report.json
