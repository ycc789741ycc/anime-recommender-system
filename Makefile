test:
	python -m pytest

build:
	rm -R -f dist/
	poetry version $(BUILD_VERSION)
	poetry build
	tar -xvf dist/*.tar.gz --wildcards --no-anchored '*/setup.py' --strip=1
	poetry export -f requirements.txt --output requirements.txt

publish:
	poetry publish
