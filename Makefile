.PHONY:test
test:
	pytest --cov=ml_video_metrics --cov-report=html tests/

.PHONY:lint
lint:
	black ./
	autopep8 --in-place --aggressive -r ./