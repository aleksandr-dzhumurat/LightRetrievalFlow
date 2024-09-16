CURRENT_DIR = $(shell pwd)
NETWORK_NAME = service_network
PROJECT_NAME = light_retrieval_flow
 
include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data/zinc_data || true && \
	mkdir -p ${CURRENT_DIR}/data/pipelines_data || true && \
	mkdir -p ${CURRENT_DIR}/data/pipelines_data/models || true

run-zinc:
	docker run -it --rm --network backtier \
	-v ${CURRENT_DIR}/data/zinc_data:/data \
	-p 4080:4080 \
	-e ZINC_DATA_PATH="/data" \
	-e ZINC_FIRST_ADMIN_USER=admin \
	-e ZINC_FIRST_ADMIN_PASSWORD=admin \
	--name zincsearch public.ecr.aws/zinclabs/zincsearch:0.4.10

run-jupyter:
	source ~/.pyenv/versions/ltr-research/bin/activate  && \
	\
	PYTHONPATH=${CURRENT_DIR}/src \
	DATA_DIR=${CURRENT_DIR}/data  \
	CONFIG_DIR=${CURRENT_DIR}  \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 9999 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser

prepare-embeds:
	PYTHONPATH=${CURRENT_DIR}/src \
	DATA_DIR=${CURRENT_DIR}/data  \
	CONFIG_DIR=${CURRENT_DIR}  \
	python3 src/light_retrieval_flow/vectorize.py

inference:
	PYTHONPATH=${CURRENT_DIR}/src \
	DATA_DIR=${CURRENT_DIR}/data  \
	CONFIG_DIR=${CURRENT_DIR}  \
	python3 src/light_retrieval_flow/inference.py

build-vector-search:
	docker build -f services/vector_search/Dockerfile -t adzhumurat/${PROJECT_NAME}_vector_search:latest .

prepare-onnx:
	docker run --rm \
		--env-file ${CURRENT_DIR}/services/vector_search/.env  \
	    -v "${CURRENT_DIR}/services/vector_search/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name ${PROJECT_NAME}_container_onnx \
		adzhumurat/${PROJECT_NAME}_vector_search:latest \
		python3 src/prepare_onnx.py

serve:
	docker run --rm \
		--env-file ${CURRENT_DIR}/services/vector_search/.env  \
	    -v "${CURRENT_DIR}/services/vector_search/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name ${PROJECT_NAME}_container_onnx \
		adzhumurat/${PROJECT_NAME}_vector_search:latest \
		serve