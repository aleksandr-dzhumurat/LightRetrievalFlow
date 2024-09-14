CURRENT_DIR = $(shell pwd)
NETWORK_NAME = service_network
 
include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data/zinc_data || true

run-zinc:
	docker run -it --rm --network backtier \
	-v ${CURRENT_DIR}/data/zinc_data:/data \
	-p 4080:4080 \
	-e ZINC_DATA_PATH="/data" \
	-e ZINC_FIRST_ADMIN_USER=admin \
	-e ZINC_FIRST_ADMIN_PASSWORD=admin \
	--name zincsearch public.ecr.aws/zinclabs/zincsearch:0.4.10

run-jupyter:
	PYTHONPATH=${CURRENT_DIR}/src \
	DATA_DIR=${CURRENT_DIR}/data  \
	python3 src/sweed_rnd/metrics/marketing.py \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8899 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 