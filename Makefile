REG=localhost:5000
TF_VER=1.13.2-gpu-py3-jupyter
VER=0.0.1
IMG=$(REG)/tf/tf:$(VER)

dr=docker run --rm -ti

test:
	docker run --rm -ti --gpus all $(IMG) \
	python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([10000, 10000])))"

ci:
	docker build -f ci.dockerfile -t $(IMG) .


push pull:
	docker $@ $(IMG)

bash:
	$(dr) -p 8888:80 $(IMG) bash
run:
	docker run --gpus all $(IMG)

run-lab:
	$(dr) --gpus all \
	-w /srv \
	-v `pwd`/src:/srv \
	-v `pwd`/data:/srv/data \
	-p 8888:80 \
	$(IMG) \
	jupyter lab --ip=0.0.0.0 --port=80 --no-browser --notebook-dir=/srv --allow-root --LabApp.token=''