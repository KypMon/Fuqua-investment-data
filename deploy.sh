#!/bin/bash

HOST=0.0.0.0
PORT=5001
#NODE_CONTAINER=fa
#NODE_IMAGE=fa_node_image
IMAGE=financial_analyzer_image
NPM_REGISTRY=https://beergame.vm.duke.edu:4873/

# docker stop $NODE_CONTAINER 2>/dev/null || true
# docker rm $NODE_CONTAINER 2>/dev/null || true

# docker rmi -f $NODE_IMAGE 2>/dev/null || true

docker build  \
-f fa.Dockerfile \
--build-arg http_proxy=$HTTP_PROXY \
--build-arg https_proxy=$HTTPS_PROXY \
--build-arg NPM_REGISTRY=$NPM_REGISTRY \
--build-arg PORT=$PORT \
--build-arg HOST=$HOST \
-t $IMAGE \
.

#docker run -it -w /fa --entrypoint /bin/sh $NODE_IMAGE
docker run -it -w /fa --entrypoint /bin/sh $IMAGE