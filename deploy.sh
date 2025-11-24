#!/bin/bash


NODE_CONTAINER=fa
NODE_IMAGE=financial_analyzer_image

docker stop $NODE_CONTAINER 2>/dev/null || true
docker rm $NODE_CONTAINER 2>/dev/null || true

docker rmi -f $NODE_IMAGE 2>/dev/null || true

docker build  \
-f fa.Dockerfile \
--build-arg http_proxy=$HTTP_PROXY \
--build-arg https_proxy=$HTTPS_PROXY \
--build-arg NPM_REGISTRY="https://beergame.vm.duke.edu:4873/" \
-t financial_analyzer_image \
.

docker run -it -w /fa --entrypoint /bin/sh $NODE_IMAGE