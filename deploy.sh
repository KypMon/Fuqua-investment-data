#!/bin/bash

docker build  \
-f fs.Dockerfile \
--build-arg http_proxy=$HTTP_PROXY \
--build-arg https_proxy=$HTTPS_PROXY \
--build-arg NPM_REGISTRY="https://beergame.vm.duke.edu:4873/"
-t financial_analyzer_image \
.