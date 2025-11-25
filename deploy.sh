#!/bin/bash

HOST=0.0.0.0
PORT=5002
IMAGE=financial_analyzer_image
CONTAINER=fa
NPM_REGISTRY=https://beergame.vm.duke.edu:4873/
AUTH_COOKIE_NAME=_FSB_G
ISSUER=https://go-dev.fuqua.duke.edu/auth
JWKS_URI=https://go-dev.fuqua.duke.edu/auth/jwks
ALGORITHM=RS256
AUDIENCE=FuquaWorld
FW_LOGIN_URL=https://authdev.fuqua.duke.edu/auth/duke?service=
HOME_PAGE_REDIRECT=http://go-dev.fuqua.duke.edu/financial_analyzer/
DATA_DIRECTORY=data
CONFIG_DIRECTORY=config
F_F_MOMENTUM_FACTOR=F-F_Momentum_Factor.csv
F_F_RESEARCH_DATA_5_FACTORS_2BY3=F-F_Research_Data_5_Factors_2x3.csv
F_F_RESEARCH_DATA_FACTORS=F-F_Research_Data_Factors.CSV
STOCKER_ETF=stocks_mf_ETF_data_final.csv
STATIC_DIR=/data/static
REACT_BUILD_DIR=react_build
VOLUME=fa_volume

docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

docker volume rm $VOLUME 2>/dev/null || true

docker rmi -f $IMAGE 2>/dev/null || true

docker build  \
-f fa.Dockerfile \
--build-arg http_proxy=$HTTP_PROXY \
--build-arg https_proxy=$HTTPS_PROXY \
--build-arg NPM_REGISTRY=$NPM_REGISTRY \
--build-arg PORT=$PORT \
--build-arg HOST=$HOST \
--build-arg AUTH_COOKIE_NAME=${AUTH_COOKIE_NAME} \
--build-arg ISSUER=${ISSUER} \
--build-arg JWKS_URI=${JWKS_URI} \
--build-arg ALGORITHM=${ALGORITHM} \
--build-arg AUDIENCE=${AUDIENCE} \
--build-arg FW_LOGIN_URL=${FW_LOGIN_URL} \
--build-arg HOME_PAGE_REDIRECT=${HOME_PAGE_REDIRECT} \
--build-arg DATA_DIRECTORY=${DATA_DIRECTORY} \
--build-arg CONFIG_DIRECTORY=${CONFIG_DIRECTORY} \
--build-arg F_F_MOMENTUM_FACTOR=${F_F_MOMENTUM_FACTOR} \
--build-arg F_F_RESEARCH_DATA_5_FACTORS_2BY3=${F_F_RESEARCH_DATA_5_FACTORS_2BY3} \
--build-arg F_F_RESEARCH_DATA_FACTORS=${F_F_RESEARCH_DATA_FACTORS} \
--build-arg STOCKER_ETF=${STOCKER_ETF} \
--build-arg STATIC_DIR=${STATIC_DIR} \
--build-arg REACT_BUILD_DIR=${REACT_BUILD_DIR} \
-t $IMAGE \
.

docker volume create --driver local $VOLUME;
docker run --rm -v $VOLUME:/data alpine mkdir -p /data/logs  /data/static;

#docker run -it -w /fa --entrypoint /bin/sh $NODE_IMAGE
#docker run -it -w /app/fa --entrypoint /bin/sh $IMAGE

docker run  --detach  -v $VOLUME:/data  --name fa  $IMAGE