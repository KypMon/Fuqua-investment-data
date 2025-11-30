#!/bin/bash
#HTTP_PROXY=http://ha-proxy.fuqua.duke.edu:3128
HTTP_PROXY=http://proxy.fuqua.duke.edu:3128
HTTPS_PROXY=$HTTP_PROXY
APP_PREFIX=/financial_analyzer
HOST=0.0.0.0
HOST_PORT=5002
CONTAINER_PORT=80
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
DATA_DIRECTORY=/data/input
CONFIG_DIRECTORY=config
F_F_MOMENTUM_FACTOR=F-F_Momentum_Factor.csv
F_F_RESEARCH_DATA_5_FACTORS_2BY3=F-F_Research_Data_5_Factors_2x3.csv
F_F_RESEARCH_DATA_FACTORS=F-F_Research_Data_Factors.CSV
STOCKER_ETF=stocks_mf_ETF_data_final.csv
#STATIC_DIR=/data/static
STATIC_DIR=/static
REACT_BUILD_DIR=react_build
VOLUME_UPLOAD_DOWNLOAD=fa_volume_upload_download   # See deploy_upload_download_volume.sh
VOLUME_CSV_INPUT=fa_volume_csv_input  # read-only!  See deploy_csv_volume.sh
#
REACT_APP_API_BASE_URL=http://go-dev.fuqua.duke.edu:5002/financial_analyzer
#REACT_APP_API_BASE_URL=http://go-dev.fuqua.duke.edu:5002
REACT_APP_VALIDATE_URL=https://authdev.fuqua.duke.edu/auth/getjwt
REACT_APP_AUTH_URL=https://authdev.fuqua.duke.edu/auth/duke?service=

docker stop $CONTAINER 2>/dev/null || true
docker rm $CONTAINER 2>/dev/null || true

# docker volume rm $VOLUME_UPLOADS_DOWNLOADS 2>/dev/null || true  NO ! ! ! !

docker rmi -f $IMAGE 2>/dev/null || true

docker build  \
-f fa.Dockerfile \
--build-arg APP_PREFIX=$APP_PREFIX \
--build-arg HTTP_PROXY=$HTTP_PROXY \
--build-arg HTTPS_PROXY=$HTTPS_PROXY \
--build-arg NPM_REGISTRY=$NPM_REGISTRY \
--build-arg PORT=$CONTAINER_PORT \
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
--build-arg REACT_APP_BASE_URL=${REACT_APP_BASE_URL} \
--build-arg REACT_APP_AUTH_URL=${REACT_APP_AUTH_URL} \
--build-arg REACT_APP_VALIDATE_URL=${REACT_APP_VALIDATE_URL} \
--no-cache \
-t $IMAGE \
.


#docker run -it -w /fa --entrypoint /bin/sh $NODE_IMAGE
#docker run -it -w /app/fa --entrypoint /bin/sh $IMAGE

docker run  --detach  \
-p ${HOST_PORT}:${CONTAINER_PORT} \
-v $VOLUME_UPLOAD_DOWNLOAD:/static \
-v $VOLUME_CSV_INPUT:/data \
--name $CONTAINER  $IMAGE
