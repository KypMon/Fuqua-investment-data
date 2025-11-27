# So that our final FA image is smaller,
# we complete the node build stage first (React frontend artifacts). 
FROM node:24-alpine AS node-build

RUN npm install -g npm@10.9.2
RUN npm --version

# Duke FSB NPM registry
ARG NPM_REGISTRY
ENV NPM_REGISTRY=${NPM_REGISTRY}

ARG REACT_APP_VALIDATE_URL
ENV REACT_APP_VALIDATE_URL=${REACT_APP_VALIDATE_URL}

ARG REACT_APP_AUTH_URL
ENV REACT_APP_AUTH_URL=${REACT_APP_AUTH_URL}

ARG REACT_APP_BASE_URL
ENV REACT_APP_BASE_URL=${REACT_APP_BASE_URL}

RUN mkdir -p /fa
WORKDIR /fa
COPY frontend/ /fa

# we do this so static resources will properly be served in non-localhost environment
RUN sed -i "s@/appstatic@/financial_analyzer/appstatic@g" /fa/package.json 

RUN npm config set registry=${NPM_REGISTRY}
RUN npm config set access=public
RUN npm config set strict-ssl=false
RUN npm config set scope=@fuquaschoolofbusiness
RUN npm config set @fuquaschoolofbusiness:registry=${NPM_REGISTRY}
RUN npm install --legacy-peer-deps

RUN npm run build


FROM python:3.13-alpine

# ## the ENV environment variable exists for backward compatibility reasons 
# ##  (we want Geetha to be able to continue to develop in her local, non-Docker environment)
# ARG ENV

# ## need the proxy URL to successfully build the image (when outside of gitlab ci/cd environment)
# ARG http_proxy
# ARG https_proxy

# app will run as root and listen on container port 80 (mapped to host port 5002)
ARG PORT
# container listens on 0.0.0.0
ARG HOST

# for authentication
ARG AUTH_COOKIE_NAME
ARG ISSUER
ARG JWKS_URI
ARG ALGORITHM
ARG AUDIENCE
ARG FW_LOGIN_URL
ARG HOME_PAGE_REDIRECT

# data files (not sure if I will need to rework this for volumes)
ARG DATA_DIRECTORY
ARG CONFIG_DIRECTORY
ARG F_F_MOMENTUM_FACTOR
ARG F_F_RESEARCH_DATA_5_FACTORS_2BY3
ARG F_F_RESEARCH_DATA_FACTORS
ARG STOCKER_ETF

# for user uploads and downloads (not sure if I will need to rework this for volumes)
ARG STATIC_DIR

# location for React artifacts
ARG REACT_BUILD_DIR

ENV PORT=${PORT}
ENV HOST=${HOST}
ENV AUTH_COOKIE_NAME=${AUTH_COOKIE_NAME}
ENV ISSUER=${ISSUER}
ENV JWKS_URI=${JWKS_URI}
ENV ALGORITHM=${ALGORITHM}
ENV AUDIENCE=${AUDIENCE}
ENV FW_LOGIN_URL=${FW_LOGIN_URL}
ENV HOME_PAGE_REDIRECT=${HOME_PAGE_REDIRECT}
ENV DATA_DIRECTORY=${DATA_DIRECTORY}
ENV CONFIG_DIRECTORY=${CONFIG_DIRECTORY}
ENV F_F_MOMENTUM_FACTOR=${F_F_MOMENTUM_FACTOR}
ENV F_F_RESEARCH_DATA_5_FACTORS_2BY3=${F_F_RESEARCH_DATA_5_FACTORS_2BY3}
ENV F_F_RESEARCH_DATA_FACTORS=${F_F_RESEARCH_DATA_FACTORS}
ENV STOCKER_ETF=${STOCKER_ETF}
ENV STATIC_DIR=${STATIC_DIR}
ENV REACT_BUILD_DIR=${REACT_BUILD_DIR}

#RUN apk add  --no-cache  gcc g++ musl-dev python3-dev libffi-dev openssl-dev make
#RUN apk add --no-cache \
#    gcc g++ musl-dev python3-dev libffi-dev openssl-dev make \
#    lapack-dev blas-dev rust cargo
RUN apk add --no-cache \
    gcc g++ musl-dev python3-dev libffi-dev openssl-dev make \
    lapack-dev blas-dev rust cargo git cmake

# /app/fa
ENV HOME=/app
ENV APP_HOME=fa
ENV APP_PATH=${HOME}/${APP_HOME} 

RUN mkdir -p  ${APP_PATH}
WORKDIR ${APP_PATH}
COPY backend ${APP_PATH}/backend
COPY --from=node-build /fa/build  ${APP_PATH}/backend/react_build
COPY backend/config/.env_docker   backend/config/.env

RUN sed -i "s/__port/${PORT}/g"                                                           ${APP_PATH}/backend/config/.env
RUN sed -i "s/__host/${HOST}/g"                                                           ${APP_PATH}/backend/config/.env
#
RUN sed -i "s@__data_directory@${DATA_DIRECTORY}@g"                                       ${APP_PATH}/backend/config/.env
RUN sed -i "s@__config_directory@${CONFIG_DIRECTORY}@g"                                   ${APP_PATH}/backend/config/.env
RUN sed -i "s@__f_f_momentum_factor@${F_F_MOMENTUM_FACTOR}@g"                             ${APP_PATH}/backend/config/.env
RUN sed -i "s@__f_f_research_data_5_factors_2by3@${F_F_RESEARCH_DATA_5_FACTORS_2BY3}@g"   ${APP_PATH}/backend/config/.env
RUN sed -i "s@__f_f_research_data_factors@${F_F_RESEARCH_DATA_FACTORS}@g"                 ${APP_PATH}/backend/config/.env
RUN sed -i "s@__stocker_etf@${STOCKER_ETF}@g"                                             ${APP_PATH}/backend/config/.env
#
RUN sed -i "s@__static_dir@${STATIC_DIR}@g"                                               ${APP_PATH}/backend/config/.env
RUN sed -i "s@__react_build_dir@${REACT_BUILD_DIR}@g"                                     ${APP_PATH}/backend/config/.env
#
RUN sed -i "s/__auth_cookie_name/${AUTH_COOKIE_NAME}/g"                                   ${APP_PATH}/backend/config/.env
RUN sed -i "s@__issuer@${ISSUER}@g"                                                       ${APP_PATH}/backend/config/.env
RUN sed -i "s@__jwks_uri@${JWKS_URI}@g"                                                   ${APP_PATH}/backend/config/.env
RUN sed -i "s/__algorithm/${ALGORITHM}/g"                                                 ${APP_PATH}/backend/config/.env
RUN sed -i "s/__audience/${AUDIENCE}/g"                                                   ${APP_PATH}/backend/config/.env
RUN sed -i "s@__fw_login_url@${FW_LOGIN_URL}@g"                                           ${APP_PATH}/backend/config/.env
RUN sed -i "s@__home_page_redirect@${HOME_PAGE_REDIRECT}@g"                               ${APP_PATH}/backend/config/.env

EXPOSE 80

ENV VIRTUAL_ENV=${APP_PATH}/backend/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m venv ${APP_PATH}/backend/.venv
RUN ${APP_PATH}/backend/.venv/bin/pip install  --no-cache-dir   -r ${APP_PATH}/backend/requirements.txt

# WORKDIR ${APP_PATH}
RUN chmod +x ${APP_PATH}/backend/go.sh
#ENTRYPOINT ["tail", "-f", "/dev/null"]
CMD ["/bin/sh", "/app/fa/backend/go.sh"]


