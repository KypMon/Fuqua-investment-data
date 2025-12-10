#FROM python:3.13-alpine
FROM python:3.13-slim

# HTTP_PROXY and HTTPS_PROXY not needed inside gitlab ci/cd environment
# but needed if building via shell script
ARG HTTP_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ARG HTTPS_PROXY
ENV HTTPS_PROXY=${HTTPS_PROXY}

ARG NPM_REGISTRY
ENV NPM_REGISTRY=${NPM_REGISTRY}

ARG REACT_APP_VALIDATE_URL
ENV REACT_APP_VALIDATE_URL=${REACT_APP_VALIDATE_URL}

ARG REACT_APP_AUTH_URL
ENV REACT_APP_AUTH_URL=${REACT_APP_AUTH_URL}

ARG REACT_APP_API_BASE_URL
ENV REACT_APP_API_BASE_URL=${REACT_APP_API_BASE_URL}

ARG REACT_APP_BASENAME
ENV REACT_APP_BASENAME=${REACT_APP_BASENAME}

ARG HOME_PAGE_REDIRECT
ENV HOME_PAGE_REDIRECT=${HOME_PAGE_REDIRECT}

ARG APP_PREFIX
ENV APP_PREFIX=${APP_PREFIX}

# app will run as root and listen on container port 80 (mapped to host port 5002)
ARG PORT
ENV PORT=${PORT}
# container listens on 0.0.0.0
ARG HOST
ENV HOST=${HOST}

ARG AUTH_COOKIE_NAME
ENV AUTH_COOKIE_NAME=${AUTH_COOKIE_NAME}

ARG ISSUER
ENV ISSUER=${ISSUER}

ARG JWKS_URI
ENV JWKS_URI=${JWKS_URI}

ARG ALGORITHM
ENV ALGORITHM=${ALGORITHM}

ARG AUDIENCE
ENV AUDIENCE=${AUDIENCE}

ARG FW_LOGIN_URL
ENV FW_LOGIN_URL=${FW_LOGIN_URL}

# data files 
ARG DATA_DIRECTORY
ENV DATA_DIRECTORY=${DATA_DIRECTORY}

ARG CONFIG_DIRECTORY
ENV CONFIG_DIRECTORY=${CONFIG_DIRECTORY}

ARG F_F_MOMENTUM_FACTOR
ENV F_F_MOMENTUM_FACTOR=${F_F_MOMENTUM_FACTOR}

ARG F_F_RESEARCH_DATA_5_FACTORS_2BY3
ENV F_F_RESEARCH_DATA_5_FACTORS_2BY3=${F_F_RESEARCH_DATA_5_FACTORS_2BY3}

ARG F_F_RESEARCH_DATA_FACTORS
ENV F_F_RESEARCH_DATA_FACTORS=${F_F_RESEARCH_DATA_FACTORS}

ARG STOCKER_ETF
ENV STOCKER_ETF=${STOCKER_ETF}

ARG STATIC_DIR
ENV STATIC_DIR=${STATIC_DIR}

ARG REACT_BUILD_DIR
ENV REACT_BUILD_DIR=${REACT_BUILD_DIR}

# Install system packages for both Node and Python builds
RUN apt-get update && apt-get install -y \
    curl gnupg build-essential git \
    libffi-dev libssl-dev liblapack-dev libblas-dev rustc cargo cmake \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js LTS (e.g., 20.x or 22.x)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && node --version && npm --version

# /app/fa
ENV HOME=/app
ENV APP_HOME=fa
ENV APP_PATH=${HOME}/${APP_HOME} 

RUN mkdir -p  ${APP_PATH}
WORKDIR ${APP_PATH}

# begin react build stuff
COPY frontend ${APP_PATH}/frontend

WORKDIR ${APP_PATH}/frontend
RUN jq --arg url "$HOME_PAGE_REDIRECT" '.homepage = $url' package.json > tmp.json && mv tmp.json package.json

COPY frontend/.env_docker  ${APP_PATH}/frontend/.env
RUN sed -i "s@__react_app_api_base_url@${REACT_APP_API_BASE_URL}@g" ${APP_PATH}/frontend/.env \
    && sed -i "s@__react_app_validate_url@${REACT_APP_VALIDATE_URL}@g" ${APP_PATH}/frontend/.env \
    && sed -i "s@__react_app_auth_url@${REACT_APP_AUTH_URL}@g"         ${APP_PATH}/frontend/.env \
    && sed -i "s@__react_app_basename@${REACT_APP_BASENAME}@g"         ${APP_PATH}/frontend/.env

RUN npm config set registry=${NPM_REGISTRY} \
    && npm config set access=public \
    && npm config set strict-ssl=false \
    && npm config set scope=@fuquaschoolofbusiness \
    && npm config set @fuquaschoolofbusiness:registry=${NPM_REGISTRY} 
RUN npm install --legacy-peer-deps

RUN npm run build

# Remove Node.js and npm to slim the final image
RUN apt-get purge -y nodejs npm && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.npm /usr/lib/node_modules /usr/local/lib/node_modules
# end react build stuff

RUN mkdir -p ${APP_PATH}/backend/react_build \
    && cp -r ${APP_PATH}/frontend/build/* ${APP_PATH}/backend/react_build/

# Copy backend source
COPY backend ${APP_PATH}/backend

# Switch into backend/config so sed runs in the right spot
WORKDIR ${APP_PATH}/backend/config
COPY backend/config/.env_docker .env

RUN sed -i "s@__app_prefix@${APP_PREFIX}@g"                                                  .env \
    && sed -i "s/__port/${PORT}/g"                                                           .env \
    && sed -i "s/__host/${HOST}/g"                                                           .env \
    #
    && sed -i "s@__data_directory@${DATA_DIRECTORY}@g"                                       .env \
    && sed -i "s@__config_directory@${CONFIG_DIRECTORY}@g"                                   .env \
    && sed -i "s@__f_f_momentum_factor@${F_F_MOMENTUM_FACTOR}@g"                             .env \
    && sed -i "s@__f_f_research_data_5_factors_2by3@${F_F_RESEARCH_DATA_5_FACTORS_2BY3}@g"   .env \
    && sed -i "s@__f_f_research_data_factors@${F_F_RESEARCH_DATA_FACTORS}@g"                 .env \
    && sed -i "s@__stocker_etf@${STOCKER_ETF}@g"                                             .env \
    #
    && sed -i "s@__static_dir@${STATIC_DIR}@g"                                               .env \
    && sed -i "s@__react_build_dir@${REACT_BUILD_DIR}@g"                                     .env \
    #
    && sed -i "s/__auth_cookie_name/${AUTH_COOKIE_NAME}/g"                                   .env \
    && sed -i "s@__issuer@${ISSUER}@g"                                                       .env \
    && sed -i "s@__jwks_uri@${JWKS_URI}@g"                                                   .env \
    && sed -i "s/__algorithm/${ALGORITHM}/g"                                                 .env \
    && sed -i "s/__audience/${AUDIENCE}/g"                                                   .env \
    && sed -i "s@__fw_login_url@${FW_LOGIN_URL}@g"                                           .env \
    && sed -i "s@__home_page_redirect@${HOME_PAGE_REDIRECT}@g"                               .env

EXPOSE 80

ENV VIRTUAL_ENV=${APP_PATH}/backend/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m venv ${APP_PATH}/backend/.venv

# Upgrade pip tools first (best practice)
RUN ${APP_PATH}/backend/.venv/bin/pip install --upgrade pip setuptools wheel

# Upgrade core SSL-related dependencies
RUN ${APP_PATH}/backend/.venv/bin/pip install --upgrade requests urllib3 cryptography certifi

RUN ${APP_PATH}/backend/.venv/bin/pip install  --no-cache-dir   -r ${APP_PATH}/backend/requirements.txt

# WORKDIR ${APP_PATH}
RUN chmod +x ${APP_PATH}/backend/go.sh
#ENTRYPOINT ["tail", "-f", "/dev/null"]
#CMD ["/bin/sh", "/app/fa/backend/go.sh"]
CMD ["/bin/bash", "/app/fa/backend/go.sh"]


