# So that our final FA image is smaller,
# we complete the node build stage first (React frontend artifacts). 
FROM node:24-alpine AS node-build

RUN npm install -g npm@10.9.2
RUN npm --version

# Duke FSB NPM registry
ARG NPM_REGISTRY
ENV NPM_REGISTRY=${NPM_REGISTRY}

RUN mkdir -p /fa
WORKDIR /fa
#COPY frontend/package.json ./
COPY frontend/ /fa

RUN npm config set registry=${NPM_REGISTRY}
RUN npm config set access=public
RUN npm config set strict-ssl=false
RUN npm config set scope=@fuquaschoolofbusiness
RUN npm config set @fuquaschoolofbusiness:registry=${NPM_REGISTRY}
RUN npm install --legacy-peer-deps

RUN npm run build





#FROM python:3.13-alpine

# ## the ENV environment variable exists for backward compatibility reasons 
# ##  (we want Geetha to be able to continue to develop in her local, non-Docker environment)
# ARG ENV

# ## need the proxy URL to successfully build the image (when outside of gitlab ci/cd environment)
# ARG http_proxy
# ARG https_proxy

# # Oracle database access
# ARG DB_USER
# ARG DB_PWD
# ARG DB_DSN

# # app will run as root and listen on container port 80 (mapped to host port 5001)
# ARG PORT
# # container listens on 0.0.0.0
# ARG HOST

# # for authentication
# ARG AUTH_COOKIE_NAME
# ARG ISSUER
# ARG JWKS_URI
# ARG ALGORITHM
# ARG AUDIENCE
# ARG FW_LOGIN_URL
# ARG HOME_PAGE_REDIRECT

# # email support
# ARG MAIL_SERVER
# ARG MAIL_PORT
# ARG MAIL_USERNAME
# #ARG MAIL_PASSWORD
# ARG EMAIL_TESTER

# # application name in Oracle CMLEDB.APP_ACCESS (column name is APP_NAME)
# ARG APP_NAME

# ENV ENV=${ENV}
# ENV DB_USER=${DB_USER}
# ENV DB_PWD=${DB_PWD}
# ENV DB_DSN=${DB_DSN}
# ENV PORT=${PORT}
# ENV HOST=${HOST}
# ENV AUTH_COOKIE_NAME=${AUTH_COOKIE_NAME}
# ENV ISSUER=${ISSUER}
# ENV JWKS_URI=${JWKS_URI}
# ENV ALGORITHM=${ALGORITHM}
# ENV AUDIENCE=${AUDIENCE}
# ENV FW_LOGIN_URL=${FW_LOGIN_URL}
# ENV HOME_PAGE_REDIRECT=${HOME_PAGE_REDIRECT}
# ENV MAIL_SERVER=${MAIL_SERVER}
# ENV MAIL_PORT=${MAIL_PORT}
# ENV MAIL_USERNAME=${MAIL_USERNAME}
# ENV EMAIL_TESTER=${EMAIL_TESTER}
# ENV APP_NAME=${APP_NAME}

# ENV HOME=/app

# # /app/ais_coursepacks
# ENV APP_HOME=ais_coursepacks
# ENV APP_PATH=${HOME}/${APP_HOME} 

# #RUN apt-get update && \
# #    apt-get install -y  python3.12  python3.12-venv  python3-pip  && \
# #    rm -rf /var/lib/apt/lists/*

# RUN apk add  --no-cache  gcc musl-dev python3-dev libffi-dev openssl-dev make

# RUN mkdir -p ${APP_PATH}
# RUN mkdir -p ${APP_PATH}/src  
# RUN mkdir -p ${APP_PATH}/resources  
# RUN mkdir -p ${APP_PATH}/static
# RUN mkdir -p ${APP_PATH}/templates
# RUN mkdir -p ${APP_PATH}/log

# WORKDIR ${APP_PATH}

# COPY app.py ./
# COPY requirements.txt ./
# COPY gunicorn.conf.py ./
# COPY wsgi.py ./
# COPY src/ ${APP_PATH}/src
# COPY resources/.env_docker ${APP_PATH}/resources/.env
# #COPY static/ ${APP_PATH}/static
# COPY static/css/  ${APP_PATH}/static/css
# COPY static/js/  ${APP_PATH}/static/js
# COPY --from=node-build  /coursepacks/lib  ${APP_PATH}/static/lib
# COPY --from=node-build  /coursepacks/node_modules  ${APP_PATH}/static/node_modules
# COPY templates/ ${APP_PATH}/templates
# COPY go.sh ./

# RUN chmod +x ${APP_PATH}/*.sh

# #WORKDIR ${APP_PATH}/static
# #RUN tar -xvf lib.tar -C .  

# RUN sed -i "s/__env/${ENV}/g"                               ${APP_PATH}/resources/.env
# RUN sed -i "s/__db_user/${DB_USER}/g"                       ${APP_PATH}/resources/.env
# RUN sed -i "s/__db_pwd/${DB_PWD}/g"                         ${APP_PATH}/resources/.env
# RUN sed -i "s@__db_dsn@${DB_DSN}@g"                         ${APP_PATH}/resources/.env
# RUN sed -i "s/__port/${PORT}/g"                             ${APP_PATH}/resources/.env
# RUN sed -i "s/__host/${HOST}/g"                             ${APP_PATH}/resources/.env
# RUN sed -i "s/__auth_cookie_name/${AUTH_COOKIE_NAME}/g"     ${APP_PATH}/resources/.env
# RUN sed -i "s@__issuer@${ISSUER}@g"                         ${APP_PATH}/resources/.env
# RUN sed -i "s@__jwks_uri@${JWKS_URI}@g"                     ${APP_PATH}/resources/.env
# RUN sed -i "s/__algorithm/${ALGORITHM}/g"                   ${APP_PATH}/resources/.env
# RUN sed -i "s/__audience/${AUDIENCE}/g"                     ${APP_PATH}/resources/.env
# RUN sed -i "s@__fw_login_url@${FW_LOGIN_URL}@g"             ${APP_PATH}/resources/.env
# RUN sed -i "s@__home_page_redirect@${HOME_PAGE_REDIRECT}@g" ${APP_PATH}/resources/.env
# RUN sed -i "s/__mail_server/${MAIL_SERVER}/g"               ${APP_PATH}/resources/.env
# RUN sed -i "s/__mail_port/${MAIL_PORT}/g"                   ${APP_PATH}/resources/.env
# RUN sed -i "s/__mail_username/${MAIL_USERNAME}/g"           ${APP_PATH}/resources/.env
# RUN sed -i "s/__mail_password//g"                           ${APP_PATH}/resources/.env
# RUN sed -i "s/__email_tester/${EMAIL_TESTER}/g"             ${APP_PATH}/resources/.env
# RUN sed -i "s/__app_name/${APP_NAME}/g"                     ${APP_PATH}/resources/.env
# RUN sed -i "s@__app_path@${APP_PATH}@g"                     ${APP_PATH}/resources/.env

# EXPOSE 80

# ENV VIRTUAL_ENV=${APP_PATH}/.venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# RUN python3 -m venv ${APP_PATH}/.venv
# RUN ${APP_PATH}/.venv/bin/pip install  --no-cache-dir   -r ${APP_PATH}/requirements.txt

# WORKDIR ${APP_PATH}

# #ENTRYPOINT ["tail", "-f", "/dev/null"]
# CMD ["/bin/sh", "go.sh"]


