# ---- Stage 1: build React frontend ----
FROM node:24-alpine AS node-build

# Build arguments for your private registry if needed
ARG NPM_REGISTRY
ENV NPM_REGISTRY=${NPM_REGISTRY}

WORKDIR /frontend
COPY ../frontend/package*.json ./   # adjust this path if your React project is elsewhere

# Optional: configure registry the same way as your other Dockerfile
RUN if [ -n "$NPM_REGISTRY" ]; then \
    npm config set registry=${NPM_REGISTRY}; \
    fi

RUN npm install
COPY ../frontend/ ./                 # copy the rest of the React source
RUN npm run build                    # create /frontend/build

# ---- Stage 2: Flask backend ----
FROM python:3.12-alpine

# Environment / proxy / DB / auth args (same pattern you already use)
ARG http_proxy
ARG https_proxy

# Flask & app details
ARG PORT=5001
ARG HOST=0.0.0.0

ENV PORT=${PORT}
ENV HOST=${HOST}

# same environment structure as your other project
ENV HOME=/app
ENV APP_HOME=finance_analyzer
ENV APP_PATH=${HOME}/${APP_HOME}

RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev openssl-dev make

# create directories
RUN mkdir -p ${APP_PATH}/src ${APP_PATH}/react_build ${APP_PATH}/static ${APP_PATH}/log

WORKDIR ${APP_PATH}

# Copy backend code
COPY app.py requirements.txt gunicorn.conf.py wsgi.py ./   # include any that exist
COPY src/ ./src
COPY static/ ./static

# --- Copy React build output from Node stage ---
COPY --from=node-build /frontend/build/ ./react_build

# Install dependencies
RUN python3 -m venv ${APP_PATH}/.venv
ENV PATH="${APP_PATH}/.venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

# optional: use Gunicorn instead of flask run
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]