#!/bin/bash

echo "$(date) Financial Analyzer is running...."
## Set up proxy host
export HTTP_PROXY=http://ha-proxy.fuqua.duke.edu:3128
export HTTPS_PROXY=$HTTP_PROXY

#APP_HOME=/app/financial_analyzer
APP_HOME=/app/fa/backend

cd $APP_HOME
export PYTHONPATH=$APP_HOME

#export APP_CONFIG_FILE="$APP_HOME/config/.env"
#export APP_LOG_FILE="$APP_HOME/log/app.log"

VIRTUAL_ENV=${APP_HOME}/.venv
PATH="$VIRTUAL_ENV/bin:$PATH"
#$VIRTUAL_ENV/bin/python  $APP_HOME/app.py   --log=$LOG_FILE  --config=$CONFIG_FILE
#$VIRTUAL_ENV/bin/python  $APP_HOME/app.py
#$VIRTUAL_ENV/bin/gunicorn --workers=4   --bind 0.0.0.0:80    --access-logfile log/gunicorn.log    --error-logfile log/financial_analyzer.log  wsgi:app
$VIRTUAL_ENV/bin/gunicorn -c $APP_HOME/gunicorn.conf.py
