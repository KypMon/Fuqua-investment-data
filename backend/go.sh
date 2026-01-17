#!/bin/bash

echo "$(date) Financial Analyzer is running...."

## Set up proxy host
export HTTP_PROXY=http://ha-proxy.fuqua.duke.edu:3128
export HTTPS_PROXY=$HTTP_PROXY

# --------------------------------------------
# Ensure static volume structure exists
# --------------------------------------------
# Initialize /static subdirectories individually if missing
for dir in /static/matrix /static/lifecycle /static/regression; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory $dir ..."
        mkdir -p "$dir/tokens"
    fi
done

# --------------------------------------------
# Validate /data/input CSV volume and required files
# --------------------------------------------
DATA_DIR="/data/input"
REQUIRED_FILES=(
    "F-F_Momentum_Factor.csv"
    "F-F_Research_Data_5_Factors_2x3.csv"
    "F-F_Research_Data_Factors.CSV"
    "stocks_mf_ETF_data_final.csv"
)

echo "Checking read-only data volume at ${DATA_DIR} ..."
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: expected volume mounted at ${DATA_DIR}, but directory does not exist."
    echo "Did you forget to pass '-v fa_volume_csv_input:/data' to docker run?"
    exit 1
fi

for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "Error: required file missing: ${DATA_DIR}/${f}"
        echo "Ensure your CSV input volume is correctly mounted and populated."
        exit 1
    fi
done

echo "All required CSV input files are present."

# --------------------------------------------
# Start the gunicorn server
# --------------------------------------------
APP_HOME=/app/fa/backend

cd $APP_HOME
export PYTHONPATH=$APP_HOME

VIRTUAL_ENV=${APP_HOME}/.venv
PATH="$VIRTUAL_ENV/bin:$PATH"
#$VIRTUAL_ENV/bin/python  $APP_HOME/app.py
#$VIRTUAL_ENV/bin/gunicorn --workers=4   --bind 0.0.0.0:80    --access-logfile log/gunicorn.log    --error-logfile log/financial_analyzer.log  wsgi:app
$VIRTUAL_ENV/bin/gunicorn -c $APP_HOME/gunicorn.conf.py
