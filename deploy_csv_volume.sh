#!/bin/bash
# generate the docker volume that holds the 4 input CSV files:
# F-F_Momentum_Factor.csv
# F-F_Research_Data_5_Factors_2x3.csv
# F-F_Research_Data_Factors.CSV
# stocks_mf_ETF_data_final.csv
#
# We keep these files on a separate volume, because they are not source-controlled in gitlab.
# This creates a problem when doing an auto-deployment via .gitlab-ci.yml.
# The solution is to maintain this volume separately.
# When any content changes, we must stop the running application, then edit/run this script to
#  rebuild this volume, then trigger .gitlab-ci.yml to run (this volume will be attached).

VOLUME=fa_volume_csv_input

docker volume rm $VOLUME 2>/dev/null || true

docker volume create --driver local $VOLUME;
docker run --rm -v $VOLUME:/data alpine mkdir -p /data/input;

# Copy the files into the volume
docker container create --name temp-container -v $VOLUME:/data alpine
docker cp backend/data/F-F_Momentum_Factor.csv temp-container:/data/input
docker cp backend/data/F-F_Research_Data_5_Factors_2x3.csv temp-container:/data/input
docker cp backend/data/F-F_Research_Data_Factors.CSV temp-container:/data/input
docker cp backend/data/stocks_mf_ETF_data_final.csv temp-container:/data/input

# Change ownership and permissions inside the volume
docker run --rm -v $VOLUME:/data alpine sh -c \
" \
chown root:root /data/input/*.csv && chmod 644 /data/input/*.csv \
  &&  chown root:root /data/input/*.CSV && chmod 644 /data/input/*.CSV \
"


# inspect
# docker run --rm -it -v fa_volume_csv_input:/data alpine sh

docker rm temp-container




