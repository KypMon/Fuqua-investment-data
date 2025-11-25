#!/bin/bash
# generate the docker volume that holds the 4 input CSV files:
# F-F_Momentum_Factor.csv
# F-F_Research_Data_5_Factors_2x3.csv
# F-F_Research_Data_Factors.CSV
# stocks_mf_ETF_data_final.csv

VOLUME=fa_volume_csv_input

docker volume rm $VOLUME 2>/dev/null || true

docker volume create --driver local $VOLUME;
docker run --rm -v $VOLUME:/data alpine mkdir -p /data/input;

docker container create --name temp-container -v $VOLUME:/data alpine
docker cp backend/data/F-F_Momentum_Factor.csv temp-container:/data/input
docker cp backend/data/F-F_Research_Data_5_Factors_2x3.csv temp-container:/data/input
docker cp backend/data/F-F_Research_Data_Factors.CSV temp-container:/data/input
docker cp backend/data/stocks_mf_ETF_data_final.csv temp-container:/data/input

# Change ownership and permissions inside the volume
#docker start temp-container
#docker exec temp-container chown root:root /data/input/*.csv
#docker exec temp-container chmod 644 /data/input/*.csv
docker run --rm -v $VOLUME:/data alpine sh -c \
" \
chown root:root /data/input/*.csv && chmod 644 /data/input/*.csv \
  &&  chown root:root /data/input/*.CSV && chmod 644 /data/input/*.CSV \
"


# inspect
# docker run --rm -it -v $VOLUME:/data alpine sh

docker rm temp-container




