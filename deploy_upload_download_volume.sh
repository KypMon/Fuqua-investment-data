#!/bin/bash
# generate the docker volume that holds the 4 user upload/download files.
#
# We keep these files on a separate volume, because they are not source-controlled in gitlab.
# This creates a problem when doing an auto-deployment via .gitlab-ci.yml.
# The solution is to maintain this volume separately.
# We NEVER want to delete/recreate this volume, once it is created.
# Well, perhaps at the end of a term or session.  Need guidance from the professors.
#
##
## To view the files on the volume:
##
##  docker run --rm -it -v fa_volume_upload_download:/static alpine /bin/sh
##
##
VOLUME=fa_volume_upload_download

# docker volume rm $VOLUME 2>/dev/null || true  NO !!!!!!

docker volume create --driver local $VOLUME;
docker run --rm \
-v $VOLUME:/static alpine mkdir \
-p /static /static/matrix /static/matrix/tokens /static/lifecycle /static/lifecycle/tokens

# inspect
docker run --rm -it -v $VOLUME:/static alpine sh



