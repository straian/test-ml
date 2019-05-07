#!/bin/bash

CONTAINER_NAME=test-container

echo $CONTAINER_NAME

docker pull straian/test-ml

rm -fr charts
rm -fr checkpoints
mkdir charts
mkdir checkpoints
docker rm -f $CONTAINER_NAME
docker run --runtime=nvidia -dit --name=$CONTAINER_NAME \
    -v `pwd`/datasets:/datasets \
    -v `pwd`/charts:/charts \
    -v `pwd`/checkpoints:/checkpoints \
    straian/test-ml bash

docker exec -t $CONTAINER_NAME python ml/coco.py

# Delete all but last
cd checkpoints; rm -f `ls|sort -r|awk 'NR>1'`; cd -

docker rm -f $CONTAINER_NAME

