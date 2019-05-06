#!/bin/bash

CONTAINER_NAME=test-container

echo $CONTAINER_NAME

docker pull straian/test-ml

rm -fr charts
docker rm -f $CONTAINER_NAME
docker run --runtime=nvidia -dit --name=$CONTAINER_NAME -v `pwd`/datasets:/datasets straian/test-ml bash

docker exec -t $CONTAINER_NAME rm -fr charts
docker exec -t $CONTAINER_NAME mkdir charts

docker exec -t $CONTAINER_NAME python ml/coco.py

docker cp $CONTAINER_NAME:charts .

docker rm -f $CONTAINER_NAME

