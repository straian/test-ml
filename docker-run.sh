#!/bin/bash

CONTAINER_NAME=test-container

echo $CONTAINER_NAME

docker pull straian/test-ml

rm -fr plot.png
docker rm -f $CONTAINER_NAME
docker run --runtime=nvidia -it -d --name=$CONTAINER_NAME straian/test-ml bash

docker exec $CONTAINER_NAME python ml/app.py

docker cp $CONTAINER_NAME:plot.png .

docker rm -f $CONTAINER_NAME

