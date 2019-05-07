#!/bin/bash

# How to generate key: https://stackoverflow.com/a/53524211/5258187
SSH_KEY=~/.ssh/gcloud-test-ml

# instance-1
#HOST_ADDR=34.83.191.160

# gpu-preemptible-1
#HOST_ADDR=35.197.30.148

# gpu-preemptible-2
#HOST_ADDR=34.83.72.35

./build.sh

echo $HOST_ADDR

rm -fr charts-$HOST_ADDR
rm -fr checkpoints-$HOST_ADDR
scp -i $SSH_KEY docker-run.sh straian@$HOST_ADDR:.
ssh -i $SSH_KEY straian@$HOST_ADDR bash docker-run.sh
scp -r -i $SSH_KEY straian@$HOST_ADDR:charts charts-$HOST_ADDR
scp -r -i $SSH_KEY straian@$HOST_ADDR:checkpoints checkpoints-$HOST_ADDR
open charts-$HOST_ADDR/*.png

