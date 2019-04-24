#!/bin/bash

HOST_ADDR=34.83.191.160
SSH_KEY=~/.ssh/gcloud-test-ml
scp -i $SSH_KEY docker-run.sh straian@$HOST_ADDR:.
ssh -i $SSH_KEY straian@$HOST_ADDR bash docker-run.sh
scp -i $SSH_KEY straian@$HOST_ADDR:plot.png .
open plot.png
