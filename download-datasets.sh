#!/bin/bash

mkdir -p datasets/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P datasets/coco
sudo apt-get install unzip
unzip -o datasets/coco/annotations_trainval2017.zip -d datasets/coco

# NOT NEEDED on GCE! If you do need it, you might need to install python 2.7 first:
#sudo apt-get install python
#curl https://sdk.cloud.google.com | bash
#exec bash

#mkdir -p datasets/coco/train2017
mkdir -p datasets/coco/val2017
#mkdir -p datasets/coco/test2017

# Might encounter 'No module named google_compute_engine' error, fix: https://stackoverflow.com/a/46606975/5258187
#gsutil -m rsync gs://images.cocodataset.org/train2017 datasets/coco/train2017
gsutil -m rsync gs://images.cocodataset.org/val2017 datasets/coco/val2017
#gsutil -m rsync gs://images.cocodataset.org/test2017 datasets/coco/test2017

# Manual command to populate npydata from dev machine
tar -zcvf npydata.tar.gz npydata
scp -r -i $SSH_KEY npydata.tar.gz straian@$HOST_ADDR:.
mkdir -p npydata

