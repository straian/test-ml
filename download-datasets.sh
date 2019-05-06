#!/bin/bash

mkdir -p datasets/coco
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip -P datasets/coco
unzip -o datasets/coco/panoptic_annotations_trainval2017.zip -d datasets/coco
unzip -o datasets/coco/annotations/panoptic_train2017.zip -d datasets/coco/annotations

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P datasets/coco
unzip -o datasets/coco/annotations_trainval2017.zip -d datasets/coco

#curl https://sdk.cloud.google.com | bash
#exec bash

#mkdir -p datasets/coco/train2017
#mkdir -p datasets/coco/val2017
#mkdir -p datasets/coco/test2017

#gsutil -m rsync gs://images.cocodataset.org/train2017 datasets/coco/train2017
#gsutil -m rsync gs://images.cocodataset.org/val2017 datasets/coco/val2017
#gsutil -m rsync gs://images.cocodataset.org/test2017 datasets/coco/test2017

