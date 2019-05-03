#!/bin/bash

curl https://sdk.cloud.google.com | bash
exec bash

mkdir -p download/coco
#mkdir download/coco/train2017
mkdir download/coco/val2017
#mkdir download/coco/test2017

#gsutil -m rsync gs://images.cocodataset.org/train2017 download/coco/train2017
gsutil -m rsync gs://images.cocodataset.org/val2017 download/coco/val2017
#gsutil -m rsync gs://images.cocodataset.org/test2017 download/coco/test2017

#mv download/coco/train2017 datasets/coco
#mv download/coco/val2017 datasets/coco
#mv download/coco/test2017 datasets/coco
