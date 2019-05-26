import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

import psutil

import read_dataset

# Load images
#train_images, train_targets, val_images, val_targets = read_coco.load_dataset(IMAGE_SIZE, TRAIN_SAMPLES, VAL_SAMPLES)
#test_images = val_images
#test_targets = val_targets

#Model

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

GPUS = get_available_gpus()

print("GPU count: ", GPUS)

EPOCHS = 50 if GPUS else 2
TRAIN_SAMPLES = None if GPUS else 4
VAL_SAMPLES = None if GPUS else 1

BATCH_SIZE = (16 * GPUS) if GPUS else 4
CHART_INTERVAL = 1
MEMORY_INTERVAL = 10
MEMORY_INTERVAL_READ = 1000

read_dataset()
