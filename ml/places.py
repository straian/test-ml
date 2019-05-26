import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

import psutil

import read_dataset
import plot_coco
import matplotlib.pyplot as plt

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

GPUS = get_available_gpus()

print("GPU count: ", GPUS)

EPOCHS = 20 if GPUS else 2
TRAIN_SAMPLES = 20000 if GPUS else 4
VAL_SAMPLES = 2000 if GPUS else 1

BATCH_SIZE = (16 * GPUS) if GPUS else 4
CHART_INTERVAL = 1
MEMORY_INTERVAL = 10
MEMORY_INTERVAL_READ = 1000

# Base paper
# http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf
KERNEL=(3, 3)
IMAGE_SIZE=224
model = keras.Sequential([
    keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 4), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 4)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(64, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.Conv2D(64, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(128, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.Conv2D(128, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2DTranspose(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2DTranspose(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(128, KERNEL, padding='same'), keras.layers.ReLU(),
    keras.layers.Conv2DTranspose(128, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.Conv2DTranspose(128, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(64, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.Conv2DTranspose(64, KERNEL, strides=1, padding='same'), keras.layers.ReLU(),
    keras.layers.Conv2DTranspose(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2DTranspose(3, (1, 1), padding='same'),
    keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
])
print(model.summary())

if GPUS >= 2:
  model = keras.utils.multi_gpu_model(model, gpus=GPUS)

model.compile(optimizer='adam',
              loss='mean_squared_error',
              #loss='sparse_categorical_crossentropy',
              metrics=['mae'])
print("Memory info: ", psutil.virtual_memory())

# Load images
train_images, train_targets, val_images, val_targets = read_dataset.load_dataset(IMAGE_SIZE, TRAIN_SAMPLES, VAL_SAMPLES)
#train_images, train_targets, val_images, val_targets = read_dataset.read_dataset_places(IMAGE_SIZE, TRAIN_SAMPLES, VAL_SAMPLES)
read_dataset.save_dataset(IMAGE_SIZE, train_images, train_targets, val_images, val_targets)
test_images = val_images
test_targets = val_targets
print("Data shapes: ", train_images.shape, train_targets.shape, val_images.shape, val_targets.shape)

logs_history = {
  'loss': [],
  'val_loss': [],
  'mean_absolute_error': [],
  'val_mean_absolute_error': []
}
def epoch_report(epoch, logs):
  global logs_history
  logs_history['loss'].append(logs['loss'])
  logs_history['val_loss'].append(logs['val_loss'])
  logs_history['mean_absolute_error'].append(logs['mean_absolute_error'])
  logs_history['val_mean_absolute_error'].append(logs['val_mean_absolute_error'])
  if epoch % MEMORY_INTERVAL is 0:
    print("Memory info: ", psutil.virtual_memory())
  if epoch % CHART_INTERVAL > 0:
    return
  train_predictions = model.predict(train_images[0:1])
  test_predictions = model.predict(test_images[0:1])
  plot_coco.print_run(
      train_images[0][:,:,0:3], train_targets[0], train_predictions[0],
      test_images[0][:,:,0:3], test_targets[0], test_predictions[0], epoch, logs_history)

fit_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: epoch_report(epoch, logs))
model_checkpoint = keras.callbacks.ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5", save_best_only=True)

#with tf.device("/cpu:0"):
#with tf.device("/device:GPU:0"):
history = model.fit(
    train_images, train_targets,
    validation_data=(val_images, val_targets),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[fit_callback, model_checkpoint])

test_loss, test_mae = model.evaluate(test_images, test_targets)
print('Test loss:', test_loss)
print('Test mean abs err:', test_mae)

