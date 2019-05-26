from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

import psutil

import read_dataset
import plot_coco

# Use half precision
# https://medium.com/@noel_kennedy/how-to-use-half-precision-float16-when-training-on-rtx-cards-with-tensorflow-keras-d4033d59f9e4
#keras.backend.set_floatx('float16')

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

# SegNet: https://arxiv.org/pdf/1511.00561.pdf
# VGG16: https://neurohive.io/en/popular-networks/vgg16/
# TeranusNet: https://arxiv.org/pdf/1801.05746.pdf
# U-NET: https://arxiv.org/pdf/1505.04597.pdf
KERNEL=(3, 3)
IMAGE_SIZE=224
model = keras.Sequential([
    keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    # Beneficial instead of input normalization
    # https://www.reddit.com/r/MachineLearning/comments/9g7m9x/d_what_would_happen_if_a_model_used_batch/?depth=2
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2DTranspose(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2DTranspose(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2DTranspose(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2DTranspose(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2DTranspose(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2DTranspose(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2DTranspose(1, (1, 1), padding='same', activation=tf.nn.sigmoid),
    keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE))
])
print(model.summary())

train_images, train_targets, val_images, val_targets = read_dataset.load_dataset(IMAGE_SIZE, TRAIN_SAMPLES, VAL_SAMPLES)
test_images = val_images
test_targets = val_targets

#TODO: One hot encoding for image segmentation (person or no person) like here
# https://www.jeremyjordan.me/semantic-segmentation/#representing
#TODO: Also do cross entropy loss if i have one hot encoding. Acc -> probability, or something?

if GPUS >= 2:
  model = keras.utils.multi_gpu_model(model, gpus=GPUS)

model.compile(optimizer='adam', 
              #loss='sparse_categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("Memory info: ", psutil.virtual_memory())

logs_history = {
  'loss': [],
  'val_loss': [],
  'acc': [],
  'val_acc': []
}
def epoch_report(epoch, logs):
  global logs_history
  logs_history['loss'].append(logs['loss'])
  logs_history['val_loss'].append(logs['val_loss'])
  logs_history['acc'].append(logs['acc'])
  logs_history['val_acc'].append(logs['val_acc'])
  if epoch % MEMORY_INTERVAL is 0:
    print("Memory info: ", psutil.virtual_memory())
  if epoch % CHART_INTERVAL > 0:
    return
  train_predictions = model.predict(train_images[0:1])
  test_predictions = model.predict(test_images[0:1])
  plot_coco.print_run(
      train_images[0], train_targets[0], train_predictions[0],
      test_images[0], test_targets[0], test_predictions[0], epoch, logs_history)

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

test_loss, test_acc = model.evaluate(test_images, test_targets)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

plot_coco.print_end(history, test_loss, test_acc)
