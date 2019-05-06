from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import json

from pycocotools.coco import COCO
import skimage.io as io
import skimage

# Helper libraries
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

offline = not "DISPLAY" in os.environ

chart_count = 0
def show():
  global chart_count
  if offline:
    plt.savefig('charts/plot-{}.png'.format(chart_count))
    chart_count += 1
  else:
    plt.show()
  # Clear the current axes.
  plt.cla()
  # Clear the current figure.
  plt.clf()
  # Closes all the figure windows.
  plt.close('all')

EPOCHS = 100 if offline else 50
CHART_INTERVAL = 1
TRAIN_SAMPLES = 10000 if offline else 20
TEST_SAMPLES = 1
BATCH_SIZE = 16 if offline else 4

GPUS = 2 if offline else 0

if offline:
  mpl.use("Agg")

def pad(img):
  p00 = int((640 - img.shape[0]) / 2)
  p01 = 640 - img.shape[0] - p00
  p10 = int((640 - img.shape[1]) / 2)
  p11 = 640 - img.shape[1] - p10
  if len(img.shape) == 2:
    return np.pad(img, ((p00, p01), (p10, p11)), 'constant')
  else:
    assert len(img.shape) is 3
    return np.pad(img, ((p00, p01), (p10, p11), (0, 0)), 'constant')



def plot_history(history, test_loss, test_acc):
  plt.figure(figsize=(12, 6))
  splt = plt.subplot(1, 2, 1)
  val_loss = splt.plot(history.epoch, history.history['val_loss'], '--', label='Val Loss')
  splt.plot(history.epoch, history.history['loss'], color=val_loss[0].get_color(), label='Train Loss')
  splt.plot(history.epoch, [test_loss] * len(history.epoch), label='Test loss')
  splt.set_xlabel('Epochs')
  splt.set_yscale('log')
  splt.set_ylabel('Loss')
  splt = plt.subplot(1, 2, 2)
  val_acc = splt.plot(history.epoch, history.history['val_acc'], '--', label='Val Accuracy')
  splt.plot(history.epoch, history.history['acc'], color=val_acc[0].get_color(), label='Train Accuracy')
  splt.plot(history.epoch, [test_acc] * len(history.epoch), label='Test Accuracy')
  splt.set_xlabel('Epochs')
  splt.set_yscale('log')
  splt.set_ylabel('Accuracy')
  plt.legend()
  plt.xlim([0, max(history.epoch)])

def print_images(input, target, predict):
  plt.figure(figsize=(12, 6))
  splt = plt.subplot(1, 3, 1)
  splt.imshow(input)
  splt = plt.subplot(1, 3, 2)
  splt.imshow(target)
  splt = plt.subplot(1, 3, 3)
  splt.imshow(predict)

# SegNet: https://arxiv.org/pdf/1511.00561.pdf
# VGG16: https://neurohive.io/en/popular-networks/vgg16/
KERNEL=(3, 3)
IMAGE_SIZE=224
model = keras.Sequential([
    keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    keras.layers.Conv2D(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(512, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(256, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(128, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(64, KERNEL, padding='same'), keras.layers.BatchNormalization(), keras.layers.ReLU(),
    keras.layers.Conv2D(1, (1, 1), padding='same', activation=tf.nn.sigmoid),
    keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE))
])
print(model.summary())

dataDir='datasets/coco'
dataType='train2017'
inputImgDir='{}/{}'.format(dataDir, dataType)
targetImgDir='{}/annotations/panoptic_{}'.format(dataDir, dataType)
annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

imgs = coco.loadImgs(imgIds[:TRAIN_SAMPLES + TEST_SAMPLES])

train_images = []
train_targets = []
test_images = []
test_targets = []
i = 0
for img in imgs:
  input_img = io.imread('{}/{}'.format(inputImgDir, img['file_name']))
  if (len(input_img.shape) < 3):
    continue
  anns = coco.loadAnns(coco.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = None))
  mask = np.zeros(input_img.shape[0:2], int)
  for ann in anns:
    mask = mask | coco.annToMask(ann)
  mask = mask * 1.
  if (i < TRAIN_SAMPLES):
    inputs = train_images
    targets = train_targets
  else:
    inputs = test_images
    targets = test_targets
  input_img = pad(input_img)
  target_img = pad(mask)
  input_img = skimage.transform.resize(input_img, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
  target_img = skimage.transform.resize(target_img, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=False)
  inputs.append(input_img)
  targets.append(target_img)
  i += 1
  if i % 10 == 0:
    print(i)

#TODO: One hot encoding for image segmentation (person or no person) like here
# https://www.jeremyjordan.me/semantic-segmentation/#representing
#TODO: Also do cross entropy loss if i have one hot encoding. Acc -> probability, or something?

train_images = np.array(train_images)
train_targets = np.array(train_targets)
test_images = np.array(test_images)
test_targets = np.array(test_targets)

if GPUS >= 2:
  model = keras.utils.multi_gpu_model(model, gpus=GPUS)

model.compile(optimizer='adam', 
              #loss='sparse_categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])

def predict(epoch):
  if epoch % CHART_INTERVAL > 0:
    return
  predictions = model.predict(train_images[0:1])
  print_images(train_images[0], train_targets[0], predictions[0])
  show()

fit_callback = keras.callbacks.LambdaCallback(
    on_epoch_begin=lambda epoch, logs: predict(epoch))

#with tf.device("/cpu:0"):
with tf.device("/device:GPU:0"):
  history = model.fit(train_images, train_targets, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[fit_callback])
  print(history)
  test_loss, test_acc = model.evaluate(test_images, test_targets)
  plot_history(history, test_loss, test_acc)
  show()
  print('Test loss:', test_loss)
  print('Test accuracy:', test_acc)


