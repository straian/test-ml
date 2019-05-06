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

GPUS=0

if offline:
  mpl.use("Agg")

class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def plot_history(history, test_loss, test_acc):
  plt.figure(figsize=(12, 6))
  splt = plt.subplot(1, 2, 1)
  val_loss = splt.plot(history.epoch, history.history['val_loss'], '--', label='Val Loss')
  splt.plot(history.epoch, history.history['loss'], color=val_loss[0].get_color(), label='Train Loss')
  splt.plot(history.epoch, [test_loss] * len(history.epoch), label='Test loss')
  splt.set_xlabel('Epochs')
  splt.set_ylabel('Loss')
  splt = plt.subplot(1, 2, 2)
  val_acc = splt.plot(history.epoch, history.history['val_acc'], '--', label='Val Accuracy')
  splt.plot(history.epoch, history.history['acc'], color=val_acc[0].get_color(), label='Train Accuracy')
  splt.plot(history.epoch, [test_acc] * len(history.epoch), label='Test Accuracy')
  splt.set_xlabel('Epochs')
  splt.set_ylabel('Accuracy')
  plt.legend()
  plt.xlim([0, max(history.epoch)])

dataDir='datasets/coco'
downloadDir='download/coco'
dataType='train2017'
#annFile='{}/annotations/panoptic2detection_{}.json'.format(dataDir, dataType)
annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

IMAGE_SIZE=224
TRAIN_SAMPLES = 100
TEST_SAMPLES = 10
imgs = coco.loadImgs(imgIds[:TRAIN_SAMPLES + TEST_SAMPLES])

train_images = []
train_targets = []
test_images = []
test_targets = []
i = 0
for img in imgs:
  inputImg0 = io.imread('{}/{}/{}'.format(downloadDir, dataType, img['file_name']))
  targetImg0 = io.imread('{}/panoptic_{}/{}'.format(dataDir, dataType, img['file_name'].replace(".jpg", ".png")))
  if (i < TRAIN_SAMPLES):
    inputs = train_images
    targets = train_targets
  else:
    inputs = test_images
    targets = test_targets
  inputImg = skimage.transform.resize(inputImg0, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
  targetImg = skimage.transform.resize(targetImg0, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
  if (len(inputImg.shape) < 3):
    continue
  inputs.append(inputImg)
  targets.append(targetImg)
  #print(inputImg.shape)
  #print(targetImg.shape)
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

#plt.imshow(I)
#plt.axis('off')
#plt.show()
#annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#anns = coco.loadAnns(annIds)
#coco.showAnns(anns)

KERNEL=(3, 3)
model = keras.Sequential([
    keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    keras.layers.Conv2D(32, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(256, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(512, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(256, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(128, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(64, KERNEL, padding='same', activation=tf.nn.relu),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(32, KERNEL, padding='same', activation=tf.nn.relu),
    #keras.layers.Dropout(0.5),
    #keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
    keras.layers.Conv2D(3, KERNEL, padding='same', activation=tf.nn.relu)
])
print(model.summary())

if GPUS >= 2:
  model = keras.utils.multi_gpu_model(model, gpus=GPUS)

model.compile(optimizer='adam', 
              #loss='sparse_categorical_crossentropy',
              loss='mean_squared_error',
              metrics=['accuracy'])

#with tf.device("/cpu:0"):
with tf.device("/device:GPU:0"):
  history = model.fit(train_images, train_targets, validation_split=0.2, epochs=10, batch_size=8)
  print(history)
  test_loss, test_acc = model.evaluate(test_images, test_targets)

  plot_history(history, test_loss, test_acc)
  print('Test loss:', test_loss)
  print('Test accuracy:', test_acc)
  #predictions = model.predict(test_images)
  #print(predictions[0])
  #print(test_images)

if offline:
  plt.savefig('plot.png')
else:
  plt.show()

