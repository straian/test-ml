from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import skimage
import read_dataset
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

IMAGE_SIZE=224
N = 1

model1 = keras.models.load_model("checkpoints/checkpoints-coco/weights.08-0.08.hdf5")
print(model1.summary())

model2 = keras.models.load_model("checkpoints/checkpoints-places/weights.19-0.003.hdf5")
print(model2.summary())

_, _, test_images, test_targets = read_dataset.load_dataset(IMAGE_SIZE, 8, 8) # 8, 8 is for sure COCO

images = test_images[0:N]
targets = test_targets[0:N]
masks = model1.predict(images)
masks = np.where(masks < 0.1, 0, 1)
masks = 1. - masks

masks_triple = np.array([masks.transpose(), masks.transpose(), masks.transpose()]).transpose()
inputs2 = images * masks_triple
inputs2 = inputs2 / 256.
masks_reshaped = masks.reshape(N, IMAGE_SIZE, IMAGE_SIZE, 1)
inputs_combined = np.append(inputs2, 1. - masks_reshaped, axis=3)
masks = masks.reshape(N, IMAGE_SIZE, IMAGE_SIZE)
outputs = model2.predict(inputs_combined)

combined = np.where(masks_reshaped > 0.5, images / 256., outputs)

for i in range(N):
  skimage.io.imsave("images/original-{}.tiff".format(i), images[i])
  skimage.io.imsave("images/masked-{}.tiff".format(i), inputs2[i])
  skimage.io.imsave("images/mask-{}.tiff".format(i), 1. - masks[i])
  skimage.io.imsave("images/output-{}.tiff".format(i), outputs[i])
  skimage.io.imsave("images/combined-{}.tiff".format(i), combined[i])
  mpl.use("Agg")
  plt.figure(figsize=(12, 8))
  splt = plt.subplot(2, 3, 1)
  splt.imshow(images[i])
  splt = plt.subplot(2, 3, 2)
  splt.imshow(inputs2[i])
  splt = plt.subplot(2, 3, 3)
  splt.imshow(masks[i])
  splt = plt.subplot(2, 3, 4)
  splt.imshow(outputs[i])
  splt = plt.subplot(2, 3, 5)
  splt.imshow(combined[i])
  plt.savefig('images/plot-{}.png'.format(i))

