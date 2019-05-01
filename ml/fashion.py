from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import json

# Helper libraries
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

offline = not "DISPLAY" in os.environ

if offline:
  mpl.use("Agg")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def printImg(ax, i, image, label, predictions=None):
  label = class_names[label]
  pred = label
  softmax_value = 1.1

  if predictions is not None:
    print(predictions)
    print(np.argmax(predictions))
    print(class_names[np.argmax(predictions)])
    print(label)
    pred = class_names[np.argmax(predictions)]
    softmax_value = np.around(predictions[np.argmax(predictions)], 2)

  ax.imshow(image, cmap=plt.cm.binary)
  s = label
  s = s + "\n" + pred + "\n" + str(softmax_value)
  if label is not pred:
    ax.xaxis.label.set_color('red')
  ax.set_xlabel(s)

def printGallery(images, labels, predictions=None):
  k = 0
  i = 0
  while k < 15 and i < len(images):
    subplot = plt.subplot(3, 5, k + 1)

    if class_names[np.argmax(predictions[i])] is class_names[labels[i]]:
      i += 1
      continue

    printImg(subplot, i, images[i], labels[i], predictions[i] if predictions is not None else None)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    k += 1
    i += 1
  plt.tight_layout()
  if offline:
    plt.savefig('plot.png')
  else:
    plt.show()

def proc():
  fashion_mnist = keras.datasets.fashion_mnist
  data = fashion_mnist.load_data()
  (train_images, train_labels), (test_images, test_labels) = data

  print(train_images.shape)

  train_images = train_images / 255.0
  test_images = test_images / 255.0

  #printGallery(train_images, train_labels)

  #exit()

  #plt.colorbar()
  #plt.grid(False)
  #plt.show()

  print(tf.__version__)

  model = keras.Sequential([
      keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
      keras.layers.Conv2D(32, (5, 5), activation=tf.nn.relu),
      keras.layers.MaxPooling2D(),
      keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
      keras.layers.MaxPooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation=tf.nn.relu),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  print(model.summary())

  model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_images, train_labels, epochs=30, batch_size=32)

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy:', test_acc)
  predictions = model.predict(test_images)
  printGallery(test_images, test_labels, predictions)


#with tf.device("/cpu:0"):
with tf.device("/device:GPU:0"):
  proc()

