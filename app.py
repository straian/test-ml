from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def proc:
  print(tf.__version__)

  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  #print(train_images.shape)

  train_images = train_images / 255.0
  test_images = test_images / 255.0

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  #plt.figure(figsize=(5, 5))
  #plt.imshow(train_images[0])

  #for i in range(20):
  #  plt.subplot(5, 4 if i % 2 else 4, i + 1)
  #  plt.xticks([])
  #  plt.yticks([])
  #  plt.grid(False)
  #  plt.imshow(train_images[i], cmap=plt.cm.binary)
  #  plt.xlabel(class_names[train_labels[i]])
  #plt.show()

  #plt.colorbar()
  #plt.grid(False)
  #plt.show()

  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation=tf.nn.relu),
      keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_images, train_labels, epochs=5)

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy:', test_acc)

  predictions = model.predict(test_images)
  print(predictions[0])
  print(np.argmax(predictions[0]))
  print(test_labels[0])

proc()

