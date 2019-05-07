from __future__ import absolute_import, division, print_function

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

if offline:
  mpl.use("Agg")

def plot_loss(subplot, epochs, history, test_loss = None):
  val_loss = subplot.plot(epochs, history['val_loss'], '--', label='Val Loss')
  subplot.plot(epochs, history['loss'], color=val_loss[0].get_color(), label='Train Loss')
  if test_loss:
    subplot.plot(epochs, [test_loss] * len(epochs), label='Test loss')
  subplot.set_xlabel('Epochs')
  #subplot.set_yscale('log')
  subplot.set_ylabel('Loss')
  subplot.set_xlim([0, max(epochs)])

def plot_acc(subplot, epochs, history, test_acc = None):
  val_acc = subplot.plot(epochs, history['val_acc'], '--', label='Val Accuracy')
  subplot.plot(epochs, history['acc'], color=val_acc[0].get_color(), label='Train Accuracy')
  if test_acc:
    subplot.plot(epochs, [test_acc] * len(epochs), label='Test Accuracy')
  subplot.set_xlabel('Epochs')
  #subplot.set_yscale('log')
  subplot.set_ylabel('Accuracy')
  subplot.set_xlim([0, max(epochs)])

def print_run(
    train_input, train_target, train_predict, test_input, test_target, test_predict, epoch, history):
  plt.figure(figsize=(12, 8))

  splt = plt.subplot(2, 4, 1)
  splt.imshow(train_input)
  splt = plt.subplot(2, 4, 2)
  splt.imshow(train_target)
  splt = plt.subplot(2, 4, 3)
  splt.imshow(train_predict)

  plot_loss(plt.subplot(2, 4, 4), list(range(epoch + 1)), history)

  splt = plt.subplot(2, 4, 5)
  splt.imshow(test_input)
  splt = plt.subplot(2, 4, 6)
  splt.imshow(test_target)
  splt = plt.subplot(2, 4, 7)
  splt.imshow(test_predict)

  plot_acc(plt.subplot(2, 4, 8), list(range(epoch + 1)), history)
  plt.tight_layout()
  show()

def print_end(history, test_loss, test_acc):
  plt.figure(figsize=(12, 8))
  plot_loss(plt.subplot(1, 2, 1), history.epoch, history.history, test_loss)
  plot_acc(plt.subplot(1, 2, 2), history.epoch, history.history, test_acc)
  #plt.legend()
  show()

