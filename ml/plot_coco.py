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
  plot_metric('loss', 'Loss', subplot, epochs, history, test_loss)

def plot_acc(subplot, epochs, history, test_metric = None):
  plot_metric('acc', 'Accuracy', subplot, epochs, history, test_metric)

def plot_mae(subplot, epochs, history, test_metric = None):
  plot_metric('mean_absolute_error', 'Mean Abs Err', subplot, epochs, history, test_metric)

def plot_metric(metric, metric_name, subplot, epochs, history, test_metric = None):
  val_acc = subplot.plot(epochs, history['val_{}'.format(metric)], '--', label='Val {}'.format(metric_name))
  subplot.plot(epochs, history[metric], color=val_acc[0].get_color(), label='Train {}'.format(metric_name))
  if test_metric:
    subplot.plot(epochs, [test_acc] * len(epochs), label='Test {}'.format(metric_name))
  subplot.set_xlabel('Epochs')
  #subplot.set_yscale('log')
  subplot.set_ylabel(metric_name)
  subplot.set_xlim([0, max(epochs)])

def print_run(
    train_input, train_target, train_predict, test_input, test_target, test_predict, epoch, history):
  plt.figure(figsize=(12, 8))

  splt = plt.subplot(2, 4, 1)
  splt.imshow(train_input.astype("float32"))
  splt = plt.subplot(2, 4, 2)
  splt.imshow(train_target.astype("float32"))
  splt = plt.subplot(2, 4, 3)
  splt.imshow(train_predict.astype("float32"))

  plot_loss(plt.subplot(2, 4, 4), list(range(epoch + 1)), history)

  splt = plt.subplot(2, 4, 5)
  splt.imshow(test_input.astype("float32"))
  splt = plt.subplot(2, 4, 6)
  splt.imshow(test_target.astype("float32"))
  splt = plt.subplot(2, 4, 7)
  splt.imshow(test_predict.astype("float32"))

  if "acc" in history:
    plot_acc(plt.subplot(2, 4, 8), list(range(epoch + 1)), history)
  else:
    plot_mae(plt.subplot(2, 4, 8), list(range(epoch + 1)), history)
  plt.tight_layout()
  show()

def print_end(history, test_loss, test_acc):
  plt.figure(figsize=(12, 8))
  plot_loss(plt.subplot(1, 2, 1), history.epoch, history.history, test_loss)
  plot_acc(plt.subplot(1, 2, 2), history.epoch, history.history, test_acc)
  #plt.legend()
  show()

