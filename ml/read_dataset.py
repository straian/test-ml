from pycocotools.coco import COCO
import skimage
import numpy as np
import psutil
import math
import subprocess

import plot_coco

from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from random import shuffle

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

PRINT_COUNT_INTERVAL = 50
PRINT_MEMORY_INTERVAL = 1000

PLACES_SIZE = 256
PLACES_MAX_COUNT = 1

#https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib
def generate_random_shape():
  n = 8 # Number of possibly sharp edges
  r = .5 # magnitude of the perturbation from the unit circle,
  # should be between 0 and 1
  N = n*3+1 # number of points in the Path
  # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve
  angles = np.linspace(0,2*np.pi,N)
  codes = np.full(N,Path.CURVE4)
  codes[0] = Path.MOVETO
  lengths = (2*r*np.random.random(N)+1-r) / (1 + r)
  # Alter lengths: [-1, 1] to [-X/2, X/2]
  min_fill_ratio = 0.1 / 2
  max_fill_ratio = 0.5 / 2
  diff = max_fill_ratio - min_fill_ratio
  scale_x = np.random.random() * diff + min_fill_ratio
  scale_y = np.random.random() * diff + min_fill_ratio
  move_x = scale_x + np.random.random() * (1 - 2 * scale_x)
  move_y = scale_y + np.random.random() * (1 - 2 * scale_y)
  #print(scale_x, scale_y, move_x, move_y, move_x + scale_x, move_y + scale_y)
  diff = max_fill_ratio - min_fill_ratio
  verts_x = move_x + np.cos(angles) * lengths * scale_x
  verts_y = move_y + np.sin(angles) * lengths * scale_y
  size = PLACES_SIZE
  verts_x *= size
  verts_y *= size
  verts_x = np.uint16(verts_x)
  verts_y = np.uint16(verts_y)
  img = np.zeros((size, size), dtype=np.uint8)
  rr, cc = skimage.draw.polygon(verts_x, verts_y)
  img[rr, cc] = 1
  return img

def generate_random_shapes():
  img = generate_random_shape()
  for _ in range(4):
    img = img | generate_random_shape()
  return img

def run_cmd(cmd):
  return list(filter(lambda x: x is not "", subprocess.Popen(cmd,
      shell=True,
      stdout=subprocess.PIPE,
      universal_newlines=True).communicate()[0].split("\n")))

def read_set_places(data_type, image_size, count=None):
  data_dir='datasets/places365_standard/train'
  img_files = run_cmd("find {} -type f".format(data_dir))
  if not count:
    count = len(img_files)
  count = min(count, len(img_files), PLACES_MAX_COUNT)
  images = np.zeros([count, image_size, image_size, 4], dtype='uint8')
  targets = np.zeros([count, image_size, image_size, 3], dtype='uint8')
  shuffle(img_files)
  print("Count: ", data_type, len(img_files))
  for i in range(count):
    input_img = skimage.io.imread(img_files[i])
    input_img = input_img * 1.
    input_img = np.array(skimage.transform.resize(input_img, (image_size, image_size), anti_aliasing=True), dtype='uint8')
    targets[i] = input_img
    # Sometimes don't add a mask.
    if (np.random.random() < 0.05):
      mask = np.zeros([image_size, image_size, 1], dtype="uint8")
    else:
      mask = generate_random_shapes().reshape(image_size, image_size, 1)
    input_img *= np.logical_not(mask)
    images[i] = np.append(input_img, mask, axis=2)
    i += 1

  print("images.nbytes: ", images.shape, images.nbytes, targets.shape, targets.nbytes)
  print("Memory info: ", psutil.virtual_memory())
  return images, targets

images, targets = read_set_places("train", 256)
plt.figure(figsize=(12, 6))
splt = plt.subplot(1, 3, 1)
splt.imshow(targets[0])
splt = plt.subplot(1, 3, 2)
splt.imshow(images[0][:,:,-1])
splt = plt.subplot(1, 3, 3)
splt.imshow(images[0][:,:,0:3])
plot_coco.show()

def read_set_coco(data_type, image_size, count=None):
  data_dir='datasets/coco'
  inputImgDir='{}/{}'.format(data_dir, data_type)
  annFile='{}/annotations/instances_{}.json'.format(data_dir, data_type)
  coco=COCO(annFile)
  catIds = coco.getCatIds(catNms=['person'])
  imgIds = coco.getImgIds(catIds=catIds)[0:count]
  if not count:
    # Add some images without a label present at all -- extra 10%, capped at 1000.
    EXTRA_IMAGES = min(1000, int(len(imgIds) / 10))
    set_diff = set(coco.getImgIds()) - set(imgIds)
    imgIds.extend(list(set_diff)[0:EXTRA_IMAGES])
  shuffle(imgIds)
  imgs = coco.loadImgs(imgIds)
  print("Count: ", data_type, len(imgs))
  num_train = 0
  images = np.zeros([len(imgs), image_size, image_size, 3], dtype='uint8')
  targets = np.zeros([len(imgs), image_size, image_size], dtype='uint8')
  print("Memory info: ", psutil.virtual_memory())
  i = 0
  for img in imgs:
    input_img = np.array(skimage.io.imread('{}/{}'.format(inputImgDir, img['file_name'])))
    if (len(input_img.shape) < 3):
      # Add one extra dim
      print(input_img.shape)
      input_img = np.array([input_img.transpose(), input_img.transpose(), input_img.transpose()]).transpose()
      print(input_img.shape)
    anns = coco.loadAnns(coco.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = None))
    mask = np.zeros(input_img.shape[0:2], int)
    for ann in anns:
      mask = mask | coco.annToMask(ann)
    input_img = input_img * 1.
    mask = mask * 1.
    input_img = pad(input_img)
    target_img = pad(mask)
    input_img = np.array(skimage.transform.resize(input_img, (image_size, image_size), anti_aliasing=True), dtype='uint8')
    target_img = np.array(skimage.transform.resize(target_img, (image_size, image_size), anti_aliasing=False), dtype='uint8')
    images[i] = input_img
    targets[i] = target_img
    i += 1
    if i % PRINT_COUNT_INTERVAL == 0:
      print(i)
    if i % PRINT_MEMORY_INTERVAL == 0:
      print("Memory info: ", psutil.virtual_memory())
  print("images.nbytes: ", images.shape, images.nbytes)
  print("targets.nbytes: ", targets.shape, targets.nbytes)
  print("Memory info: ", psutil.virtual_memory())
  return images, targets

def get_name(sz, l, s):
  if l:
    return "npydata/{}-{}-{}.npy".format(s, sz, l)
  else:
    return "npydata/{}-{}.npy".format(s, sz)

def read_dataset_places(image_size, train_count, val_count):
  train_images, train_targets = read_set_places('train', image_size, train_count)
  val_images, val_targets = read_set_places('val', image_size, val_count)
  return train_images, train_targets, val_images, val_targets

def read_dataset_coco(image_size, train_count, val_count):
  train_images, train_targets = read_set_coco('train2017', image_size, train_count)
  val_images, val_targets = read_set_coco('val2017', image_size, val_count)
  return train_images, train_targets, val_images, val_targets

def load_dataset(image_size, train_count, val_count):
  train_images  = np.load(get_name(image_size, train_count, "train_images"))
  train_targets = np.load(get_name(image_size, train_count, "train_targets"))
  val_images    = np.load(get_name(image_size, val_count  , "val_images"))
  val_targets   = np.load(get_name(image_size, val_count  , "val_targets"))
  return train_images, train_targets, val_images, val_targets

def save_dataset(image_size, train_images, train_targets, val_images, val_targets):
  np.save(get_name(image_size, len(train_images ), "train_images"), train_images)
  np.save(get_name(image_size, len(train_targets), "train_targets"), train_targets)
  np.save(get_name(image_size, len(val_images   ), "val_images"   ), val_images)
  np.save(get_name(image_size, len(val_targets  ), "val_targets"  ), val_targets)

#train_images, train_targets, val_images, val_targets = read_dataset_coco(224, 1024, 32)
#save_dataset(224, train_images, train_targets, val_images, val_targets)
#load_dataset(224, 4, 1)
