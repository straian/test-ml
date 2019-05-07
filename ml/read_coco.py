from pycocotools.coco import COCO
import skimage
import numpy as np
import psutil

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

def read_set(data_type, image_size, count=None):
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

def read_dataset(image_size, train_count, val_count):
  train_images, train_targets = read_set('train2017', image_size, train_count)
  val_images, val_targets = read_set('val2017', image_size, val_count)
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

#train_images, train_targets, val_images, val_targets = read_dataset(224, 1024, 32)
#save_dataset(224, train_images, train_targets, val_images, val_targets)
#load_dataset(224, 4, 1)
