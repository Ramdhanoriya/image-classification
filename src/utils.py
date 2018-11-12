'''

Read root directory containing images and return filenames,labels and class indices

@author : srijith

'''
from __future__ import print_function
import os
import numpy as np


filenames = []
labels = []
labels_constants = []

def get_image_paths(class_dir, label):
  '''
  Get Image paths from the filesystem

  Arguments : 
    class_dir : Directory containing input images
  
  Returns :
    A list containing absolute image paths
  '''
  if os.path.isdir(class_dir):
    images = os.listdir(class_dir)
    for img in images:
      if os.path.isfile(os.path.join(class_dir, img)):
        filenames.append(os.path.join(class_dir, img))
        labels.append(label)

def get_images_labels(path):
  '''
  Get the image dataset from the given path and prepare filenames , labels list

  Arguments:
    path : Directory containing all input images

  Returns:
    Image filenames list and labels list
  '''
  path_exp = os.path.expanduser(path)
  classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
  classes.sort()
  no_classes = len(classes)
  for i in range(no_classes):
    class_name = classes[i]
    class_dir = os.path.join(path_exp, class_name)
    get_image_paths(class_dir,class_name)

  x = np.array(labels)
  unique_labels = np.unique(x)
  class_indices = dict(zip(unique_labels, range(len(unique_labels))))
 
  for lbl in labels:
    labels_constants.append(class_indices.get(lbl))

  return filenames, labels_constants, class_indices

  
