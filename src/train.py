import os
import sys
import argparse
import utils
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization, Dense, MaxPooling2D, Flatten, Dropout, Input
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, model_from_json, save_model, load_model

from sklearn.metrics import classification_report, confusion_matrix


class ImageClassifier:
  def __init__(self, args):
    self.args = args

    if not os.path.isdir(args.model_dir):
      os.makedirs(args.model_dir)

    if not os.path.isdir(args.log_dir):
      os.makedirs(args.log_dir)

    tensorboard = TensorBoard(log_dir=args.log_dir, histogram_freq=0,write_graph=True, write_images=True)
    earlystop =  EarlyStopping(monitor='val_acc', patience=25, mode='max', verbose=1)
    checkpoint = ModelCheckpoint(args.model_dir+'/weights-{epoch:02d}-{val_acc:.2f}.hdf5',
                 monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1, save_weights_only=True)
    self.callbacks = [tensorboard,checkpoint,earlystop]

    if K.image_data_format() == 'channels_first':
      self.input_shape = (3, args.image_size, args.image_size)
    else:
      self.input_shape = (args.image_size, args.image_size, 3)

  # Initialize train mode params
  def init_train(self, train_set, val_set, class_indices):
    self.class_indices = class_indices
    self.train_set = train_set
    self.val_set = val_set

  # Initialize predict/evaluvate mode params
  def init_predict(self, test_set, class_indices):
    self.test_set = test_set
    self.class_indices = class_indices

  # Keras CNN model    
  def cnn_model(self):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape= self.input_shape, activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (2, 2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(self.args.no_class, activation='softmax'))

    return model

  def read_convert_image(self, img_path, label):
    """
    Read image from disk and crop/resize based on args
    Arguments:
      img_path : path to read image file 
      label    : image class label
      args     : command arguments to the program
    Returns:
      image_data : cropped/resized image
      label : iage class label
    """
    # TO DO : replace read_file with tf.gfile.FastGFile for faster read
    image_data = tf.read_file(img_path)
    image_data = tf.image.decode_jpeg(image_data, channels=3)

    if self.args.center_crop:
      image_data  = tf.image.resize_image_with_crop_or_pad(image_data, self.args.crop_size, self.args.crop_size)

    if self.args.resize:
      image_data = tf.image.resize_images(image_data, [self.args.image_size, self.args.image_size])
    
    one_hot_label = tf.one_hot(label, self.args.no_class)
    
    return image_data, one_hot_label
 
  # Random hue, brightness, contrast and saturation 
  def random_augment(self, img, lbl):
    img = tf.image.random_brightness(img, max_delta=32. / 255.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    return img, lbl

  # Flip the images left right
  def flip(self, img, lbl):
    image = tf.image.flip_left_right(img)
    return image, lbl

  # Transpose image
  def transpose(self, img, lbl):
    image = tf.image.transpose_image(img)
    return image, lbl

  def prepare_data(self, dataset):
    '''
    Read and decode image. Crop/Resize image or both

    Arguments:
      tf.data.Dataset contains (image file paths, labels) pair
    Returns:
      tf.data.Datset contains (image, labels) pair
    '''
    process_fn = lambda img, lbl: self.read_convert_image(img, lbl)
    dataset = dataset.map(process_fn, num_parallel_calls=self.args.no_threads).apply(tf.contrib.data.ignore_errors())
    return dataset

  # get the compiled model
  def get_cnn_model(self):
    train_model = self.cnn_model()

    optimizer = optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0011)
    train_model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return train_model

  # Train the images and cross-validate
  def train_validate(self):
    K.set_learning_phase(1)
    # image augmentation lambda functions
    flip_fn = lambda img, lbl: self.flip(img,lbl)
    transpose_fn = lambda img, lbl: self.transpose(img,lbl)
    random_aug_fn = lambda img, lbl: self.random_augment(img,lbl)

    train_set = self.prepare_data(self.train_set)

    # image augmentation , flipping and transpose
    # concatenate flipped/transposed/augmented image data sets to original dataset
    if self.args.flip:
      fliped_data = train_set.map(flip_fn, num_parallel_calls=self.args.no_threads).apply(tf.contrib.data.ignore_errors())
      train_set = train_set.concatenate(fliped_data)

    if self.args.transpose:
      transpose_data = train_set.map(transpose_fn, num_parallel_calls=self.args.no_threads).apply(tf.contrib.data.ignore_errors())
      train_set = train_set.concatenate(transpose_data)

    if self.args.random_augment:
      noise_data = train_set.map(random_aug_fn, num_parallel_calls=self.args.no_threads).apply(tf.contrib.data.ignore_errors())
      train_set = train_set.concatenate(noise_data)
   
    train_set = train_set.batch(self.args.batch_size).apply(tf.contrib.data.shuffle_and_repeat(self.args.batch_size))

    # prepare validation dataset
    val_set = self.prepare_data(self.val_set)
    val_set = val_set.batch(self.args.batch_size).repeat()

    train_model = self.get_cnn_model()
    model_json = train_model.to_json()   
    with open(self.args.model_dir +'/model.json', "w") as json_file:
      json_file.write(model_json)
    
    train_model.fit(train_set, epochs=self.args.no_epochs,
                steps_per_epoch=self.args.epochs_steps,callbacks=self.callbacks,
                validation_data=val_set, validation_steps=self.args.epochs_steps)

    train_model.save(self.args.model_dir+"/model.h5", overwrite=True, include_optimizer=True)

  # Evaluvate on test data / predict
  def evaluvate_predict(self):
    K.set_learning_phase(0)

    # If required to load weights and re-initiaze model, uncomment this block
    '''json_file = open(self.args.model_json, 'r')
    model_json = json_file.read()
    json_file.close()
  
    test_model = model_from_json(model_json)
    print('Loaded model json from {} '.format(self.args.model_json))
    if self.args.weights_file is None:
      test_model.load_weights(tf.train.latest_checkpoint(self.args.weights_dir))
    else:
      test_model.load_weights(self.args.weights_file)
    print('Loaded model weights from ')
    optimizer = optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0011)
    test_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    print('Model compiled...')'''

    test_model = load_model(self.args.model_file)
    test_model.summary()
   
    test_set = self.prepare_data(self.test_set)
    test_set = test_set.batch(50)
    iterator = test_set.make_one_shot_iterator()
    images, labels = iterator.get_next()

    loss, acc = test_model.evaluate(test_set, steps=self.args.epochs_steps)
    print("Test Accuracy : {} and Test Loss : {} ".format(acc, loss))
  
    actual= []
    predicted = []
    with tf.keras.backend.get_session() as sess:
      while True:
        try:
          imgs, lbls = sess.run([images, labels])
          for img, lbl in zip(imgs, lbls):
            img = (np.expand_dims(img, 0))
            result = test_model.predict(img)
            #print('predicted : {} and actual label :  {} '.format(np.argmax(result[0]), lbl))
            actual_lbl = tf.argmax(lbl, axis=0)
            actual.append(self.getKeyByValue(self.class_indices, sess.run(actual_lbl)))
            predicted.append(self.getKeyByValue(self.class_indices, np.argmax(result[0])))         
        except tf.errors.OutOfRangeError:
          print("End of test dataset...")
          break
    print('\n Classification report \n\n', classification_report(actual, predicted, labels=list(self.class_indices.keys())))
    print('Confusion Metrics : \n\n', confusion_matrix(actual,predicted, labels=list(self.class_indices.keys())))

  
  #Get a list of keys from dictionary which has the given value
  def getKeyByValue(self, indices, valueToFind):
    key = ''
    items = indices.items()
    for item  in items:
      if item[1] == valueToFind:
        key = item[0]  
    return key

def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  # input parameters
  parser.add_argument('--data_dir', type=str, help='Path to the directory contains train images', default='../data/train_val')
  parser.add_argument('--seed', type=int, help='Random seed.', default=666)
  parser.add_argument('--batch_size', type=int, help='Batch size to fetch from dataset', default=150)
  parser.add_argument('--no_threads', type=int, help='Number of parllel threads to run in dataset', default=20)
  parser.add_argument('--mode', type=str, help='Train or Predict', default='train')

  # Train parameters
  parser.add_argument('--no_epochs', type=int, help='Number of epochs', default=50)
  parser.add_argument('--no_class', type=int, help='Number of classes', default=20)
  parser.add_argument('--epochs_steps', type=int, help='Steps per epoch', default=20)
  parser.add_argument('--val_split_ratio', type=float, help='Validation split ratio',default=0.2)

  # Pre-processing parameters
  parser.add_argument('--center_crop', help='Performs center cropping/pad of training images.', action='store_true')
  parser.add_argument('--crop_size', type=int, help='Crop size', default=1000)
  parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=300)
  parser.add_argument('--resize', help='Resize image based on image_size', action='store_true')
  parser.add_argument('--flip', help='Apply left right flip', action='store_true')
  parser.add_argument('--transpose', help='Apply image transpose', action='store_true')
  parser.add_argument('--random_augment', help='Apply random  hue, satuuration and brightness', action='store_true')

  # Pre-trained model parameters
  parser.add_argument('--model', type=str, choices=['Xception', 'VGG16', 'VGG19', 'ResNet50', 'InceptionV3','InceptionResNetV2','MobileNet','MobileNetV2','DenseNet121','DenseNet169','DenseNet201','NASNetMobile','NASNetLarge'], help='Pre-trained model to use', default='InceptionResNetV2')
  parser.add_argument('--use_weights', help='Use the weights from pre-trained models or not', action='store_true')

  # Output parameters
  parser.add_argument('--model_dir', type=str, help='Model dir to write checkpoints', default='../model')
  parser.add_argument('--log_dir', type=str, help='Dir to write logs', default='../logs')

  # Predict parameters
  parser.add_argument('--test_dir', type=str, help='Path to the directory contains test images', default='../data/test')
  parser.add_argument('--model_file', type=str, help='h5 model file to load', default= '../model/model.h5')

  '''
  If you want to load model.json and weights instead of a full model , uncomment this params and uncomment the block
  in evaluvate_predict method accordingly

  parser.add_argument('--model_json', type=str, help='keras model json file path')
  parser.add_argument('--weights_file', type=str, help='hdf5 weights file to load')
  parser.add_argument('--weights_dir', type=str, help='If you provide a dir , then it will take latest checkpoint')
  '''

  return parser.parse_args(argv)

if __name__ == "__main__":
  args = parse_arguments(sys.argv[1:])

  classify_object = ImageClassifier(args)
  if args.mode == 'train':
    # create a filenames, labels list
    filenames, labels, class_indices = utils.get_images_labels(args.data_dir)
    # split train, val and test set
    train_files, val_files, train_labels, val_labels = train_test_split(filenames,
                                                       labels, test_size=args.val_split_ratio, random_state=args.seed) 
    # create tf.data.Dataset for train set and test set
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels)) 
    # train and validate the images
    classify_object.init_train(train_dataset, val_dataset, class_indices)
    classify_object.train_validate()
  elif args.mode == 'test':
    test_files, test_labels, class_indices = utils.get_images_labels(args.test_dir)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    classify_object.init_predict(test_dataset,class_indices)
    classify_object.evaluvate_predict()
  else:
    print('unknown mode')

 
