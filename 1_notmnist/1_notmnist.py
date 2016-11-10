
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 1
# ------------
# 
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# 
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import cv2

import pdb

# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

url = 'http://yaroslavvb.com/upload/notMNIST/'


def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labelled A through J.

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

# ---
# Problem 1
# ---------
# 
# Let's take a peek at some of the data to make sure it looks sensible. 
# Each exemplar should be an image of a character A through J rendered in a different font. 
# Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
# 
# ---

def display_file(folders, index):
  for d in folders:
    files = sorted(os.listdir(d))
    img = cv2.imread(d + '/' + files[index])
    cv2.imshow(d, img)
    cv2.waitKey(200)

display_file(train_folders, 0)
display_file(test_folders, -1)

# Now let's load the data in a more manageable format. 
# Since, depending on your computer setup you might not be able to fit it all in memory,
# we'll load each class into a separate dataset, store them on disk and curate them independently. 
# Later we'll merge them into a single dataset of manageable size.
#
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, 
# normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 
# 
# A few images might not be readable, we'll just skip them.

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# ---
# Problem 2, Problem 3
# ---------
# 
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. 
# Hint: you can use matplotlib.pyplot.
# ---------
# Another check: we expect the data to be balanced across classes. Verify that.
# 
# ---

# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, 
# and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
# 
# Also create a validation dataset for hyperparameter tuning.

# ---

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def balance_check(db):
  mean_val = mean(zip(*db)[1])
  print('mean of # images :', mean_val)
  for i in db:
    if abs(i[1]-mean_val) > 0.1*mean_val:
      print("Too much or less images")
    else:
      print("Well balanced", i[0])

def load_and_display_pickle(datasets, index):
  num_of_images = []
  for pickle_file in datasets:
    with open(pickle_file, 'rb') as f:
      data = pickle.load(f)
      print('Total images in',  pickle_file, ':',  len(data))
      cv2.imshow('sample',data[index])
      cv2.waitKey(200)
      num_of_images.append(len(data))

  db = zip(datasets, num_of_images)
  balance_check(db)

load_and_display_pickle(train_datasets, 0)
load_and_display_pickle(test_datasets, -1)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# Next, we'll randomize the data. 
# It's important!!! to have the labels well shuffled for the training and test distributions to match.

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
# 
# ---
def display_data(datasets, index):
  cv2.imshow('shuffled',datasets[index])
  cv2.waitKey(200)
      
display_data(train_dataset, 0)
display_data(test_dataset, 0)
display_data(valid_dataset, 0)

# Finally, let's save the data for later reuse:

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


# ---
# Problem 5
# ---------
# 
# By construction, this dataset might contain a lot of overlapping samples, 
# including training data that's also contained in the validation and test set! 
# Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, 
# but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
# 
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

def merge_data_with_label(dataset, label):
  num, w, h = dataset.shape
  data = np.ndarray((num, w * h + 1), dtype=np.float32)
  
  data_2d = reshape_3d_to_2d(dataset)

  for i in range(num):
    data[i, :] = np.append(data_2d[i],label[i])

  return data

def split_data_with_label(data):
  num, length = data.shape
  length = length - 1 # for label

  dataset = reshape_2d_to_3d(data.T[:length].T, image_size, image_size)
  labels =  data.T[-1].astype('int32')

  return dataset, labels

def reshape_2d_to_3d(data_2d, w, h):
  data_3d = data_2d.reshape(-1, w, h)
  return data_3d  

def reshape_3d_to_2d(data_3d):
  num, w, h = data_3d.shape
  data_2d = data_3d.reshape(num, w * h)
  return data_2d

def remove_duplicate(data):                 
  sorted_data =  data[np.lexsort(data.T),:]
  row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
  return sorted_data[row_mask]

def overlapping_check(train, valid, test):
  train_val = np.append(train, valid, axis = 0)
  data = np.append(train_val, test, axis = 0 )

  removed_data = remove_duplicate(data)
  return removed_data

train = merge_data_with_label(train_dataset, train_labels)
valid = merge_data_with_label(valid_dataset, valid_labels)
test = merge_data_with_label(test_dataset, test_labels)

refined_data = overlapping_check(train, valid, test)
new_dataset, new_labels = split_data_with_label(refined_data)
new_dataset, new_labels = randomize(new_dataset, new_labels)

total_length = len(new_dataset)

# According to Andrew Ng's recommendation 
# (Train, Test) = (7, 3)
# (Train, Valid, Test) = (6, 2, 2) 

new_train_size = int(total_length * 0.6)
new_valid_size = int(total_length * 0.2)
new_test_size = int(total_length * 0.2)

off_set = new_train_size + new_valid_size

new_train_dataset = new_dataset[:new_train_size]
new_train_labels = new_labels[:new_train_size]
new_valid_dataset = new_dataset[new_train_size:off_set]
new_valid_labels = new_labels[new_train_size:off_set]
new_test_dataset = new_dataset[off_set:]
new_test_labels = new_labels[off_set:]

display_data(new_train_dataset, 0)
display_data(new_test_dataset, 0)
display_data(new_valid_dataset, 0)

print('Training (refined):', new_train_dataset.shape, new_train_labels.shape)
print('Validation (refined):', new_valid_dataset.shape, new_valid_labels.shape)
print('Testing (refined):', new_test_dataset.shape, new_test_labels.shape)


pickle_file_refined = 'notMNIST_refined.pickle'

try:
  f = open(pickle_file_refined, 'wb')
  save = {
    'train_dataset': new_train_dataset,
    'train_labels': new_train_labels,
    'valid_dataset': new_valid_dataset,
    'valid_labels': new_valid_labels,
    'test_dataset': new_test_dataset,
    'test_labels': new_test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size (refined):', statinfo.st_size)



# ---
# Problem 6
# ---------
# 
# Let's get an idea of what an off-the-shelf classifier can give you on this data. 
# It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. 
# Hint: you can use the LogisticRegression model from sklearn.linear_model.
# Optional question: train an off-the-shelf model on all the data!
# 
# ---
LR = LogisticRegression()
train_1d = train_dataset.reshape(-1, image_size * image_size)

# Data is too big to train using cpu!
train_1d = train_1d[:1000]
train_labels = train_labels[:1000]

LR.fit(train_1d, train_labels)

valid_1d = valid_dataset.reshape(-1, image_size * image_size)

valid_accuracy = LR.score(valid_1d, valid_labels)
print("Valid set Accuracy : ", valid_accuracy)

test_1d = test_dataset.reshape(-1, image_size * image_size)

test_accuracy = LR.score(test_1d, test_labels)
print("Test set Accuracy : ", test_accuracy)

# Optional questions:
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

LR_new = LogisticRegression()
train_1d = new_train_dataset.reshape(-1, image_size * image_size)

# Data is too big to train using cpu!
train_1d = train_1d[:1000]
train_labels = new_train_labels[:1000]

LR_new.fit(train_1d, train_labels)

valid_1d = new_valid_dataset.reshape(-1, image_size * image_size)

valid_accuracy = LR_new.score(valid_1d[:10000], new_valid_labels[:10000])
print("Valid set Accuracy (refined) : ", valid_accuracy)

test_1d = new_test_dataset.reshape(-1, image_size * image_size)

test_accuracy = LR_new.score(test_1d[:10000], new_test_labels[:10000])
print("Test set Accuracy (refined): ", test_accuracy)