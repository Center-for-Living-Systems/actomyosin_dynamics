import tensorflow as tf
import numpy as np
import tifffile
import os

def download_images():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    return (train_images, train_labels), (test_images, test_labels)


def normalize(train_images, test_images):
    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    return train_images, test_images


def load_dataset(batch_size=100):
    
    data_dir = '/net/projects/CLS/actomyosin_dynamics/data/LifeAct-NMY2-GFP_NMY2_wt_patchsize_96_p95'
    patch_size = 96
    
    filenames = [x for x in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, x)) and ('.tif' in x)]

    num_of_samples = len(filenames)
    
    train_images = np.zeros([num_of_samples, patch_size, patch_size])
    train_labels= np.zeros([num_of_samples])
    
    for filename_ind in range(filenames):
        filename = filenames[filename_ind]               
        train_img = tifffile.imread(os.path.join(data_dir,filename))
        train_images[filename_ind,:,:] = train_img
        train_labels[filename_ind] = 0
    
    train_images = normalize(train_images)

    TRAIN_BUF = 60000
    TEST_BUF = 10000

    BATCH_SIZE = batch_size

    train_dataset_image = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
    train_dataset_label = tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((train_dataset_image, train_dataset_label)).shuffle(TRAIN_BUF)
 
    return train_dataset
