import numpy as np
import pandas as pd
from glob import glob
from os import path

from keras.preprocessing.image import ImageDataGenerator

from app import DATA_DIR


MAX_PIX_VAL = 255.0
CIFAR_CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
NUM_CIFAR_CLASSES = CIFAR_CLASSES.size


def load_cifar10():
    from keras.datasets import cifar10
    from keras.utils import to_categorical

    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_x = train_x.astype('float32') / MAX_PIX_VAL
    test_x = test_x.astype('float32') / MAX_PIX_VAL

    train_y = to_categorical(train_y, NUM_CIFAR_CLASSES)
    test_y = to_categorical(test_y, NUM_CIFAR_CLASSES)

    return (train_x, train_y), (test_x, test_y)


def load_mnist():
    from keras.datasets import mnist

    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_x = train_x.astype('float32') / MAX_PIX_VAL
    train_x = train_x.reshape(train_x.shape + (1,))
    test_x = test_x.astype('float32') / MAX_PIX_VAL
    test_x = test_x.reshape(test_x.shape + (1,))

    return (train_x, train_y), (test_x, test_y)


def load_celeb(target_size, batch_size, shuffle):
    image_dir = path.join(DATA_DIR, 'celeb')
    filenames = np.array(glob(path.join(image_dir, '*/*.jpg')))

    data_gen = ImageDataGenerator(rescale=1./255.)
    data_flow = data_gen.flow_from_directory(directory=image_dir, 
                                             target_size=target_size, batch_size=batch_size, shuffle=shuffle, 
                                             class_mode='input', subset='training')
    num_images = len(filenames)
    return data_flow, num_images


def load_celeb_attr(target_size, batch_size, shuffle, label=None):
    index_col = 'image_id'
    attr_file = path.join(DATA_DIR, 'celeb', 'list_attr_celeba.csv')

    dataframe = pd.read_csv(attr_file, skiprows=[0], delim_whitespace=True)
    dataframe[index_col] = dataframe.index

    image_dir = path.join(DATA_DIR, 'celeb', 'img_align_celeba')
    data_gen = ImageDataGenerator(rescale=1. / MAX_PIX_VAL)

    if label is None:
        return data_gen.flow_from_dataframe(dataframe=dataframe, directory=image_dir, x_col=index_col,
                                            target_size=target_size, batch_size=batch_size, shuffle=shuffle, 
                                            class_mode='input')
    return data_gen.flow_from_dataframe(dataframe=dataframe, directory=image_dir, x_col=index_col, y_col=label,
                                        target_size=target_size, batch_size=batch_size, shuffle=shuffle, 
                                        class_mode='raw')
