import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical


MAX_PIX_VAL = 255.0
CIFAR_CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
NUM_CIFAR_CLASSES = CIFAR_CLASSES.size


def load_cifar10():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_x = train_x.astype('float32') / MAX_PIX_VAL
    test_x = test_x.astype('float32') / MAX_PIX_VAL

    train_y = to_categorical(train_y, NUM_CIFAR_CLASSES)
    test_y = to_categorical(test_y, NUM_CIFAR_CLASSES)

    return (train_x, train_y), (test_x, test_y)

