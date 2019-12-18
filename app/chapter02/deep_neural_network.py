import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam


MAX_PIX_VAL = 255.0
NUM_CLASSES = 10


def load_cifar10():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    train_x = train_x.astype('float32') / MAX_PIX_VAL
    test_x = test_x.astype('float32') / MAX_PIX_VAL

    train_y = to_categorical(train_y, NUM_CLASSES)
    test_y = to_categorical(test_y, NUM_CLASSES)

    return (train_x, train_y), (test_x, test_y)

class NeuralNetwork(object):
    def __init__(self, input_shape, num_classes, learning_rate):
        self.__input_shape = input_shape
        self.__num_out_nodes = num_classes
        self.__learning_rate = learning_rate

        self.__model = None

    def build_model(self):
        input_layer = Input(shape=self.__input_shape)
        x = Flatten()(input_layer)
        x = Dense(units=200, activation='relu')(x)
        x = Dense(units=150, activation='relu')(x)
        output_layer = Dense(units=self.__num_out_nodes, activation='softmax')(x)

        self.__model = Model(input_layer, output_layer)
        self.__model.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(lr=self.__learning_rate),
            metrics=['accuracy'],
        )
        return self.__model

if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = load_cifar10()
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    net = NeuralNetwork(input_shape=train_x.shape[1:], num_classes=train_y.shape[1], learning_rate=0.01)
    model = net.build_model()
    model.summary()
