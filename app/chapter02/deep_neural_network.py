import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense
from keras.models import Model


class NeuralNetwork(object):
    __NUMC_LASSES = 10
    __MAX_PIX_VAL = 255.0

    def load_data(self):
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()

        train_x = train_x.astype('float32') / self.__MAX_PIX_VAL
        test_x = test_x.astype('float32') / self.__MAX_PIX_VAL

        train_y = to_categorical(train_y, self.__NUMC_LASSES)
        test_y = to_categorical(test_y, self.__NUMC_LASSES)

        return (train_x, train_y), (test_x, test_y)

    def build_model(self, input_shape, num_out_nodes):
        input_layer = Input(shape=input_shape)
        x = Flatten()(input_layer)
        x = Dense(units=200, activation='relu')(x)
        x = Dense(units=150, activation='relu')(x)
        output_layer = Dense(units=num_out_nodes, activation='softmax')(x)

        return Model(input_layer, output_layer)


if __name__ == "__main__":
    net = NeuralNetwork()
    (train_x, train_y), (test_x, test_y) = net.load_data()

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    model = net.build_model(input_shape=train_x.shape[1:], num_out_nodes=train_y.shape[1])
    model.summary()
