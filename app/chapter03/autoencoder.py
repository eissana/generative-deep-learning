import numpy as np
import pickle
from os import path, makedirs
import matplotlib.pyplot as plt

from keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout, 
    BatchNormalization, Reshape, Activation)
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 

from app.data import load_mnist


class AutoencoderModel(object):
    def __init__(self, input_shape, learning_rate, use_batch_norm, use_dropout):
        self.__input_shape = input_shape
        self.__learning_rate = learning_rate

        self.__z_dim = 2
        self.__shape_before_flattening = None

        self.__encoder_input_layer = None
        self.__encoder_output_layer = None
        self.__encoder = None

        self.__decoder_input_layer = None
        self.__decoder_output_layer = None
        self.__decoder = None

        self.__model = None

        self.__use_batch_norm = use_batch_norm
        self.__use_dropout = use_dropout

    def fit(self, x, epochs, batch_size, shuffle, weights_file):
        checkpoint = ModelCheckpoint(weights_file, save_weights_only=True, verbose=1)
        return ae.model().fit(x=x, y=x, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=[checkpoint])

    def model(self):
        if self.__model is None:
            if self.__encoder_input_layer is None:
                self.__encoder_layers()

            input_layer = self.__encoder_input_layer
            output_layer = self.decoder()(self.__encoder_output_layer)

            self.__model = Model(input_layer, output_layer)
            self.__model.compile(optimizer=Adam(lr=self.__learning_rate), loss=loss_func)
        return self.__model

    def encoder(self):
        if self.__encoder is None:
            if self.__encoder_input_layer is None:
                self.__encoder_layers()

            self.__encoder = Model(self.__encoder_input_layer, self.__encoder_output_layer)
        return self.__encoder

    def decoder(self):
        if self.__decoder is None:
            if self.__decoder_input_layer is None:
                self.__decoder_layers()

            self.__decoder = Model(self.__decoder_input_layer, self.__decoder_output_layer)
        return self.__decoder

    def __encoder_layers(self):
        self.__encoder_input_layer = Input(shape=self.__input_shape, name='encoder_input')

        x = self.__encoder_input_layer
        x = self.__combine(Conv2D, x, filters=32, strides=1, name='encoder_conv_0')
        x = self.__combine(Conv2D, x, filters=64, strides=2, name='encoder_conv_1')
        x = self.__combine(Conv2D, x, filters=64, strides=2, name='encoder_conv_2')
        x = self.__combine(Conv2D, x, filters=64, strides=1, name='encoder_conv_3')

        self.__shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.__encoder_output_layer = Dense(units=self.__z_dim, name='encoder_output')(x)

    def __decoder_layers(self):
        if self.__shape_before_flattening is None:
            self.__encoder_layers()

        self.__decoder_input_layer = Input(shape=(self.__z_dim,), name='decoder_input')

        x = self.__decoder_input_layer
        x = Dense(np.prod(self.__shape_before_flattening))(x)
        x = Reshape(self.__shape_before_flattening)(x)
        
        x = self.__combine(Conv2DTranspose, x, filters=64, strides=1, name='decoder_conv_0')
        x = self.__combine(Conv2DTranspose, x, filters=64, strides=2, name='decoder_conv_1')
        x = self.__combine(Conv2DTranspose, x, filters=32, strides=2, name='decoder_conv_2')        
        x = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', name='encoder_conv_3')(x)

        self.__decoder_output_layer = Activation('sigmoid')(x)

    def __combine(self, conv_type, layer, filters, strides, name):
        x = conv_type(filters=filters, kernel_size=3, strides=strides, padding='same', name=name)(layer)
        x = LeakyReLU()(x)
        if self.__use_batch_norm:
            x = BatchNormalization()(x)
        if self.__use_dropout:
            x = Dropout(rate = 0.25)(x)
        return x


    def save(self, file):
        _ = self.model() # make sure model in built
        with open(file, 'wb') as f:
            pickle.dump([
                self.__input_shape,
                self.__learning_rate,
                self.__use_batch_norm,
                self.__use_dropout,
                ], f)

    def load_weights(self, filepath):
        self.model().load_weights(filepath)

    def reconstruct_images(self, num_show, test_x):
        example_idx = np.random.choice(range(len(test_x)), num_show)
        example_images = test_x[example_idx]

        z_points = self.encoder().predict(example_images)
        reconst_images = self.decoder().predict(z_points)

        fig = plt.figure(figsize=(15, 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i in range(num_show):
            img = example_images[i].squeeze()
            ax = fig.add_subplot(2, num_show, i+1)
            ax.axis('off')
            ax.text(0.5, -0.35, str(np.round(z_points[i],1)), fontsize=10, ha='center', transform=ax.transAxes)   
            ax.imshow(img, cmap='gray_r')

        for i in range(num_show):
            img = reconst_images[i].squeeze()
            ax = fig.add_subplot(2, num_show, i+num_show+1)
            ax.axis('off')
            ax.imshow(img, cmap='gray_r')

    def latent_space(self, num_show, test_x, test_y):
        example_idx = np.random.choice(range(len(test_x)), num_show)
        example_images = test_x[example_idx]
        example_labels = test_y[example_idx]

        z_points = self.encoder().predict(example_images)

        plt.figure(figsize=(7, 7))
        plt.scatter(z_points[:, 0] , z_points[:, 1], cmap='rainbow', c=example_labels, alpha=0.5, s=2)
        plt.colorbar()

def loss_func(x, y):
    return K.mean(K.square(x - y), axis = [1,2,3])


if __name__ == "__main__":
    import argparse
    from app.__init__ import PARAMS_DIR, WEIGHTS_DIR

    parser = argparse.ArgumentParser()

    overwrite_parser = parser.add_mutually_exclusive_group(required=False)
    overwrite_parser.add_argument('--overwrite', dest='overwrite', action='store_true', help="overwrite model and parameters")
    overwrite_parser.add_argument('--no-overwrite', dest='overwrite', action='store_false', help="load model from file if exists")
    parser.set_defaults(overwrite=False)

    train_parser = parser.add_mutually_exclusive_group(required=False)
    train_parser.add_argument("--train", dest='train', action='store_true', help="continue training model")
    train_parser.add_argument("--no-train", dest='train', action='store_false', help="don't continue training")
    parser.set_defaults(train=True)

    args = parser.parse_args()

    params_file = path.join(PARAMS_DIR, 'ae_params.pkl')
    weights_file = path.join(WEIGHTS_DIR, 'weights.h5')

    (train_x, train_y), (test_x, test_y) = load_mnist()

    if path.isfile(params_file) and path.isfile(weights_file) and not args.overwrite:
        with open(params_file, 'rb') as f:
            params = pickle.load(f)
        ae = AutoencoderModel(*params)
        ae.load_weights(weights_file)
    else:
        ae = AutoencoderModel(
            input_shape=train_x.shape[1:], 
            learning_rate=0.0005,
            use_batch_norm=False,
            use_dropout=False)

        ae.save(params_file)

    ae.model().summary()

    if args.train:
        ae.fit(x=train_x, epochs=200, batch_size=32, shuffle=True, weights_file=weights_file)

    ae.reconstruct_images(num_show=10, test_x=test_x)
    ae.latent_space(num_show=5000, test_x=test_x, test_y=test_y)
    plt.show()

