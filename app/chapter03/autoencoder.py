import numpy as np
from os import path, makedirs

from keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout, 
    BatchNormalization, Reshape, Activation)
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam

from app.data import load_mnist


class AutoencoderModel(object):
    def __init__(self, input_shape, learning_rate, use_batch_norm, use_dropout):
        self.__input_shape = input_shape
        self.__learning_rate = learning_rate

        self.__z_dim = 2
        self.__shape_before_flattening = None

        self.__encoder_input_layer = None
        self.__encoder_output_layer = None

        self.__decoder_input_layer = None
        self.__decoder_output_layer = None

        self.__use_batch_norm = use_batch_norm
        self.__use_dropout = use_dropout

    def model(self):
        if self.__encoder_input_layer is None:
            self.__encoder_layers()

        if self.__decoder_input_layer is None:
            self.__decoder_layers()

        decoder_model = Model(self.__decoder_input_layer, self.__decoder_output_layer)

        input_layer = self.__encoder_input_layer
        output_layer = decoder_model(self.__encoder_output_layer)

        m = Model(input_layer, output_layer)
        m.compile(
            loss=lambda x, y: K.mean(K.square(x - y), axis = [1,2,3]), 
            optimizer=Adam(lr=self.__learning_rate),
        )
        return m

    def plot_model(self):
        from app.__init__ import MODELS_PLOT_DIR

        filepath = path.join(MODELS_PLOT_DIR, path.dirname(__file__))

        if not path.exists(filepath):
            makedirs(filepath)

        plot_model(self.model(), to_file=path.join(filepath ,'model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder_model(), to_file=path.join(filepath ,'encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder_model(), to_file=path.join(filepath ,'decoder.png'), show_shapes = True, show_layer_names = True)

    def encoder_model(self):
        if self.__encoder_input_layer is None:
            self.__encoder_layers()

        m = Model(self.__encoder_input_layer, self.__encoder_output_layer)
        return m

    def decoder_model(self):
        if self.__decoder_input_layer is None:
            self.__decoder_layers()

        m = Model(self.__decoder_input_layer, self.__decoder_output_layer)
        return m

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
        x = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='encoder_conv_3')(x)

        self.__decoder_output_layer = Activation('sigmoid')(x)

    def __combine(self, conv_type, layer, filters, strides, name):
        x = conv_type(filters=filters, kernel_size=3, strides=strides, padding='same', name=name)(layer)
        x = LeakyReLU()(x)
        if self.__use_batch_norm:
            x = BatchNormalization()(x)
        if self.__use_dropout:
            x = Dropout(rate = 0.25)(x)
        return x

if __name__ == "__main__":
    from app.__init__ import MODELS_DIR

    model_file = path.join(f"{MODELS_DIR}", f"{__file__.split('.')[0]}.h5")

    if not path.exists(path.dirname(model_file)):
        makedirs(path.dirname(model_file))

    (train_x, train_y), (test_x, test_y) = load_mnist()

    if path.isfile(model_file):
        model = load_model(model_file)
    else:
        ae = AutoencoderModel(
            input_shape=train_x.shape[1:], 
            learning_rate=0.0005,
            use_batch_norm=False,
            use_dropout=False)
        model = ae.model()
        model.summary()
        model.fit(train_x[:1000], train_x[:1000], batch_size=32, epochs=2, shuffle=True)
        model.save(model_file)

    ae.plot_model()


