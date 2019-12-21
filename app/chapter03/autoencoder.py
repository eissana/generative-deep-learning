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
        self.__encoder = None

        self.__decoder_input_layer = None
        self.__decoder_output_layer = None
        self.__decoder = None

        self.__model = None

        self.__use_batch_norm = use_batch_norm
        self.__use_dropout = use_dropout

    def fit(self, x, *args):
        return self.model().fit(x=x, y=x, *args)

    def model(self):
        if self.__model is None:
            if self.__encoder_input_layer is None:
                self.__encoder_layers()

            input_layer = self.__encoder_input_layer
            output_layer = self.decoder()(self.__encoder_output_layer)

            self.__model = Model(input_layer, output_layer)
            self.__model.compile(
                loss=lambda x, y: K.mean(K.square(x - y), axis = [1,2,3]), 
                optimizer=Adam(lr=self.__learning_rate),
            )
        return self.__model

    def plot_model(self):
        from app.__init__ import MODELS_PLOT_DIR

        filepath = path.join(MODELS_PLOT_DIR, path.dirname(__file__))

        if not path.exists(filepath):
            makedirs(filepath)

        plot_model(self.model(), to_file=path.join(filepath ,'model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder(), to_file=path.join(filepath ,'encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder(), to_file=path.join(filepath ,'decoder.png'), show_shapes = True, show_layer_names = True)

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


    def reconstruct_images(self, num_show, test_x)
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


if __name__ == "__main__":
    from app.__init__ import MODELS_DIR

    model_file = path.join(f"{MODELS_DIR}", f"{__file__.split('.')[0]}.h5")

    if not path.exists(path.dirname(model_file)):
        makedirs(path.dirname(model_file))

    (train_x, train_y), (test_x, test_y) = load_mnist()

    if path.isfile(model_file):
        with open(model_file, 'rb') as f:
            ae = pickle.load(f)
    else:
        ae = AutoencoderModel(
            input_shape=train_x.shape[1:], 
            learning_rate=0.0005,
            use_batch_norm=False,
            use_dropout=False)
        ae.fit(x=train_x[:1000], batch_size=32, epochs=2, shuffle=True)

        with open(model_file, 'wb') as f:
            pickle.dump(ae, f)

    ae.model().summary()
    ae.plot_model()
    ae.reconst_images(num_show=10, test_x)
