import numpy as np
import pickle
from os import path, makedirs
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout, 
    BatchNormalization, Reshape, Activation, Lambda)
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from app.data import load_celeb, load_celeb_attr
from app import get_logger

log = get_logger(__name__)


class VarAutoencoderModel(object):
    def __init__(self, input_shape, learning_rate, use_batch_norm, use_dropout, r_loss_factor, z_dim):
        self.__input_shape = input_shape
        self.__learning_rate = learning_rate

        self.__z_dim = z_dim
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

        self.__r_loss_factor = r_loss_factor

        self.__mu = None
        self.__log_var = None

    def fit(self, data_flow, epochs, shuffle, weights_file, steps_per_epoch):
        step_size = 1
        decay_factor = 1
        initial_lr = self.__learning_rate

        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))

        lr_sched = LearningRateScheduler(schedule)
        checkpoint = ModelCheckpoint(weights_file, save_weights_only=True, verbose=1)

        return self.model().fit_generator(generator=data_flow, epochs=epochs, steps_per_epoch=steps_per_epoch, 
                                        shuffle=shuffle, callbacks=[checkpoint, lr_sched])

    def model(self):
        if self.__model is None:
            if self.__encoder_input_layer is None:
                self.__encoder_layers()

            input_layer = self.__encoder_input_layer
            output_layer = self.decoder()(self.__encoder_output_layer)

            def r_loss(x, y):
                r_loss = K.mean(K.square(x - y), axis=[1,2,3])
                return self.__r_loss_factor * r_loss

            def kl_loss(x, y):
                if self.__mu is None or self.__log_var is None:
                    self.__encoder_layers()
                kl_loss =  -0.5 * K.sum(1 + self.__log_var - K.square(self.__mu) - K.exp(self.__log_var), axis=1)
                return kl_loss

            def loss_func(x, y):
                return r_loss(x, y) + kl_loss(x, y)

            self.__model = Model(input_layer, output_layer)
            self.__model.compile(optimizer=Adam(lr=self.__learning_rate), loss=loss_func, metrics=[r_loss, kl_loss])
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

    def get_z_dim(self):
        return self.__z_dim

    def __encoder_layers(self):
        self.__encoder_input_layer = Input(shape=self.__input_shape, name='encoder_input')

        x = self.__encoder_input_layer
        x = self.__combine(Conv2D, x, filters=32, strides=2, name='encoder_conv_0')
        x = self.__combine(Conv2D, x, filters=64, strides=2, name='encoder_conv_1')
        x = self.__combine(Conv2D, x, filters=64, strides=2, name='encoder_conv_2')
        x = self.__combine(Conv2D, x, filters=64, strides=2, name='encoder_conv_3')

        self.__shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        self.__mu = Dense(self.get_z_dim(), name='mu')(x)
        self.__log_var = Dense(self.get_z_dim(), name='log_var')(x)

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        self.__encoder_output_layer = Lambda(sampling, name='encoder_output')([self.__mu, self.__log_var])

    def __decoder_layers(self):
        if self.__shape_before_flattening is None:
            self.__encoder_layers()

        self.__decoder_input_layer = Input(shape=(self.get_z_dim(),), name='decoder_input')

        x = self.__decoder_input_layer
        x = Dense(np.prod(self.__shape_before_flattening))(x)
        x = Reshape(self.__shape_before_flattening)(x)
        
        x = self.__combine(Conv2DTranspose, x, filters=64, strides=2, name='decoder_conv_0')
        x = self.__combine(Conv2DTranspose, x, filters=64, strides=2, name='decoder_conv_1')
        x = self.__combine(Conv2DTranspose, x, filters=32, strides=2, name='decoder_conv_2')        
        x = Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same', name='encoder_conv_3')(x)

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
                self.__r_loss_factor,
                self.get_z_dim(),
                ], f)

    def load_weights(self, filepath):
        self.model().load_weights(filepath)

    def reconstruct_images(self, data):
        data_size = len(data)
        z_points = self.encoder().predict(data)

        fig = plt.figure(figsize=(15, 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        num_rows = 2
        offset = 0

        for i in range(data_size):
            offset += 1
            img = data[i].squeeze()
            ax = fig.add_subplot(num_rows, data_size, offset)
            ax.axis('off')
            ax.imshow(img, cmap='gray_r')

        self.decoded_images(z_points, fig, num_rows, data_size)

    def decoded_images(self, z_points, fig=None, num_rows=1, offset=0):
        if fig is None:
            fig = plt.figure(figsize=(15, 3))

        reconst_images = self.decoder().predict(z_points)
        data_size = len(reconst_images)

        for i in range(data_size):
            offset += 1
            img = reconst_images[i].squeeze()
            ax = fig.add_subplot(num_rows, data_size, offset)
            ax.axis('off')
            # ax.text(0.5, 1.2, str(np.round(z_points[i],1)), fontsize=10, ha='center', transform=ax.transAxes)   
            ax.imshow(img, cmap='gray_r')

    def latent_space(self, images_flow):
        x = np.linspace(-3, 3, 100)
        z_points = self.encoder().predict_generator(images_flow, steps=20, verbose=1)

        fig = plt.figure(figsize=(7, 7))
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        for i in range(50):
            ax = fig.add_subplot(5, 10, i+1)
            ax.hist(z_points[:, i], density=True, bins=20)
            ax.axis('off')
            ax.text(0.5, -0.35, str(i+1), fontsize=10, ha='center', transform=ax.transAxes)
            ax.plot(x, norm.pdf(x))

    def get_vector_from_label(self, data_flow_label):
        class Current(object):
            _sum = np.zeros(shape=self.get_z_dim(), dtype='float32')
            _len = 0
            _mean = np.zeros(shape=self.get_z_dim(), dtype='float32')

        current_pos = Current()
        current_neg = Current()

        current_vector = np.zeros(shape=self.get_z_dim(), dtype='float32')
        current_dist = 0

        while(current_pos._len < 10000):
            batch = next(data_flow_label)
            im = batch[0]
            attribute = batch[1]

            z_points = self.encoder().predict(np.array(im))

            z_points_pos = z_points[attribute==1]
            z_points_neg = z_points[attribute==-1]

            if len(z_points_pos) > 0:
                current_pos._sum += np.sum(z_points_pos, axis=0)
                current_pos._len += len(z_points_pos)
                new_mean_pos = current_pos._sum / current_pos._len
                movement_pos = np.linalg.norm(new_mean_pos-current_pos._mean)

            if len(z_points_neg) > 0: 
                current_neg._sum += np.sum(z_points_neg, axis=0)
                current_neg._len += len(z_points_neg)
                new_mean_neg = current_neg._sum / current_neg._len
                movement_neg = np.linalg.norm(new_mean_neg-current_neg._mean)

            current_vector = new_mean_pos-new_mean_neg
            new_dist = np.linalg.norm(current_vector)
            dist_change = new_dist - current_dist

            log.info(f'images: {current_pos._len:5d}, '
                         f'pos-move: {movement_pos:.3f}, '
                         f'neg-move: {movement_neg:.3f}, '
                         f'dist: {new_dist:.3f}, '
                         f'delta-dist: {dist_change:.3f}'
                         )

            current_pos._mean = np.copy(new_mean_pos)
            current_neg._mean = np.copy(new_mean_neg)
            current_dist = np.copy(new_dist)

            if np.sum([movement_pos, movement_neg]) < 0.08:
                current_vector = current_vector / current_dist
                log.info(f'Found the {label} vector')
                break

        return current_vector   

    def add_vector_to_images(self, images, feature_vec):
        z_points = self.encoder().predict(images)
        fig = plt.figure(figsize=(18, 10))
        counter = 1
        factors = range(-4, 5)

        for i, image in enumerate(images):
            img = image.squeeze()
            sub = fig.add_subplot(len(images), len(factors) + 1, counter)
            sub.axis('off')
            sub.imshow(img)
            counter += 1

            for factor in factors:
                changed_z_point = z_points[i] + feature_vec * factor
                changed_image = self.decoder().predict(np.array([changed_z_point]))

                img = changed_image[0].squeeze()
                sub = fig.add_subplot(len(images), len(factors) + 1, counter)
                sub.axis('off')
                sub.imshow(img)
                counter += 1


if __name__ == "__main__":
    import argparse
    from app import PARAMS_DIR, WEIGHTS_DIR, VECTORS_DIR

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

    filename = path.basename(__file__)[:-3]
    params_file = path.join(PARAMS_DIR, f'{filename}_params.pkl')
    weights_file = path.join(WEIGHTS_DIR, f'{filename}_weights.h5')

    input_shape = (128,128,3)
    batch_size = 32
    shuffle = True
    target_size = input_shape[:2]

    data_flow, num_images = load_celeb(target_size=target_size, batch_size=batch_size, shuffle=shuffle)

    if path.isfile(params_file) and path.isfile(weights_file) and not args.overwrite:
        with open(params_file, 'rb') as f:
            params = pickle.load(f)
        vae = VarAutoencoderModel(*params)
        vae.load_weights(weights_file)
    else:
        vae = VarAutoencoderModel(
            input_shape=input_shape, 
            learning_rate=0.0005,
            use_batch_norm=True,
            use_dropout=True,
            r_loss_factor=10000,
            z_dim=200,
            )

        vae.save(params_file)

    vae.model().summary()

    if args.train:
        steps_per_epoch = num_images / batch_size
        vae.fit(data_flow=data_flow, epochs=200, shuffle=True, weights_file=weights_file, steps_per_epoch=steps_per_epoch)

    # reconstruct 10 images
    num_show = 10
    images_flow = load_celeb_attr(target_size=target_size, batch_size=num_show, shuffle=shuffle)
    batch = next(images_flow)
    example_images = batch[0]
    vae.reconstruct_images(data=example_images)

    # plot latent space distributions
    vae.latent_space(images_flow)

    # randomly generate numbers in a reasonable range in the latent space, and construct their corresponding images
    rand_z_points = np.random.normal(size=(num_show, vae.get_z_dim()))
    vae.decoded_images(rand_z_points)

    # add vectors (smiling, eyeglasse, attractive, ...) to images
    vectors_file = path.join(VECTORS_DIR, f'{filename}_vectors.pkl')
    if path.isfile(vectors_file):
        with open(vectors_file, 'rb') as f:
            vectors = pickle.load(f)
    else:
        vectors = {}
        for label in ['Attractive', 'Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick', 'High_Cheekbones', 'Male', 'Eyeglasses', 'Blond_Hair']:  
            data_flow_label = load_celeb_attr(target_size=target_size, batch_size=500, shuffle=shuffle, label=label)
            vectors[label] = vae.get_vector_from_label(data_flow_label=data_flow_label)
        with open(vectors_file, 'wb') as f:
            pickle.dump(vectors, f)

    vae.add_vector_to_images(example_images[:5], vectors['Eyeglasses'])

    plt.show()

