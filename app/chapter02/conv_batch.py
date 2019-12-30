from keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam

from app.data import load_cifar10
from app.plot import plot_sample_images


def conv_batch_model(input_shape, num_classes, learning_rate):
    input_layer = Input(shape=input_shape)
    x = __combine(input_layer, filters=32, strides=1)
    x = __combine(x, filters=32, strides=2)
    x = __combine(x, filters=64, strides=1)
    x = __combine(x, filters=64, strides=2)
    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(units=num_classes, activation='softmax')(x)

    m = Model(input_layer, output_layer)
    m.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy'],
    )
    return m

def __combine(layer, filters, strides):
    x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')(layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


if __name__ == "__main__":
    import os.path
    from app import MODELS_DIR

    model_file = f"{MODELS_DIR}/conv_batch.h5"
    (train_x, train_y), (test_x, test_y) = load_cifar10()

    if os.path.isfile(model_file):
        model = load_model(model_file)
    else:
        model = conv_batch_model(input_shape=train_x.shape[1:], num_classes=train_y.shape[1], learning_rate=0.0005)
        model.summary()
        model.fit(train_x, train_y, batch_size=32, epochs=10, shuffle=True)
        model.save(model_file)

    model.evaluate(test_x, test_y)
    predictions = model.predict(test_x)

    plot_sample_images(test_x, test_y, predictions)
