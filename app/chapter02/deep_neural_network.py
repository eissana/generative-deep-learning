import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense
from keras.models import Model, load_model
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

def model(input_shape, num_classes, learning_rate):
    input_layer = Input(shape=input_shape)
    x = Flatten()(input_layer)
    x = Dense(units=200, activation='relu')(x)
    x = Dense(units=150, activation='relu')(x)
    output_layer = Dense(units=num_classes, activation='softmax')(x)

    m = Model(input_layer, output_layer)
    m.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy'],
    )
    return m

if __name__ == "__main__":
    import os.path

    model_file = "models/nn.h5"
    (train_x, train_y), (test_x, test_y) = load_cifar10()

    if os.path.isfile(model_file):
        model = load_model(model_file)
    else:
        model = model(input_shape=train_x.shape[1:], num_classes=train_y.shape[1], learning_rate=0.0005)
        model.summary()
        model.fit(train_x, train_y, batch_size=32, epochs=10, shuffle=True)
        model.save(model_file)

    model.evaluate(test_x, test_y)
