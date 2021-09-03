from inspect import CO_VARARGS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.engine.input_layer import Input


def Model(INPUT_SIZE=64):

    # Building the model layers
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3),
               input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=32, kernel_size=(3, 3),),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3),),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(64),
        Activation('relu'),
        Dropout(0.5),

        # Used only neuron for output and sigmoid
        # becaues iam using binary loss entropy
        # Dense(1),
        Dense(2),
        Activation('softmax'),
        # Activation('sigmoid')

    ])

    # Compiling the model with the loss and optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model
