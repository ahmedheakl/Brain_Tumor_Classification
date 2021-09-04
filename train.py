from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
from model import Model

# Loading the saved dataset
dataset = np.load('dataset.npy')
labels = np.load('labels.npy')


# Splitting the data into test and train
X_train, X_test, y_train, y_test = train_test_split(
    dataset, labels, random_state=0, test_size=0.2, shuffle=True)

# Take a look at the shape (n_samples, img_width, img_height, n_channels)
print(X_train.shape)

# Normalizing the images for better training performance
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# Coverting the data into categorical data (from binary)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Initializing the model
model = Model()
model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=10,
          validation_data=(X_test, y_test), shuffle=False)

model.save('CategoricalBrainModel10epochs.h4')

# Note: the categorical model and data had a better acc than the binary
# Categorical: 0.9942
# Binary: 0.9825
# Hence, I will stick with the categorical one!
