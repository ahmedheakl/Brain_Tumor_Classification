from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize

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
