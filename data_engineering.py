import cv2
import os
from utils import customize
import matplotlib.pyplot as plt
import numpy as np


dir_path = 'dataset/'

# Load the images names
labels_names = os.listdir(dir_path)

dataset = []
labels = []

for label in labels_names:
    # List all the images in each directory
    images_names = os.listdir(dir_path + label + '/')

    fig, axs = plt.subplots(3, 1)
    for idx, image_name in enumerate(images_names):
        # Applying the changes to each image(reac, to_RGB, resize)
        image_path = dir_path + label + '/' + image_name
        image = customize(image_path)

        # Appeding each image with its label to our dataset
        dataset.append(np.array(image))
        labels.append(1 if label == 'yes' else 0)

        # Displaying Some sample images
        if idx in [1, 2, 3]:
            axs[idx-1].imshow(image)
            fig.suptitle(f'This is {label.upper()} tumor images')
    plt.show()

# Converting the dataset into a numpy array for easier usage (instead of list)
dataset = np.array(dataset)
labels = np.array(labels)

# Saving the dataset
np.save('dataset.npy', dataset)
np.save('labels.npy', labels)
