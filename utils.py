import cv2
from PIL import Image


def customize(path, INPUT_SIZE=64):
    # Read the image through open-cv
    image = cv2.imread(path)

    # Convert the image into RGB
    image = Image.fromarray(image, 'RGB')

    # Resize all the images into a fixed shape
    image = image.resize((INPUT_SIZE, INPUT_SIZE))

    return image
