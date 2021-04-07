import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_size = (180, 180)

for img_name in os.listdir("PetImages/Cat"):
    img_path = os.path.join("PetImages/Cat",img_name)
    img = keras.preprocessing.image.load_img(img_path,target_size=image_size)
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.show()

