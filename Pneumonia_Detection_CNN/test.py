import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from zipfile import ZipFile
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import model_from_json

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Hyperparameters
IMG_SIZE = 256
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Validation Image selection and Preprocessing
X = []

img = "C:/Users/User/Desktop/Pneumonia/Pneumonia_Detection_CNN/chest_xray/val/PNEUMONIA/person1952_bacteria_4883.jpeg"

img_cv2 = cv2.imread(img)
X.append(cv2.resize(img_cv2, (IMG_SIZE, IMG_SIZE)))

X = np.asarray(X)
print(X.shape)

# Creating Model
base_model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False, weights="imagenet", input_shape=IMG_SHAPE, pooling="max"
)

model = keras.Sequential(
    [
        layers.InputLayer(shape=IMG_SHAPE),
        base_model,
        layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        layers.Dense(256, activation="relu"),
        layers.Dropout(rate=0.45, seed=123),
        layers.Dense(2, activation="softmax"),
    ]
)


# Loading Weights
model.load_weights("model_weights.weights.h5")


# Predictions
output_class = model.predict(X)

print("Output class: ", output_class)
