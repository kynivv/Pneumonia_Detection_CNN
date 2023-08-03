import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from keras import layers
from keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')


# Hyperparameters
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 25
BATCH_SIZE = 60
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
RANDOM_STATE = 42


# Data Import
X = []
Y = []

train_dir = 'chest_xray/train'
test_dir = 'chest_xray/test'

classes = os.listdir(train_dir)


# Data Preprocessing
resize_list = [train_dir, test_dir]

for p in range(len(resize_list)):
    for i, name in enumerate(classes):
        images = glob(f'{resize_list[p]}/{name}/*.jpeg')

        for image in images:
            img = cv2.imread(image)

            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            Y.append(i)

X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values


# Training Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_encoded_Y, test_size= SPLIT, random_state= RANDOM_STATE)


# Creating Model
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= 'imagenet', input_shape= IMG_SHAPE, pooling= 'max')

model = keras.Sequential([
    base_model,
    layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    layers.Dense(256, activation= 'relu'),
    layers.Dropout(rate= 0.45, seed= 123),
    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
print(model.summary())


# Callbacks
checkpoint = ModelCheckpoint('Output/model_checkpoint.h5',
                             save_best_only= True,
                             verbose= 1,
                             save_weights_only= True,
                             monitor= 'val_accuracy')


# Model Training
history = model.fit(X_train, Y_train,
                    validation_data= (X_test, Y_test),
                    batch_size= BATCH_SIZE,
                    epochs= EPOCHS,
                    verbose= 1,
                    callbacks= checkpoint
                    )