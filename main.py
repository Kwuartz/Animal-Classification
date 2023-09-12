import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
  
import warnings
warnings.filterwarnings('ignore')
  
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image_dataset_from_directory
  
import os
import matplotlib.image as mpimg

trainDirectory = "train_set"
testDirectory = "test_set"

def createModel():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
    
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.1),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def trainModel(model):
    trainDatagen = image_dataset_from_directory(
        trainDirectory,
        image_size=(200,200),
        subset='training',
        seed = 1,
        validation_split=0.1,
        batch_size= 32
    )

    testDatagen = image_dataset_from_directory(
        trainDirectory,
        image_size=(200,200),
        subset='validation',
        seed = 1,
        validation_split=0.1,
        batch_size= 32
    )
    
    history = model.fit(
          trainDatagen,
          epochs=10,
          validation_data=testDatagen
    )

    return model, history

def predict(model, image):
    result = model.predict(image)

    if result >= 0.5:
        print("Dog")
    else:
        print("Cat")

model = createModel()
model, history = trainModel(model)
model.save("cats-dogs.keras")
