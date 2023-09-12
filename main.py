import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
  
import warnings
warnings.filterwarnings('ignore')
  
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image, image_dataset_from_directory
  
import os
import matplotlib.image as mpimg

trainDirectory = "training_set"
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

def multiPredict(model, directory, amount=999999):
    for index, file in enumerate(os.listdir(directory)):
        if index > amount:
            break

        testImage = image.load_img(directory + "/" + file, target_size=(200,200))        
        testImage = image.img_to_array(testImage)
        testImage = np.expand_dims(testImage,axis=0)

        predict(model, testImage)

def predict(model, testImage):
    result = model.predict(testImage)

    if result >= 0.5:
        print("Dog")
    else:
        print("Cat")

def plotHistory(history):
    historyDF = pd.DataFrame(history.history)
    historyDF.loc[:, ['loss', 'val_loss']].plot()
    historyDF.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()

model = createModel()
newModel, history = trainModel(model)
# model = load_model("cats-dogs.keras")
multiPredict(newModel, "test_set/cats", 100)
newModel.save("cats-dogs.keras")
newModel.save_weights("cats-dogs-weights.h5")