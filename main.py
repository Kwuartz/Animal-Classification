import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import random
import os
import math

import warnings
warnings.filterwarnings('ignore')
  
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras import backend as K

trainDirectory = "training_set_small"
validationDirectory = "test_set_small"
trainSamples = len(os.listdir(trainDirectory))
validationSamples = len(os.listdir(validationDirectory))
epochs = 11
batch_size = 16
img_width, img_height = 224, 224

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def createModel():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    return model

def trainModel(model):
    trainDatagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
 
    testDatagen = ImageDataGenerator(
        rescale=1. / 255
    )
 
    trainGenerator = trainDatagen.flow_from_directory(
        trainDirectory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
 
    validationGenerator = testDatagen.flow_from_directory(
        validationDirectory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
 
    history = model.fit_generator(
        trainGenerator,
        steps_per_epoch=trainSamples // batch_size,
        epochs=epochs,
        validation_data=validationGenerator,
        validation_steps=validationSamples // batch_size
    )
    
    return model, history

def multiPredict(model,  amount=math.inf):
    classesDirectories = os.listdir(validationDirectory)
    imageDirectories = []

    for classDirectory in classesDirectories:
        imagesInClass = [validationDirectory + "/" + classDirectory + "/" + imageDirectory for imageDirectory in os.listdir(validationDirectory + "/" + classDirectory)]
        imageDirectories += imagesInClass

    amount = min(amount, len(imageDirectories))
    selectedImageDirectories = random.sample(imageDirectories, amount)

    fig = plt.figure(figsize=(20, 20))
    rows = columns = math.ceil(math.sqrt(amount))

    for index, imageDirectory in enumerate(selectedImageDirectories):
        if index + 1 > amount:
            break
        
        try:
            testImage = image.load_img(imageDirectory, target_size=(224,224))      
        except:
            print("Corrupted image:", imageDirectory)
            time.sleep(1)
            return
          
        predictImage = image.img_to_array(testImage)
        predictImage = np.expand_dims(predictImage, axis=0)

        title = predict(model, predictImage)

        fig.add_subplot(rows, columns, index + 1)
        plt.imshow(testImage)
        plt.axis("off")
        plt.title(title, fontsize=50 / rows)

    plt.savefig(f"predictionsV{version}.png")

def predict(model, testImage):
    predictions = model.predict(testImage)[0].tolist()
    predicted = predictions.index(max(predictions))

    return classes[predicted][:-1].capitalize() + " - Confidence: %" + str(predictions[predicted] * 100)

def evaluateModel(model):
    testDatagen = ImageDataGenerator(
        rescale=1. / 255
    )

    validationGenerator = testDatagen.flow_from_directory(
        validationDirectory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    score = model.evaluate(validationGenerator, verbose = 0) 
    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])

def plotHistory(history):
    plt.figure()
    historyDF = pd.DataFrame(history.history)
    historyDF.loc[:, ['loss', 'val_loss']].plot()
    plt.savefig(f"lossV{version}.png")

    plt.figure()
    historyDF.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.savefig(f"accuracyV{version}.png")

version = 7
classes = os.listdir(validationDirectory)

#model = createModel()
#newModel, history = trainModel(model)
#newModel.save(f"catsDogsV{version}.keras")
#plotHistory(history)

model = load_model(f"catsDogsV{version}.keras")
multiPredict(model, 100)
