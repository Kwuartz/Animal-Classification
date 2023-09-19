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
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras import backend as K

trainDirectory = "training_set_small"
validationDirectory = "test_set_small"
trainSamples = 800
validationSamples = 100
epochs = 11
batch_size = 16
img_width, img_height = 224, 224

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def createModel():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

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
    print(selectedImageDirectories)

    fig = plt.figure(figsize=(10, 7))
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
        plt.title(title)

    plt.savefig(f"predictionsV{version}.png")

def predict(model, testImage):
    predictions = model.predict(testImage)[0]
    
    if predictions[0] > predictions[1]:  
        return "Cat - Confidence: %" + str(predictions[0] * 100)
    else:
        return "Dog - Confidence: %" + str(predictions[1] * 100)

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

version = 5

#model = createModel()
#newModel, history = trainModel(model)
#newModel.save("catsDogsV5.keras")
#plotHistory(history)

model = load_model(f"catsDogsV{version}.keras")
multiPredict(model, 9)
