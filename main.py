import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os
  
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

trainDirectory = "training_set_big"
validationDirectory = "test_set_big"
trainSamples = 800
validationSamples = 100
epochs = 15
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
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
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
        class_mode='binary'
    )
 
    validationGenerator = testDatagen.flow_from_directory(
        validationDirectory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )
 
    history = model.fit_generator(
        trainGenerator,
        steps_per_epoch=trainSamples // batch_size,
        epochs=epochs,
        validation_data=validationGenerator,
        validation_steps=validationSamples // batch_size
    )
    
    return model, history

def multiPredict(model, directory, amount=999999):
    for index, file in enumerate(os.listdir(directory)):
        if index > amount:
            break
        
        try:
            testImage = image.load_img(directory + "/" + file, target_size=(224,224))      
        except:
            print("Corrupted image:", directory + "/" + file)
            time.sleep(1)
            return
          
        testImage = image.img_to_array(testImage)
        testImage = np.expand_dims(testImage,axis=0)

        predict(model, testImage)

def predict(model, testImage):
    result = model.predict(testImage)

    if result > 0.5:
        print("Dog")

        result = (result - 0.5) * 200
        print(str(result) + "% confidence")
    else:
        print("Cat")
        result = result * 100
        print(str(result) + "% confidence")

    print()

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
    historyDF = pd.DataFrame(history.history)
    historyDF.loc[:, ['loss', 'val_loss']].plot()
    plt.savefig("lossV4.png")
    historyDF.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.savefig("accuracyV4.png")

#model = createModel()
#newModel, history = trainModel(model)
#newModel.save("catsDogsV4.keras")
#plotHistory(history)

model = load_model("catsDogsV3.keras")