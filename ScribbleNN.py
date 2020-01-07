###############################################################
# Program: ScribbleNN
# Primary Developer: Daniel Boyle
# Created: 16/12/2019
# Last Edit: 16/12/2019
# Purpose: Model for identifying scribbles (scribbleM, drawingM)
#
#This uses data from the Quick, Draw! dataset. It currently uses
#only 4 different types from there however could be expanded to use
#all of them.
#The data is all open source at quickdraw.withgoogle.com/data
###############################################################

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataPreprocessing import extractAll
import cv2

def scribbleNNTraining(tf):
    #Trains and saves scribbleNN model for identifying scribbles
    firstRun = True

    #Load data
    goodImages = []
    #extractAll('yamlDB' + os.sep + 'good.yaml', goodImages)

    hurricaneData = np.load('quickDrawDatasets' + os.sep + 'hurricane.npy')
    lineData =  np.load('quickDrawDatasets' + os.sep + 'line.npy')
    squiggleData =  np.load('quickDrawDatasets' + os.sep + 'squiggle.npy')
    zigzagData =  np.load('quickDrawDatasets' + os.sep + 'zigzag.npy')

    classNames = ['hurricane','line','squiggle','zigzag']

    imDimension = 600 #This number is not tested below 28
    resizeFrac = 0.5 #RANDOMIZE

    arrTemp = np.resize(zigzagData[1,:],(28,28))
    index = int(round(imDimension*resizeFrac))
    im_cv = cv2.resize(arrTemp,(index,index))
    arrTemp = np.array(im_cv)
    arr = np.zeros(imDimension,imDimension)

    offsetX = imDimension // 2 #RANDOMIZE
    offsetY = imDimension // 2 #RANDOMIZE

    shape = arrTemp.shape
    arr[offsetX:shape[0]+offsetX,offsetY:shape[1]+offsetY] = arrTemp

    arrFinal = np.dstack((arr,arr,arr))

    #Could also add a good image overlayed

    #Create the model
    if firstRun:
        img_shape = (600,600,3)
        baseModel = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
        baseModel.trainable = False

        globalAvLayer = tf.keras.layers.GlobalAveragePooling2D()
        predictionLayer = tf.keras.layers.Dense(4)

        model = tf.keras.Sequential([
            baseModel,
            globalAvLayer,
            predictionLayer])
    else:
        model = keras.models.load_model('ScribbleNN_model.h5')

    #Train the model
    


    #Save the model
    model.save('ScribbleNN_model.h5')
