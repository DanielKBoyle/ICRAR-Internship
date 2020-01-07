###############################################################
# Program: TopCornerNN
# Primary Developer: Daniel Boyle
# Code segments used from tensorflow.org under free use
# Created: 18/12/2019
# Last Edit: 18/12/2019
# Purpose: model designed for identifying the top corner bugs
#           (topCornerB)
###############################################################
import os
import yaml 
import numpy as np
from dataPreprocessing import extractAll

def topCornerNNTraining(tf):
    #Trains and saves model for identifying topCorner bugs
    firstRun = True

    #Load data
    filename = 'TopCornerNN_data.npz'
    classNames = ['topCornerB','good']

    if firstRun:
        topCornerImages, goodImages, dataLength = extractTrainingSets()
        topCornerLabels = np.ones((dataLength,), dtype=uint8_t)
        goodLabels = np.zeros((dataLength,), dtype=uint8_t)

        #This number defines how much data goings into training and how much into testing
        splitFraction = 0.8
        split = round(dataLength*splitFraction)

        #Construct datasets
        rawTrainingData = np.concatenate((topCornerImages[:split], goodImages[:split]), axis=0)
        rawTrainingLabels = np.concatenate((topCornerLabels[:split], goodLabels[:split]), axis=0)
        
        rawTestData = np.concatenate((topCornerImages[split:], goodImages[split:]), axis=0)
        rawTestLabels = np.concatenate((topCornerLabels[split:], goodLabels[split:]), axis=0)
        
        #Normalise data
        rawTrainingData, rawTestData = rawTrainingData / 255.0, rawTestData / 255.0
        
        #Construct an array of random indices then use them to randomise the data
        indexArrTraining = np.arange(0,len(rawTrainingData),dtype=int)
        indexArrTraining = np.random.permutation(indexArrTraining)
        trainingData = rawTrainingData[indexArrTraining]
        trainingLabels = rawTrainingLabels[indexArrTraining]

        #And do the same for test data
        indexArrTest = np.arange(0,len(rawTestData),dtype=int)
        indexArrTest = np.random.permutation(indexArrTest)
        testData = rawTestData[indexArrTest]
        testLabels = rawTestLabels[indexArrTest]

        np.savez(filename, trainingData=trainingData, trainingLabels=trainingLabels, testData=testData, testLabels=testLabels )
        #Can replace savez with savex_compressed for larger datasets
    else:
        #Load data from filename = 'TopCornerNN_data.npz'

        with np.load(filename) as data:
            trainingData = data['trainingData']
            trainingLabels = data['trainingLabels']
            testData = data['testData']
            testLabels = data['testLabels']
    #End if
    
    trainDataset = tf.data.Dataset.from_tensor_slices((trainingData, trainingLabels))
    testDataset = tf.data.Dataset.from_tensor_slices((testData, testLabels))

    #Batch data
    batchSize = 32
    trainDataset = trainDataset.batch(batchSize)
    testDataset = testDataset.batch(batchSize)
 
    #Construct model based off mobileNetV2
    if firstRun:
        img_shape = (600,600,3)
        baseModel = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
        baseModel.trainable = False

        globalAvLayer = tf.keras.layers.GlobalAveragePooling2D()
        predictionLayer = tf.keras.layers.Dense(1)

        model = tf.keras.Sequential([
            baseModel,
            globalAvLayer,
            predictionLayer])
    else:
        model = keras.models.load_model('TopCornerNN_model.h5')

    #Train the model
    baseLearningRate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=baseLearningRate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model.summary()

    intial_epochs = 10

    history = model.fit(trainDataset,
                    epochs=initial_epochs,
                    validation_data=testDataset) #Yes, I know, using test data as validation
    #Check results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print('Accuracy : ' + acc)
    print('Validation Accuracy : ' + val_acc)

    #Extra plotting
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
    #Save the model
    model.save('TopCornerNN_model.h5')


def extractTrainingSets():
    #Extracts and processes training sets using good.yaml and topCornerB.yaml
    topCornerImages = []
    goodImages = []
    extractAll('yamlDB' + os.sep + 'topCornerB.yaml', topCornerImages)
    extractAll('yamlDB' + os.sep + 'good.yaml', goodImages)

    #Cutoff to keep datasets balanced
    #They should be about the same beforehand anyway
    dataLength = min(len(topCornerImages), len(goodImages))
    topCornerImages = np.array(topCornerImages[:dataLength])
    goodImages = np.array(goodImages[:datalength])
    

    return topCornerImages, goodImages, dataLength