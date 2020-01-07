###############################################################
# Program: dataPreprocessing
# Primary Developer: Daniel Boyle
# Created: 16/12/2019
# Last Edit: 16/12/2019
# Purpose: Functions for data preprocessing
###############################################################
import numpy as np
import time
import yaml
import os
import PIL
from PIL import Image

#Unstable limitRegion Function
def limitRegion2(im, filepath):
    try:
        im = Image.open(filepath,'r')
        try:
            im = im.convert('RGBA')
            imArray = np.array(im)
            mainArray = np.full((600,600,4), [0, 0, 0, 255])
            imArray[(imArray != [0, 0, 0, 255]).all(-1)] = np.array([255, 255, 255, 0])

            im = Image.fromarray(imArray)
            im = im.convert('RGB')
            im = np.array(im) #Remove this line to receive image instead of numpy array

        except IOError:
            print('InvalidImage: ' + filepath)
    except IOError:
        print('InvalidFile: ' + filepath)
    return im


def limitRegion(im, filepath):
    #Removes any part of the image that is not the primary mask.
    #For AstroQuest masks' this limits the image to just the 
    #part of the drawing covering the galaxy.
    #If the image isn't compatible it returns the same image
    #(Error messages will be printed as well)
    try:
        im = Image.open(filepath,'r')
        try:
            im = im.convert('RGBA')
            imArray = np.array(im)
            
            print('Hello world')
            #start1 = time.process_time_ns()
            mainArray = np.full((600,600,4), [0, 0, 0, 255])
            logicArray = mainArray != imArray
            for ii in range(600):
                for jj in range(600):
                    if np.any(logicArray[ii][jj][:]):
                        imArray[ii][jj] = [255, 255, 255, 0]
            #print(time.process_time_ns()-start1)
            ''' #Iterating like this is ~50% slower
            start2 = time.process_time_ns()
            for ii in range(600):
                for jj in range(600):
                    if not np.array_equal(imArray[ii][jj],[0, 0, 0, 255]):
                        imArray[ii][jj] = [255, 255, 255, 0]
            print(time.process_time_ns() - start2)
            '''
            im = Image.fromarray(imArray)
            im = im.convert('RGB')
            im = np.array(im) #Remove this line to receive image instead of numpy array

        except IOError:
            print('InvalidImage: ' + filepath)
    except IOError:
        print('InvalidFile: ' + filepath)
    return im

#Older versison below using P mode instead of RGBA
'''
def limitRegion(image):
    #Image that converts P mode images to remove all colours that
    #aren't the primary colour. For AstroQuest masks this limits the 
    #image to just the drawing covering the galaxy
    imagCompat = True
    if not (image.format == 'P'):
        try:
            image = image.convert('P')
        except():
            print('Invalid image, could not be converted')
            imagCompat = False            
    if imagCompat:
        imArray = np.array(image)
        for ii in np.nditer(imArray, op_flags = ['readwrite']):
            if not ii == 1: #The mask covering the galaxy should be 1
                ii[...] = 0
        newImage = PIL.Image.fromarray(imArray)
    else:
        newImage = image
    return newImage
'''

#This function is unfinished and needs work
def expandRegion(im, filepath):
    #Function that takes all secondary parts and rounds them up to be
    #the same as the primary mask. In practice this rounds all non-zero
    #pixels up to black.

    return im

def createDataset(type, N, galaxies, directory):
    #Selects N masks of a certain type (if possible)
    #   type can be any of the error types or 'good'
    #   N is the number required in the training set
    #   galaxies is a list of yaml files to extract training sets from

    currSize = 0
    validRequest = True
    index = 0
    while currSize < N and validRequest:
        currGalaxy = galaxies[index]
        filename = str(currGalaxy) + '.yaml'
        inStream = open(join(directory,filename))
        results = yaml.load_all(inStream, Loader=yaml.FullLoader)
        print(results)

        index += 1
   # return dataset

def extractAll(filepath, images):
    #Extracts all images of a yaml file
    inStream = open(os.path.join(filepath))

    galaxies = yaml.load(inStream, Loader=yaml.FullLoader)
    for result, info in galaxies.items():
        filepath = os.path.join('galaxyData', str(info[1]['galaxy']), result)
        imBase = Image.open(filepath, 'r')
        imFinal = limitRegion(imBase)
        images.append(imFinal)

    inStream.close()