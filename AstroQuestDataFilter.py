###############################################################
# AstroQuest Data Filtering Project - "Take Out the Trash"
# Primary Developer: Daniel Boyle
# Supervisor: Kevin Vinsen
# Created: 9/12/2019
# Last Edit: 18/12/2019
# Purpose: This project takes a machine learning approach to
#   data filtering using tensorflow and Keras to implement
#   multiple CNNs to filter data from AstroQuest
#   https://astroquest.net.au/science/the-science/
###############################################################

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import layers, datasets, models
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import yaml 
import os
from os.path import join
import h5py
import time

from tensorflow.keras.models import model_from_json
#Other files
from dataPreprocessing import *
from TopCornerNN import topCornerNNTraining
from ScribbleNN import scribbleNNTraining
from time import perf_counter

#Machine Learing
#topCornerNNTraining(tf)
#scribbleNNTraining(tf)


#Testing and validation