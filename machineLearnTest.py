import os
import time
import struct
import librosa
import tensorflow
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr

from gtts import gTTS
from datetime import datetime 
from pydub import AudioSegment
from playsound import playsound

from keras import layers
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0) #erro nesta linha
    return mfccsscaled

def normalize(x_train):
    x_train_means = x_train.mean(axis=0)
    x_train_stds = x_train.std(axis=0)
    x_train = x_train - x_train_means
    x_train = x_train/x_train_stds


def testML(): 
    possibleTags = ["bed", "bird", "cat", "dog", "down", "go", "happy", "house", "left", "marvel", "no", "off", "on", "right", "stop", "up", "wow", "yes"]
    model = tensorflow.keras.models.load_model("64x3-CNN.model")
    audio = extract_features("demo.wav")
    print("Audio\n",audio)
    x = np.array(audio.tolist())
    normalize(x)
    print(x)
    print("original shape\n",x.shape)
    x = x.reshape(1,40)
    print("new shape\n",x.shape)
    L1 = model.predict(x)
    
    L1List = []
    for eachList in L1:
        for eachItem in eachList:
            print("a = ",eachItem)
            L1List.append(eachItem)
    print(L1List)
    prediction = max(L1List)
    print(prediction)
    indexOfPrediction = L1List.index(prediction)
    print(indexOfPrediction)
    tag = possibleTags[indexOfPrediction]
    
    print("predicted ", tag, " with ", str(L1List[indexOfPrediction]*100) ," % certain" )
    