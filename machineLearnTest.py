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

from PIL import Image
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

def testML(): 
    possibleTags = ["happy","right","wow"]
    model = tensorflow.keras.models.load_model("64x3-CNN.model")
    audio = extract_features("demo.wav")
    print("Audio\n",audio)
    x = np.array(audio.tolist())
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
    
    if ((L1List[indexOfPrediction]*100)<70):
        print("Não entendi")
    else:
        if (tag == "happy"):
            i1 = Image.open("./images/happy.png")
        elif (tag == "wow"):
            i1 = Image.open("./images/wow.jpg")
        else:
            i1 = Image.open("./images/rigth.png")
        plt.imshow(i1)
        plt.show(block=False)
        plt.axis('off')
        plt.pause(5)
