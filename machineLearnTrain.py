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

class WavFileHelper():
    
    def read_file_properties(self, filename):

        wave_file = open(filename,"rb")
        
        riff = wave_file.read(12)
        fmt = wave_file.read(36)
        
        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0) #erro nesta linha
    return mfccsscaled
 
def returnAmplitueAndTagInfoList(fulldatasetpath, metadata):
    features = []
    for index, row in metadata.iterrows():
        file_name = row['file_name']
        class_label = row['tag']
        data = extract_features(fulldatasetpath+file_name)
        features.append([data, class_label])
    return features   
 
def retrieveMetaData(file):
    metadata = pd.read_csv(file)    
    return metadata

def transformDataFrameIntoNumpyArray(featuresdf):
    # Convert features and corresponding classification labels into numpy arrays
    print("A")
    featuresdf.iloc[0]['feature']
    print("b")
    x = np.array(featuresdf.feature.tolist())
    print("c")
    y = np.array(featuresdf.class_label.tolist())
    print("d") 
    le = LabelEncoder()
    print("e")
    yy = to_categorical(le.fit_transform(y)) 
    print("f")
    return x, y, yy

def createMachineLearnModel(x, yy):
    num_labels = yy.shape[1]
    model = Sequential()
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.build(input_shape=[x.shape[0],x.shape[1]])    
    model.summary()
    return model
    
def lookForPreTrainAccuracy(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = 100*score[1]
    print("Pre-training accuracy: %.4f%%" % accuracy)    

def trainingModel(model,x_train, y_train, x_test, y_test):
    num_epochs = 100
    num_batch_size = 32
    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

def lookForPostTrainAccuracy(model, x_train, y_train, x_test, y_test):
    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: {0:.2%}".format(score[1]))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: {0:.2%}".format(score[1]))

def trainML():
    fulldatasetpath = './Data/'
    metadata = retrieveMetaData('MetaData/information.csv')
    features = returnAmplitueAndTagInfoList(fulldatasetpath, metadata)
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
    x, y, yy = transformDataFrameIntoNumpyArray(featuresdf)
    x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state = 42)
    model = createMachineLearnModel(x_train, yy)
    lookForPreTrainAccuracy(model, x_test, y_test)
    trainingModel(model,x_train, y_train, x_test, y_test)
    lookForPostTrainAccuracy(model, x_train, y_train, x_test, y_test)
    model.save('64x3-CNN.model')

