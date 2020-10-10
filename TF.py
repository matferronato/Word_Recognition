import pandas as pd
import os
import librosa
import librosa.display
import struct
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import layers
from sklearn import metrics 
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

import speech_recognition as sr

def microphone_check():
    #Habilita o microfone para ouvir o usuario
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
        microfone.adjust_for_ambient_noise(source)
        runListening()
        print("pode falar")
        audio = microfone.listen(source)
        return audio


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



def super_extract_features(audio):
    try:        
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #print(audio)
        #print(sample_rate)
        #fig = plt.figure(figsize=(15,15))
        #fig.subplots_adjust(hspace=0.4, wspace=0.4)
        #librosa.display.waveplot(audio, sr= sample_rate)
        #plt.savefig('class_examples.png')                
        #time.sleep(3)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0) #erro nesta linha
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
    return mfccsscaled

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #print(audio)
        #print(sample_rate)
        #fig = plt.figure(figsize=(15,15))
        #fig.subplots_adjust(hspace=0.4, wspace=0.4)
        #librosa.display.waveplot(audio, sr= sample_rate)
        #plt.savefig('class_examples.png')                
        #time.sleep(3)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0) #erro nesta linha
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
    return mfccsscaled
   

# Set the path to the full UrbanSound dataset 
fulldatasetpath = './Data/'

metadata = pd.read_csv('information.csv')
features = []

## Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    file_name = row['file_name']
    class_label = row['tag']
    data = extract_features(fulldatasetpath+file_name)
    features.append([data, class_label])
    print("a")

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
print("TABLE")
print(featuresdf.head())
print("A")
print(featuresdf.iloc[0]['feature'])

# Convert features and corresponding classification labels into numpy arrays
x = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

print("TREINAMENTO STUFF")
x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state = 42)

###########################################################################3

print("###########################################################################")
print("EH ESSE AQUI PORRA = ", x_train.shape)
print(x_test.shape)
print(yy.shape)
print(y_train.shape)
print(y_test.shape)
print("###########################################################################")

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
model.build(input_shape=[x_test.shape[0],x_test.shape[1]])

# Display model architecture summary

########################################################################33

# Display model architecture summary 
print(model)

model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]




print("Pre-training accuracy: %.4f%%" % accuracy)




from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(score[1]))

print("teste1")
t1 = microphone_check()
print("teste2")
t2 = microphone_check()
print("teste3")
t3 = microphone_check()
print("teste4")
t4 = microphone_check()

L1 = model.predict(super_extract_features(t1))
L2 = model.predict(super_extract_features(t2))
L3 = model.predict(super_extract_features(t3))
L4 = model.predict(super_extract_features(t4))

print(L1)
print(L2)
print(L3)
print(L4)