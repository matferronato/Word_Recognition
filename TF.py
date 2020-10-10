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
from sklearn import metrics 
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 



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


# Convert features and corresponding classification labels into numpy arrays
x = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

print("ARRAY X")
print(x)
print("ARRAY Y")
print(y)

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 
print("ARRAY YY")
print(yy)

print("TREINAMENTO STUFF")
x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state = 42)

print("X TREINAMENTO")
print(x_train)
print("X TESTE")
print(x_test)
print("Y TREINAMENTO")
print(y_train)
print("Y TESTE")
print(y_test)
###########################################################################3

num_rows = 40
num_columns = 174
num_channels = 1

#x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
#x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)


num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))
##########################################################################

print("yy")
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# Display model architecture summary 
model.summary()
print("y")
# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)

########################################################################33

num_epochs = 72
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)



# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])