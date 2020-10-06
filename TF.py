import pandas as pd
import os
import librosa
import librosa.display
import struct
import time
import numpy as np

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

print('Finished feature extraction from ', len(featuresdf), ' files')