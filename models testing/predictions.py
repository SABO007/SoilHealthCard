import warnings
import pandas as pd
# suppress the UserWarnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but MinMaxScaler was fitted with feature names")
warnings.filterwarnings("ignore", message="X has feature names, but RandomForestClassifier was fitted without feature names")


#Crop prediction
import chardet
input_crop=pd.read_csv('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\predictions\\input.csv', header=None) #I am here

input_crop.columns=['Temperature', 'Moisture', 'Nitrogen', 'Phosphorous','Potassium', 'pH', 'ElectricalConductivity']
input_crop=input_crop[['Temperature', 'Moisture', 'Nitrogen', 'Phosphorous','Potassium', 'pH']]

X_exp_crop=input_crop.loc[0]
# X_exp_crop=X_exp_crop.values
X_exp_crop=[X_exp_crop]

import pickle
model_crop = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\saved models\\model_crop_prediction.pkl', 'rb'))
crop_mms = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\saved models\\model_crop_mms.pkl', 'rb'))

X_exp_crop_sc=crop_mms.transform(X_exp_crop)
predict_crop=model_crop.predict(input_crop)

#Fertilizer prediction

# Taking Random input and crop name for Fertilizer prediction
user_crop=['Sugarcane', 'Millets', 'Cotton', 'Paddy', 'Wheat', 'Oil seeds', 'Ground Nuts', 'Pulses', 'Barley', 'Tobacco', 'Maize']

ferti_le = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\saved models\\model_ferti_le.pkl', 'rb'))
user_crop=ferti_le.transform(user_crop)

import random as rd
import csv

with open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\predictions\\input.csv', newline='') as input:
    input_ferti = csv.reader(input)
    a = []
    for row in input_ferti:
        a.extend(row)
    del a[5:7]

    a.append(user_crop[rd.randint(0,len(user_crop)-1)])

    a=[a]

x_exp_ferti=a

model_ferti = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\saved models\\model_ferti_prediction.pkl', 'rb'))
ferti_mms = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\saved models\\model_ferti_mms.pkl', 'rb'))

# Scaling 
x_exp_ferti_sc=ferti_mms.transform(x_exp_ferti)

#Fertilizer Prediction for random input
predict_ferti=model_ferti.predict(x_exp_ferti_sc)


#Soil type prediction
import tensorflow as tf
import logging

# Set TensorFlow logging to only display errors and critical messages
tf.get_logger().setLevel(logging.ERROR)


import keras.utils as image
import sys
import numpy as np

img_path = "C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\SHC_external input\\SoilSample2.jpg" #Uploaded file path
img = image.load_img(img_path, target_size=(256, 256))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

from tensorflow.keras.models import load_model

# Load the model
model_soil = load_model('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\saved models\\model_soil_detection.h5')

# model_soil=pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\saved models\\model_soil_detection.pkl','rb'))

arr = model_soil.predict(img_tensor, batch_size=377, verbose=1)
res = np.argmax(arr, axis = -1)
soil_type=[]
if(res == 0):
    soil_type.append('Clay')
elif(res == 1): 
    soil_type.append('Loam')
elif(res == 2): 
    soil_type.append('Loamy_Sand')
elif(res == 3): 
    soil_type.append('Sand')
elif(res == 4): 
    soil_type.append('Sandy_Loam')

soil_type=soil_type[0]
predict_crop=predict_crop[0]
predict_ferti=predict_ferti[0]

predict_cf=pd.DataFrame(columns=['Type','Prediction'])

predict_cf['Type']=['Soil Type','Crop predicted','Fertilizer predicted']
predict_cf['Prediction']=[soil_type,predict_crop,predict_ferti]

print(predict_cf)