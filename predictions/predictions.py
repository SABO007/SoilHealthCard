import pandas as pd

columns=['Temperature', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium',
       'pH', 'Electric Conductivity']

details = pd.read_csv('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\predictions\\input.csv',header=None)
details.columns=columns
print(details)

input = details.copy()

input['Nitrogen']= 1.35*10*details['Nitrogen']
input['Phosphorous'] = 1.35*10*details['Phosphorous']
input['Potassium'] = 1.35*10*details['Potassium']
input['Electric Conductivity']= 0.1*details['Electric Conductivity']




input=input[['Temperature', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium',
       'pH', 'Electric Conductivity']]

input['Salinity']=input['Electric Conductivity']

input=input[['Nitrogen', 'Phosphorous', 'Potassium',
       'pH', 'Electric Conductivity','Salinity','Temperature', 'Moisture']]



parameters=input.columns

# inputs=input.iloc[0:1,:]
print(input)
SHC=input.melt( 
        var_name="Parameter", 
        value_name="Test value")

# # Ideal Range
# # Tempertaure=> 20-30
# # Moisture=> (Paddy, Sugarcane=> 80-85), (Cotton, Maize=> 50-60)=> 50-75%
# # N=> 280-560
# # P=> 11-26
# # K=> 120-280
# # pH=> >7, =7, <7
# # Electrical Conductivity=> 0-2 dS/m

std=['280-560','11-26','120-280','7, Neutral','0-2','2-4','20-30','50-75']

unit=['Kg/Ha','Kg/Ha','Kg/Ha','H Potenz','dS/m','dS/m','°C','%']

SHC['Unit']=unit

SHC['Rating']=''

# # N
if (SHC['Test value'][0]<280):
    SHC['Rating'][0]='Low'
elif(SHC['Test value'][0]>=280 and SHC['Test value'][0]<=560):
    SHC['Rating'][0]='Medium'
elif(SHC['Test value'][0]>560):
    SHC['Rating'][0]='High'

# P
if (SHC['Test value'][1]<11):
    SHC['Rating'][1]='Low'
elif(SHC['Test value'][1]>=11 and SHC['Test value'][1]<=26):
    SHC['Rating'][1]='Medium'
elif(SHC['Test value'][1]>26):
    SHC['Rating'][1]='High'

# K
if (SHC['Test value'][2]<120):
    SHC['Rating'][2]='Low'
elif(SHC['Test value'][2]>=120 and SHC['Test value'][2]<=280):
    SHC['Rating'][2]='Medium'
elif(SHC['Test value'][2]>280):
    SHC['Rating'][2]='High'

# pH
if (SHC['Test value'][3]<6):
    SHC['Rating'][3]='Acidic'
elif(SHC['Test value'][3]>=6 and SHC['Test value'][3]<7):
    SHC['Rating'][3]='Slightly Acidic'
elif(SHC['Test value'][3]==7):
    SHC['Rating'][3]='Neutral'
elif(SHC['Test value'][3]>7 and SHC['Test value'][3]<=8):
    SHC['Rating'][3]='Slightly Basic'
elif(SHC['Test value'][3]>8):
    SHC['Rating'][3]='Basic'

# EC
if (SHC['Test value'][4]<0):
    SHC['Rating'][4]='Low'
elif(SHC['Test value'][4]>=0 and SHC['Test value'][4]<=2):
    SHC['Rating'][4]='Normal'
elif(SHC['Test value'][4]>2):
    SHC['Rating'][4]='High'

# Salinity
if (SHC['Test value'][5]<2):
    SHC['Rating'][5]='Low'
elif(SHC['Test value'][5]>=2 and SHC['Test value'][5]<=4):
    SHC['Rating'][5]='Normal'
elif(SHC['Test value'][5]>4):
    SHC['Rating'][5]='High'

# Temperature
if (SHC['Test value'][6]<20):
    SHC['Rating'][6]='Cool'
elif(SHC['Test value'][6]>=20 and SHC['Test value'][6]<=30):
    SHC['Rating'][6]='Ideal'
elif(SHC['Test value'][6]>30):
    SHC['Rating'][6]='Hot'

# Moisture
if (SHC['Test value'][7]<50):
    SHC['Rating'][7]='Low'
elif(SHC['Test value'][7]>=50 and SHC['Test value'][7]<=75):
    SHC['Rating'][7]='Medium'
elif(SHC['Test value'][7]>75):
    SHC['Rating'][7]='High'

SHC['Normal Level']=std

print("Soil Health Card")
print(SHC)

def color_rating(val):
    if val == 'Low':
        color = 'yellow'
    elif val == 'High':
        color = 'red'
    elif val == 'Neutral':
        color = 'grey'
    elif val == 'Ideal':
        color = 'green'
    elif val == 'Normal':
        color = 'orange'
    elif 'Acidic' in val:
        color = 'red'
    elif 'Basic' in val:
        color = 'yellow'
    else:
        color = 'black'
    return 'background-color: %s' % color

# apply the function to the 'Rating' column and export the styled DataFrame as an HTML file
SHC_c = SHC.style.map(color_rating, subset=['Rating'])


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
model_crop = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_crop_prediction.pkl', 'rb'))
crop_mms = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_crop_mms.pkl', 'rb'))

X_exp_crop_sc=crop_mms.transform(X_exp_crop)
predict_crop=model_crop.predict(input_crop)

#Fertilizer prediction

# Taking Random input and crop name for Fertilizer prediction
user_crop=['Sugarcane', 'Millets', 'Cotton', 'Paddy', 'Wheat', 'Oil seeds', 'Ground Nuts', 'Pulses', 'Barley', 'Tobacco', 'Maize']

ferti_le = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_ferti_le.pkl', 'rb'))
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

model_ferti = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_ferti_prediction.pkl', 'rb'))
ferti_mms = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_ferti_mms.pkl', 'rb'))

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
model_soil = load_model('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_soil_detection.h5')

# model_soil=pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_soil_detection.pkl','rb'))

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