from flask import Flask, render_template, request, jsonify
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import keras.utils as image
import sys
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Farmer.html')  # Renders the HTML file

@app.route('/get_farmer', methods=['POST'])
def get_farmer():
    farmer_name = request.form.get('farmerName')
    farmer_mailid = request.form.get('farmerEmail')
    farmer_address = request.form.get('farmerAddress')

    farmer_details = [farmer_name, farmer_mailid, farmer_address]
    print(farmer_details)

    # Perform any additional processing if needed

    return jsonify({"success": True})

@app.route('/get_farm', methods=['POST'])
def get_farm():
    sampleCollectionDate = request.form.get('sampleCollectionDate')
    surveyNumber = request.form.get('surveyNumber')
    farmSize = request.form.get('farmSize')
    geoPosition = request.form.get('geoPosition')

    farm_details = [sampleCollectionDate, surveyNumber, farmSize, geoPosition]
    print(farm_details)

    return jsonify({"success": True})

@app.route('/get_crop', methods=['POST'])
def get_crop():
    soilImage = request.form.get('cropImage')

    nitrogen = request.form.get('nitrogen')
    phosphorous = request.form.get('phosphorous')
    potassium = request.form.get('potassium')
    ph = request.form.get('ph')
    electricConductivity = request.form.get('electricConductivity')
    salinity = request.form.get('salinity')
    temperature = request.form.get('temperature')
    moisture = request.form.get('moisture')

    #Crop prediction
    crop=[[temperature, moisture, nitrogen, phosphorous, potassium, ph]]
    
    model_crop = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_crop_prediction.pkl', 'rb'))
    crop_mms = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_crop_mms.pkl', 'rb'))

    crop_sc=crop_mms.transform(crop)
    predict_crop=model_crop.predict(crop_sc)
    print(predict_crop)

    #Fertiliser prediction
    user_crop=['Sugarcane', 'Millets', 'Cotton', 'Paddy', 'Wheat', 'Oil seeds', 'Ground Nuts', 'Pulses', 'Barley', 'Tobacco', 'Maize']
    ferti_le = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_ferti_le.pkl', 'rb'))
    user_crop=ferti_le.transform(user_crop)

    ferti=[[temperature, moisture, user_crop[0], nitrogen, phosphorous, potassium]]

    model_ferti = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_ferti_prediction.pkl', 'rb'))
    ferti_mms = pickle.load(open('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\PickledModels\\model_ferti_mms.pkl', 'rb'))

    ferti_sc=ferti_mms.transform(ferti)
    predict_ferti=model_ferti.predict(ferti_sc)
    print(predict_ferti)

    #SoilType prediction
    if 'soilImage' in request.files:
        soil_image = request.files['soilImage']
        if soil_image.filename != '':
            # Save the image to the upload folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], soil_image.filename)
            soil_image.save(image_path)

    img_path = "C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\SHC_external input\\SoilSample2.jpg" #Uploaded file path
    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.


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
    print(soil_type)


    return jsonify({"message": "Crop details submitted successfully."})

if __name__ == '__main__':
    app.run(debug=True)
