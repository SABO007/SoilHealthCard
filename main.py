# Flask
from flask import Flask, render_template, request, jsonify, send_file
import pickle
from SHC.SHC import shc_generation

# Model prediction
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import keras.utils as image
import sys
import numpy as np
import pandas as pd
import os

# SHC Generation
from io import StringIO, BytesIO
from xhtml2pdf import pisa
from string import Template as HTMLTemplate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Farmer.html')  # Renders the HTML file

@app.route('/get_farmer', methods=['POST'])
def get_farmer():
    farmer_name = request.form.get('farmerName')
    farmer_mailid = request.form.get('farmerEmail')
    farmer_address = request.form.get('farmerAddress')

    global farmer_details
    farmer_details = [farmer_name, farmer_mailid, farmer_address]
    print(farmer_details)

    return "Farmer details submitted successfully!"


@app.route('/get_farm', methods=['POST'])
def get_farm():
    sampleCollectionDate = request.form.get('sampleCollectionDate')
    surveyNumber = request.form.get('surveyNumber')
    farmSize = request.form.get('farmSize')
    geoPosition = request.form.get('geoPosition')

    global farm_details
    farm_details = [sampleCollectionDate, surveyNumber, farmSize, geoPosition]
    print(farm_details)
    
    return "Farm details submitted successfully!"

@app.route('/get_crop', methods=['GET','POST'])
def get_crop():
    app.config['UPLOAD_FOLDER']='uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

    soilImage = request.files.get('soilImage')
    print('Soil image: ', soilImage)

    nitrogen = float(request.form.get('nitrogen'))
    phosphorous = float(request.form.get('phosphorous'))
    potassium = float(request.form.get('potassium'))
    ph = float(request.form.get('ph'))
    electricConductivity = float(request.form.get('electricConductivity'))
    temperature = float(request.form.get('temperature'))
    moisture = float(request.form.get('moisture'))

    global farmer_details, farm_details, soil_details
    soil_details = [soilImage, nitrogen, phosphorous, potassium, ph, electricConductivity, temperature, moisture]
    print(soil_details)

    # user_info updated
    farmer_details_all=pd.read_csv('user_info/farmer_details.csv')
    farm_details_all=pd.read_csv('user_info/farm_details.csv')
    soil_details_all=pd.read_csv('user_info/soil_details.csv')

    if farmer_details_all.empty & farm_details_all.empty & soil_details_all.empty:
        farmer_details_all.loc[0] = farmer_details
        farm_details_all.loc[0] = farm_details
        soil_details_all.loc[0] = soil_details[1:]
    else:
        farmer_details_all.loc[(len(farmer_details_all))] = farmer_details
        farm_details_all.loc[(len(farm_details_all))] = farm_details
        soil_details_all.loc[(len(soil_details_all))] = soil_details[1:]

    farmer_details_all.to_csv('user_info/farmer_details.csv', index=False)
    farm_details_all.to_csv('user_info/farm_details.csv', index=False)
    soil_details_all.to_csv('user_info/soil_details.csv', index=False)



    # Save image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'Uploaded Image {len(farmer_details_all)}')
    soilImage.save(image_path)

    #Crop prediction
    crop=[[temperature, moisture, nitrogen, phosphorous, potassium, ph]]

    #SHC
    global crop_SHC
    crop_SHC=[temperature, moisture, nitrogen, phosphorous, potassium, ph, electricConductivity]

    
    model_crop = pickle.load(open('saved models/model_crop_prediction.pkl', 'rb'))
    crop_mms = pickle.load(open('saved models/model_crop_mms.pkl', 'rb'))

    crop_sc=crop_mms.transform(crop)
    predict_crop=model_crop.predict(crop_sc)
    predict_crop=predict_crop[0]
    print(predict_crop)

    #Fertiliser prediction
    user_crop=['Sugarcane', 'Millets', 'Cotton', 'Paddy', 'Wheat', 'Oil seeds', 'Ground Nuts', 'Pulses', 'Barley', 'Tobacco', 'Maize']
    ferti_le = pickle.load(open('saved models/model_ferti_le.pkl', 'rb'))
    user_crop=ferti_le.transform(user_crop)

    ferti=[[temperature, moisture, user_crop[0], nitrogen, phosphorous, potassium]]

    model_ferti = pickle.load(open('saved models/model_ferti_prediction.pkl', 'rb'))
    ferti_mms = pickle.load(open('saved models/model_ferti_mms.pkl', 'rb'))

    ferti_sc=ferti_mms.transform(ferti)
    predict_ferti=model_ferti.predict(ferti_sc)
    predict_ferti=predict_ferti[0]
    print(predict_ferti)

    img_path = f'uploads/Uploaded Image {len(farmer_details_all)}'
    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.


    # Load the model
    model_soil = load_model('saved models/model_soil_detection.h5')

    arr = model_soil.predict(img_tensor, batch_size=377, verbose=1)
    res = np.argmax(arr, axis = -1)
    soil_type=[]
    if(res == 0):
        soil_type.append('Clay')
    elif(res == 1): 
        soil_type.append('Loam')
    elif(res == 2): 
        soil_type.append('Loamy Sand')
    elif(res == 3): 
        soil_type.append('Sand')
    elif(res == 4): 
        soil_type.append('Sandy Loam')

    soil_type=soil_type[0]
    print(soil_type)

    # Crop fertilizer Dataframe
    df_cf=pd.DataFrame(columns=['Type','Prediction'])
    df_cf['Type']=['Soil Type','Crop predicted','Fertilizer predicted']
    df_cf['Prediction']=[soil_type,predict_crop,predict_ferti]

    farmer_data=pd.DataFrame(columns=['Name', 'Email Id', 'Address'])
    
    if farmer_data.empty:
      farmer_data.loc[0] = farmer_details
    else:
        farmer_data = farmer_data.append(pd.Series(farmer_details, index=farmer_data.columns), ignore_index=True)


    farm_data=pd.DataFrame(columns=['Date of Sample Collection', 'Survey No., Khasra No,/ Dag No,',
       'Farm Size','Geo Position (GPS)'])
    
    if farm_data.empty:
      farm_data.loc[0] = farm_details
    else:
        farm_data = farm_data.append(pd.Series(farm_details, index=farmer_data.columns), ignore_index=True)
        
    farmer_t=farmer_data.T
    farm_t=farm_data.T

    Lst=[]
    f='User'
    for i in range(len(farmer_data)):
        I=str(i+1)
        F=f+I
        Lst.append(F)

    farmer_t.columns=Lst
    farm_t.columns=Lst


    Farmer_Info=[]
    Farm_Info=[]

    for i in range(len(farmer_data)):
        x=farmer_t[[Lst[i]]]
        x = x.style.set_caption('Farmer Details')
        Farmer_Info.append(x)

        y=farm_t[[Lst[i]]]
        y=y.style.set_caption('Farm Details')
        Farm_Info.append(y)

    print("Farmer info: ", x)
    print("Farm info: ", y)
    farmer_html = Farmer_Info[len(farmer_data)-1].to_html()
    farm_html = Farm_Info[len(farmer_data)-1].to_html()
    cf_html = df_cf.to_html()
    
    columns=['Temperature', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium',
        'pH', 'Electric Conductivity']
    details=pd.DataFrame(columns=columns)
    details.loc[0]= crop_SHC

    shc_generation

    shc_html = shc_generation(details).to_html()

    
    # Create HTML template
    html_template = HTMLTemplate('''
    <html>
        <head>
        <style>
            table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
            padding: 5px;
            }
        </style>
        </head>
        <body>
        <h1>Farmer Information</h1>
        $html1
        $html2
        <h1>Soil Health Card</h1>
        $html3
        $html4
        </body>
    </html>
    ''')

    html = html_template.substitute(html1=farmer_html, html2=farm_html, html3=cf_html, html4=shc_html)

    
    # Convert HTML to PDF
    global pdf
    pdf = BytesIO()
    pisa.CreatePDF(BytesIO(html.encode('utf-8')), pdf)
    pdf.seek(0)

    return jsonify(f"Suggested Crop: <span style='color: red;'>{predict_crop}</span>\n   Suggested Fertilizer: <span style='color: red;'>{predict_ferti}</span>\n     Detected Soil Type: <span style='color: red;'>{soil_type}</span>")

@app.route('/download_pdf')
def download_pdf():
    return send_file(
        pdf,
        mimetype='application/pdf',
        download_name='Soil Health Card.pdf',
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)
