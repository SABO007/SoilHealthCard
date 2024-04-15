# Flask
from flask import Flask, render_template, request, jsonify, send_file, redirect, session, url_for
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
import seaborn as sns
import matplotlib.pyplot as plt
import os

# SHC Generation
from io import StringIO, BytesIO
from xhtml2pdf import pisa
from string import Template as HTMLTemplate

# SMTP
from SMTP.smtp import send_email

# env variable extraction
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

import pyrebase

app=Flask(__name__, template_folder='templates')

# Firebase
firebase_config = {
    "apiKey": os.environ.get('FIREBASE_API_KEY'),
    "authDomain": os.environ.get('FIREBASE_AUTH_DOMAIN'),
    "projectId": os.environ.get('FIREBASE_PROJECT_ID'),
    "storageBucket": os.environ.get('FIREBASE_STORAGE_BUCKET'),
    "messagingSenderId": os.environ.get('FIREBASE_MESSAGING_SENDER_ID'),
    "appId": os.environ.get('FIREBASE_APP_ID'),
    "measurementId": os.environ.get('FIREBASE_MEASUREMENT_ID'),
    "databaseURL": os.environ.get('FIREBASE_DATABASE_URL')
}

firebase_config = os.getenv('FIREBASE_CONFIG')
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

app.secret_key = 'sabo'

@app.route('/')
def login():
    return render_template('login.html')


@app.route('/login_auth', methods=['GET', 'POST'])
def login_auth():
    if 'user' in session:
        return jsonify("Already logged in")

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = email
            return  jsonify("Success")
        
        except Exception as e:
            print("Invalid credentials:", str(e))
            return jsonify("Invalid credentials")
        

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register_auth', methods=['GET', 'POST'])
def register_auth():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            auth.create_user_with_email_and_password(email, password)
            return jsonify("Success")
        except Exception as e:
            print("Error:", str(e))
            return jsonify("Email exists")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/moreinfo')
def moreinfo():
    return render_template('moreinfo.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

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

    
    soil_low =    [280, 11, 120, 1, 0, 2, 20, 50]
    soil_high = [560, 26, 280, 14, 2, 4, 30, 75]

    dict = {
        'Nitrogen': [soil_low[0], soil_details[1], soil_high[0]], 
        'Phosphorous': [soil_low[1], soil_details[2], soil_high[1]],
        'Potassium': [soil_low[2], soil_details[3], soil_high[2]],
        'pH': [soil_low[3], soil_details[4], soil_high[3]],
        'Electric Conductivity': [soil_low[4], soil_details[5], soil_high[4]],
        'Temperature': [soil_low[5], soil_details[6], soil_high[5]],
        'Moisture': [soil_low[6], soil_details[7], soil_high[6]]
    }
    from plot import plot
    plot(dict)

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
    
    with open("Card/soilhealthcard.pdf", 'wb') as f:
        f.write(pdf.read())
        
    # SMTP
    to_email = farmer_details[1]
    subject = 'Generated Soil Health Report!'
    body = 'Greetings User, Please find the attached soil health report.'
    attachment_path = 'Card/soilhealthcard.pdf'
    sender_email = 'sashankharry@gmail.com'
    sender_password = os.getenv('EMAIL_PASSWORD')
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    # Send email
    send_email(to_email, subject, body, attachment_path, sender_email, sender_password, smtp_server, smtp_port)

    return jsonify(f"Suggested Crop: <span style='color: red;'>{predict_crop}</span>\n   Suggested Fertilizer: <span style='color: red;'>{predict_ferti}</span>\n     Detected Soil Type: <span style='color: red;'>{soil_type}</span>")

@app.route('/download_pdf')
def download_pdf():
    with open("Card/soilhealthcard.pdf", 'rb') as f:
            pdf = BytesIO(f.read())
    return send_file(
        pdf,
        mimetype='application/pdf',
        download_name='Soil Health Card.pdf',
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)
