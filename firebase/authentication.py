from flask import render_template, Flask, request, redirect, session
import pyrebase
import os
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

app=Flask(__name__)
config = {
    "apiKey": "AIzaSyBcORtCnca_o90J4skz8qq4vaEuf4VdLbM",
    "authDomain": "soil-scan-system.firebaseapp.com",
    "projectId": "soil-scan-system",
    "storageBucket": "soil-scan-system.appspot.com",
    "messagingSenderId": "871665361075",
    "appId": "1:871665361075:web:5984d83137bbbdb9c2d0c6",
    "measurementId": "G-3RHFDD7KZF",
    "databaseURL": "https://soil-scan-system-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

app.secret_key = 'sabo'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email = request.form('email')
        password = request.form('password')
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            print(user)
            return redirect('/home')
        except:
            return "Invalid credentials"
    return render_template('main.html')

email = "test@gmail.com"
password = "123456"

user=auth.create_user_with_email_and_password(email, password)
print(user)
