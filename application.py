from flask import Flask, request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

scaler = pickle.load(open(r"C:\Users\saman\OneDrive\Desktop\Data Science\ML Project -Logistic Regression\Models\Scaler.pkl","rb"))
model = pickle.load(open(r"C:\Users\saman\OneDrive\Desktop\Data Science\ML Project -Logistic Regression\Models\Logistic Regression.pkl","rb"))

# Route of home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result =""

    if request.method == 'POST':

        pregnancies = int(request.form.get('Pregnancies'))
        glucose = int (request.form.get('Glucose'))
        blood_Pressure = int (request.form.get('BloodPressure'))
        skin_thickness = int(request.form.get('SkinThickness'))
        insulin = int(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[pregnancies,glucose,blood_Pressure,skin_thickness,insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = model.predict(new_data)

        if predict[0] ==1:
            result ="Diabetic"
        else:
            result = "Non-Diabetic"
        
        return render_template('single_prediction.html',result=result)
    
    else:
        return render_template('home.html')

if __name__ ==  '__main__':
    app.run(host="0.0.0.0")



