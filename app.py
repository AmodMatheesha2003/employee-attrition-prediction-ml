from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
gb_model = joblib.load('Models/gb_model.pkl')
scaler = joblib.load('Models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = {
        'Age': float(request.form['Age']),
        'Gender': int(request.form['Gender']),
        'Job Role': int(request.form['Job Role']),
        'Work-Life Balance': int(request.form['Work-Life Balance']),
        'Job Satisfaction': int(request.form['Job Satisfaction']),
        'Performance Rating': int(request.form['Performance Rating']),
        'Number of Promotions': float(request.form['Number of Promotions']),
        'Overtime': int(request.form['Overtime']),
        'Distance from Home': float(request.form['Distance from Home']),
        'Education Level': int(request.form['Education Level']),
        'Marital Status': int(request.form['Marital Status']),
        'Number of Dependents': float(request.form['Number of Dependents']),
        'Job Level': int(request.form['Job Level']),
        'Company Size': int(request.form['Company Size']),
        'Remote Work': int(request.form['Remote Work']),
        'Leadership Opportunities': int(request.form['Leadership Opportunities']),
        'Innovation Opportunities': int(request.form['Innovation Opportunities']),
        'Company Reputation': int(request.form['Company Reputation']),
        'Employee Recognition': int(request.form['Employee Recognition']),
        'Experience': float(request.form['Experience']),
        'High Income': int(request.form['High Income']),
        'Promotion_Frequency': float(request.form['Promotion_Frequency'])
    }

    # Create DataFrame and scale features
    df = pd.DataFrame([features])
    scaled_features = scaler.transform(df)

    # Make prediction
    prediction = gb_model.predict(scaled_features)[0]
    result = 'Left' if prediction == 0 else 'Stayed'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)