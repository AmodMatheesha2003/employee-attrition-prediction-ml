from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("Models/employee-attrition-prediction.pkl")
scaler = joblib.load("Models/scaler.pkl")

# Categorical value mappings
CATEGORICAL_MAPPINGS = {
    'Gender': {'Male': 0, 'Female': 1},
    'Job Role': {'Healthcare': 0, 'Education': 1, 'Media': 2, 'Technology': 3, 'Finance': 4},
    'Work-Life Balance': {'Excellent': 0, 'Good': 1, 'Fair': 2, 'Poor': 3},
    'Job Satisfaction': {'High': 0, 'Very High': 1, 'Medium': 2, 'Low': 3},
    'Performance Rating': {'Average': 0, 'High': 1, 'Below Average': 2, 'Low': 3},
    'Overtime': {'Yes': 0, 'No': 1},
    'Education Level': {'Master’s Degree': 0, 'Associate Degree': 1, 'High School': 2, 'Bachelor’s Degree': 3, 'PhD': 4},
    'Marital Status': {'Married': 0, 'Single': 1, 'Divorced': 2},
    'Job Level': {'Mid': 0, 'Entry': 1, 'Senior': 2},
    'Company Size': {'Large': 0, 'Medium': 1, 'Small': 2},
    'Remote Work': {'No': 0, 'Yes': 1},
    'Leadership Opportunities': {'No': 0, 'Yes': 1},
    'Innovation Opportunities': {'No': 0, 'Yes': 1},
    'Company Reputation': {'Poor': 0, 'Good': 1, 'Fair': 2, 'Excellent': 3},
    'Employee Recognition': {'Medium': 0, 'High': 1, 'Low': 2, 'Very High': 3}
}

COLUMN_ORDER = [
    'Age', 'Gender', 'Years at Company', 'Job Role', 'Monthly Income', 'Work-Life Balance', 'Job Satisfaction',
    'Performance Rating', 'Number of Promotions', 'Overtime', 'Distance from Home', 'Education Level', 'Marital Status', 'Number of Dependents',
    'Job Level', 'Company Size', 'Company Tenure', 'Remote Work', 'Leadership Opportunities', 'Innovation Opportunities', 'Company Reputation', 'Employee Recognition'
]

NUMERICAL_COLS = ['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions',
                  'Distance from Home', 'Number of Dependents', 'Company Tenure']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}

    # Process form data
    for col in COLUMN_ORDER:
        value = request.form.get(col)

        # Check if value is None or empty
        if value is None or value == "":
            return render_template('error.html', message=f"Please provide a value for {col}.")

        # Map categorical values
        if col in CATEGORICAL_MAPPINGS:
            if value not in CATEGORICAL_MAPPINGS[col]:
                return render_template('error.html', message=f"Invalid value for {col}.")
            input_data[col] = CATEGORICAL_MAPPINGS[col][value]
        else:
            try:
                input_data[col] = float(value)
            except ValueError:
                return render_template('error.html', message=f"Invalid number for {col}.")

    # Create DataFrame with the same order of columns as the model expects
    input_df = pd.DataFrame([input_data], columns=COLUMN_ORDER)

    # Separate numerical columns and scale them
    input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])

    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "Left" if prediction == 1 else "Stayed"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
