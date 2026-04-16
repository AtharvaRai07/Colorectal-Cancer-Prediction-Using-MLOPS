from flask import Flask, render_template, request
import joblib
import numpy as np
from config.paths_config import *

model = joblib.load(MODEL_FILE_PATH)
scaler = joblib.load(SCALER_FILE_PATH)
encoder = joblib.load(LABEL_ENCODER_FILE_PATH)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        healthcare_cost = int(request.form['Healthcare_Cost'])
        tumor_size = int(request.form['Tumor_Size'])
        treatment_type = encoder['Treatment_Type'].transform([request.form['Treatment_Type']])[0]
        diabetes = encoder['Diabetes'].transform([request.form['Diabetes']])[0]
        age = int(request.form['Age'])
        survival_5_years = encoder['Survival_5_years'].transform([request.form['Survival_5_years']])[0]
        mortality_rate_per_100k = int(request.form['Mortality_Rate_per_100k'])
        country = encoder['Country'].transform([request.form['Country']])[0]
        physical_activity = encoder['Physical_Activity'].transform([request.form['Physical_Activity']])[0]
        insurance_status = encoder['Insurance_Status'].transform([request.form['Insurance_Status']])[0]

        input_data = np.array([[healthcare_cost, tumor_size, treatment_type, diabetes, age, survival_5_years, mortality_rate_per_100k, country, physical_activity, insurance_status]])

        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)[0]

        if prediction == 1:
            return render_template('index.html', prediction="The patient is likely to survive.")

        return render_template('index.html', prediction="The patient is likely to not survive.")

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
