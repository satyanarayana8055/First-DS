from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Gather form data and prepare for prediction
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),  # Corrected to match form field
            writing_score=float(request.form.get('writing_score'))   # Corrected to match form field
        )

        # Convert data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Perform prediction
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Render prediction result
        return render_template('index.html', results=results[0])  # Show the result on the same page or another

    return render_template('index.html')  # In case of GET request

app.config['DEBUG'] = True  # This will override the debug mode

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
