from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application
# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form.to_dict()
    
    # Preprocess the input data
    sex = int(data['sex'])
    smoker = int(data['smoker'])
    region = int(data['region'])
    age = int(data['age'])
    bmi = float(data['bmi'])
    children = int(data['children'])

    # Make prediction
    prediction = model.predict([[age, sex, bmi, children, smoker, region]])
    
    # Format the prediction as a string
    prediction_str = f"{prediction[0]:.2f}"

    return render_template('index.html', prediction=prediction_str)

if __name__ == "__main__":
    app.run(host="0.0.0.0")