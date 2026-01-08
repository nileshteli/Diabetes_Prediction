from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    model = None
    scaler = None

@app.route('/')
def home():
    with open('diabetes_prediction.html', 'r') as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in the correct order
        features = [
            data['pregnancies'],
            data['glucose'],
            data['bloodPressure'],
            data['skinThickness'],
            data['insulin'],
            data['bmi'],
            data['diabetesPedigree'],
            data['age']
        ]
        
        # Convert to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # Standardize the input
        if scaler:
            input_data_scaled = scaler.transform(input_data)
        else:
            return jsonify({'error': 'Model not loaded'})
        
        # Make prediction
        if model:
            prediction = model.predict(input_data_scaled)[0]
            probability = model.decision_function(input_data_scaled)[0]
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'Model not loaded'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
