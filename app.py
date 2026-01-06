"""
Flask backend API for UPI Fraud Detection
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from preprocess import DataPreprocessor
from models import FraudDetectionModels
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
preprocessor = None
fraud_models = None
model_name = 'random_forest'  # Default model

def load_models():
    """Load preprocessor and models"""
    global preprocessor, fraud_models
    
    if preprocessor is None:
        preprocessor = DataPreprocessor()
        preprocessor.load('models/preprocessor.pkl')
    
    if fraud_models is None:
        fraud_models = FraudDetectionModels()
        fraud_models.load_model(model_name, 'models')

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for fraud prediction"""
    try:
        # Load models if not already loaded
        load_models()
        
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['amount', 'merchant_category', 'time_of_day', 'day_of_week', 
                          'device_type', 'location_match', 'transaction_frequency', 
                          'account_age_days', 'previous_chargebacks']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create transaction dictionary
        transaction = {
            'transaction_id': data.get('transaction_id', 'TXN000001'),
            'amount': float(data['amount']),
            'merchant_category': data['merchant_category'],
            'time_of_day': int(data['time_of_day']),
            'day_of_week': int(data['day_of_week']),
            'device_type': data['device_type'],
            'location_match': int(data['location_match']),
            'transaction_frequency': int(data['transaction_frequency']),
            'account_age_days': int(data['account_age_days']),
            'previous_chargebacks': int(data['previous_chargebacks']),
        }
        
        # Calculate derived features
        transaction['amount_log'] = np.log1p(transaction['amount'])
        transaction['is_weekend'] = 1 if transaction['day_of_week'] >= 5 else 0
        transaction['is_night'] = 1 if (transaction['time_of_day'] >= 22 or transaction['time_of_day'] <= 5) else 0
        
        # Calculate high_value threshold (using a reasonable default)
        high_value_threshold = 10000  # You can adjust this
        transaction['high_value'] = 1 if transaction['amount'] > high_value_threshold else 0
        transaction['new_account'] = 1 if transaction['account_age_days'] < 90 else 0
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        prediction = fraud_models.predict(model_name, X)
        probability = fraud_models.predict_proba(model_name, X)
        
        fraud_prob = float(probability[0][1])
        is_fraud = int(prediction[0])
        
        # Determine risk level
        if fraud_prob > 0.7:
            risk_level = 'HIGH'
        elif fraud_prob > 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Return result
        result = {
            'is_fraud': is_fraud,
            'fraud_probability': round(fraud_prob, 4),
            'risk_level': risk_level,
            'message': 'FRAUD DETECTED - Transaction flagged as suspicious' if is_fraud else 'Transaction appears legitimate'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models = ['logistic_regression', 'random_forest', 'svm', 'xgboost']
    return jsonify({'models': models, 'current': model_name})

@app.route('/api/set_model', methods=['POST'])
def set_model():
    """Set the model to use for predictions"""
    global model_name, fraud_models
    
    data = request.json
    new_model = data.get('model')
    
    if new_model not in ['logistic_regression', 'random_forest', 'svm', 'xgboost']:
        return jsonify({'error': 'Invalid model name'}), 400
    
    model_name = new_model
    fraud_models = None  # Reset to reload with new model
    load_models()
    
    return jsonify({'message': f'Model changed to {model_name}', 'model': model_name})

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists('models/preprocessor.pkl'):
        print("ERROR: Models not found. Please run 'python train.py' first.")
        exit(1)
    
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)

