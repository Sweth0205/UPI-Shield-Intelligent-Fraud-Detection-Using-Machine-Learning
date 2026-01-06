"""
Prediction script for UPI fraud detection
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
from preprocess import DataPreprocessor
from models import FraudDetectionModels

def predict_transaction(transaction_data, model_name='random_forest', 
                       preprocessor_path='models/preprocessor.pkl',
                       model_dir='models'):
    """
    Predict if a transaction is fraudulent
    
    Parameters:
    -----------
    transaction_data : dict or pd.DataFrame
        Transaction data to predict
    model_name : str
        Name of the model to use
    preprocessor_path : str
        Path to saved preprocessor
    model_dir : str
        Directory containing saved models
        
    Returns:
    --------
    dict: Prediction results
    """
    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load(preprocessor_path)
    
    # Load model
    fraud_models = FraudDetectionModels()
    fraud_models.load_model(model_name, model_dir)
    
    # Convert to DataFrame if dict
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
    else:
        df = transaction_data.copy()
    
    # Preprocess
    X = preprocessor.transform(df)
    
    # Predict
    prediction = fraud_models.predict(model_name, X)
    probability = fraud_models.predict_proba(model_name, X)
    
    # Format results
    results = []
    for i in range(len(df)):
        results.append({
            'transaction_id': df.iloc[i].get('transaction_id', f'TXN{i}'),
            'is_fraud': int(prediction[i]),
            'fraud_probability': float(probability[i][1]),
            'risk_level': 'HIGH' if probability[i][1] > 0.7 else 'MEDIUM' if probability[i][1] > 0.3 else 'LOW'
        })
    
    return results

def main():
    """
    Main function for command-line usage
    """
    if len(sys.argv) < 2:
        print("Usage: python predict.py <csv_file> [model_name]")
        print("Example: python predict.py new_transactions.csv random_forest")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'random_forest'
    
    print(f"Loading transactions from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Predicting fraud for {len(df)} transactions using {model_name}...")
    results = predict_transaction(df, model_name=model_name)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = csv_file.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved to {output_file}")
    print("\nSummary:")
    print(f"  Total transactions: {len(results_df)}")
    print(f"  Predicted fraud: {results_df['is_fraud'].sum()}")
    print(f"  Predicted normal: {(results_df['is_fraud']==0).sum()}")
    print(f"\nRisk Level Distribution:")
    print(results_df['risk_level'].value_counts().to_string())

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        # Interactive example
        print("=" * 80)
        print("UPI FRAUD DETECTION - PREDICTION EXAMPLE")
        print("=" * 80)
        
        # Example transaction
        example_transaction = {
            'transaction_id': 'TXN000001',
            'amount': 5000.00,
            'merchant_category': 'Retail',
            'time_of_day': 23,
            'day_of_week': 6,
            'device_type': 'Mobile',
            'location_match': 0,
            'transaction_frequency': 10,
            'account_age_days': 15,
            'previous_chargebacks': 3,
            'amount_log': 8.52,
            'is_weekend': 1,
            'is_night': 1,
            'high_value': 1,
            'new_account': 1
        }
        
        print("\nExample Transaction:")
        for key, value in example_transaction.items():
            print(f"  {key}: {value}")
        
        print("\nMaking prediction...")
        result = predict_transaction(example_transaction)
        
        print("\nPrediction Result:")
        print(f"  Transaction ID: {result[0]['transaction_id']}")
        print(f"  Is Fraud: {result[0]['is_fraud']} ({'YES' if result[0]['is_fraud'] else 'NO'})")
        print(f"  Fraud Probability: {result[0]['fraud_probability']:.4f}")
        print(f"  Risk Level: {result[0]['risk_level']}")
    else:
        main()

