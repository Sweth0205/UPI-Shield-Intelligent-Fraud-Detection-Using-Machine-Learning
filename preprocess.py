"""
Data preprocessing and feature engineering for UPI fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def fit_transform(self, df):
        """
        Fit preprocessor and transform data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple: (X, y) - Features and target
        """
        df = df.copy()
        
        # Separate features and target
        if 'is_fraud' in df.columns:
            y = df['is_fraud'].values
            df = df.drop('is_fraud', axis=1)
        else:
            y = None
        
        # Drop transaction_id (not a feature)
        if 'transaction_id' in df.columns:
            df = df.drop('transaction_id', axis=1)
        
        # Encode categorical variables
        categorical_cols = ['merchant_category', 'device_type']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Store feature columns
        self.feature_columns = df.columns.tolist()
        
        # Scale numerical features
        X = self.scaler.fit_transform(df)
        
        return X, y
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        np.array: Transformed features
        """
        df = df.copy()
        
        # Drop transaction_id if present
        if 'transaction_id' in df.columns:
            df = df.drop('transaction_id', axis=1)
        
        # Encode categorical variables
        categorical_cols = ['merchant_category', 'device_type']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        
        # Ensure same columns as training
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.feature_columns]
        
        # Scale numerical features
        X = self.scaler.transform(df)
        
        return X
    
    def save(self, filepath='models/preprocessor.pkl'):
        """Save preprocessor to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath='models/preprocessor.pkl'):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        print(f"Preprocessor loaded from {filepath}")

def load_and_preprocess_data(data_path='data/upi_transactions.csv', test_size=0.2, random_state=42):
    """
    Load and preprocess data for training
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    X, y = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, preprocessor

