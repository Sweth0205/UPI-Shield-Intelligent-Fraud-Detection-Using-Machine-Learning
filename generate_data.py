"""
Script to generate synthetic UPI transaction data for fraud detection
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_upi_data(n_samples=10000, fraud_ratio=0.05):
    """
    Generate synthetic UPI transaction data
    
    Parameters:
    -----------
    n_samples : int
        Number of transactions to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (default: 5%)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic transaction data
    """
    np.random.seed(42)
    random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate normal transactions
    normal_data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(n_normal)],
        'amount': np.random.lognormal(mean=5, sigma=1, size=n_normal).round(2),
        'merchant_category': np.random.choice(['Retail', 'Food', 'Transport', 'Entertainment', 'Healthcare', 'Education'], n_normal),
        'time_of_day': np.random.randint(0, 24, n_normal),
        'day_of_week': np.random.randint(0, 7, n_normal),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_normal, p=[0.7, 0.2, 0.1]),
        'location_match': np.random.choice([0, 1], n_normal, p=[0.2, 0.8]),
        'transaction_frequency': np.random.poisson(lam=3, size=n_normal),
        'account_age_days': np.random.randint(30, 3650, n_normal),
        'previous_chargebacks': np.random.poisson(lam=0.1, size=n_normal),
        'is_fraud': [0] * n_normal
    }
    
    # Generate fraudulent transactions
    fraud_data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(n_normal, n_samples)],
        'amount': np.random.lognormal(mean=6.5, sigma=1.5, size=n_fraud).round(2),  # Higher amounts
        'merchant_category': np.random.choice(['Retail', 'Food', 'Transport', 'Entertainment', 'Healthcare', 'Education'], n_fraud),
        'time_of_day': np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], n_fraud),  # Odd hours
        'day_of_week': np.random.choice([5, 6], n_fraud, p=[0.6, 0.4]),  # Weekends
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_fraud, p=[0.5, 0.4, 0.1]),
        'location_match': np.random.choice([0, 1], n_fraud, p=[0.7, 0.3]),  # More location mismatches
        'transaction_frequency': np.random.poisson(lam=8, size=n_fraud),  # Higher frequency
        'account_age_days': np.random.randint(1, 90, n_fraud),  # Newer accounts
        'previous_chargebacks': np.random.poisson(lam=2, size=n_fraud),  # More chargebacks
        'is_fraud': [1] * n_fraud
    }
    
    # Combine and shuffle
    normal_df = pd.DataFrame(normal_data)
    fraud_df = pd.DataFrame(fraud_data)
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some derived features
    df['amount_log'] = np.log1p(df['amount'])
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['time_of_day'] >= 22) | (df['time_of_day'] <= 5)).astype(int)
    df['high_value'] = (df['amount'] > df['amount'].quantile(0.9)).astype(int)
    df['new_account'] = (df['account_age_days'] < 90).astype(int)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic UPI transaction data...")
    df = generate_synthetic_upi_data(n_samples=10000, fraud_ratio=0.05)
    df.to_csv('data/upi_transactions.csv', index=False)
    print(f"Generated {len(df)} transactions")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Normal cases: {(df['is_fraud']==0).sum()} ({(df['is_fraud']==0).mean()*100:.2f}%)")
    print("\nData saved to 'data/upi_transactions.csv'")

