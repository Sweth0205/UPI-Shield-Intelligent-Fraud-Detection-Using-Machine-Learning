# UPI Fraud Detection using Machine Learning

A comprehensive machine learning system for detecting fraudulent UPI transactions using multiple ML algorithms including Logistic Regression, Random Forest, SVM, and XGBoost.

## Features

- **Multiple ML Models**: Implements Logistic Regression, Random Forest, SVM, and XGBoost
- **Synthetic Data Generation**: Includes script to generate realistic synthetic UPI transaction data
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1-score, ROC-AUC metrics
- **Easy Prediction**: Simple interface to predict fraud on new transactions
- **Model Comparison**: Visual comparison of all models' performance
- **Web Interface**: Beautiful, modern web UI for real-time fraud detection

## Project Structure

```
.
├── data/                      # Data directory
│   └── upi_transactions.csv  # Generated transaction data
├── models/                    # Saved models directory
│   ├── *.pkl                 # Trained models
│   └── preprocessor.pkl       # Data preprocessor
├── results/                   # Results directory
│   └── model_comparison.png  # Model comparison visualization
├── templates/                 # HTML templates
│   └── index.html            # Web interface
├── static/                    # Static files (CSS, JS)
│   ├── style.css             # Stylesheet
│   └── script.js             # Frontend JavaScript
├── generate_data.py           # Synthetic data generation script
├── preprocess.py              # Data preprocessing module
├── models.py                  # ML models implementation
├── train.py                   # Training script
├── predict.py                 # Prediction script
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Generate Synthetic Data

First, generate synthetic UPI transaction data:

```bash
python generate_data.py
```

This will create `data/upi_transactions.csv` with 10,000 transactions (5% fraud rate).

### Step 2: Train Models

Train all ML models:

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression, Random Forest, SVM, and XGBoost models
- Evaluate all models on test data
- Save trained models to `models/` directory
- Generate comparison visualization in `results/`

### Step 3: Make Predictions

#### Option A: Web Interface (Recommended)

Launch the web application:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

The web interface provides:
- Beautiful, modern UI
- Real-time fraud detection
- Visual risk indicators
- Model selection dropdown
- Responsive design for mobile and desktop

#### Option B: Predict on a CSV file

Create a CSV file with transaction data (same format as training data) and run:

```bash
python predict.py new_transactions.csv random_forest
```

#### Option C: Use Python API

```python
from predict import predict_transaction

# Example transaction
transaction = {
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

# Predict
result = predict_transaction(transaction, model_name='random_forest')
print(result)
```

## Features Used

The model uses the following features:

- **amount**: Transaction amount
- **merchant_category**: Category of merchant (Retail, Food, Transport, etc.)
- **time_of_day**: Hour of day (0-23)
- **day_of_week**: Day of week (0-6)
- **device_type**: Device used (Mobile, Desktop, Tablet)
- **location_match**: Whether location matches user's usual location
- **transaction_frequency**: Number of transactions in recent period
- **account_age_days**: Age of account in days
- **previous_chargebacks**: Number of previous chargebacks
- **amount_log**: Logarithm of transaction amount
- **is_weekend**: Whether transaction is on weekend
- **is_night**: Whether transaction is during night hours
- **high_value**: Whether transaction is high value
- **new_account**: Whether account is new (< 90 days)

## Models

1. **Logistic Regression**: Linear model with balanced class weights
2. **Random Forest**: Ensemble of decision trees (100 trees, max depth 10)
3. **SVM**: Support Vector Machine with RBF kernel
4. **XGBoost**: Gradient boosting classifier (100 estimators)

## Evaluation Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Model Selection

The training script automatically selects the best model based on F1-score, which is important for fraud detection as it balances precision and recall.

## Customization

### Adjust Fraud Ratio

Edit `generate_data.py`:

```python
df = generate_synthetic_upi_data(n_samples=10000, fraud_ratio=0.1)  # 10% fraud
```

### Add More Models

Edit `models.py` to add additional models to the `FraudDetectionModels` class.

### Modify Features

Edit `preprocess.py` to add or remove features from the preprocessing pipeline.

## Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib
- imbalanced-learn
- flask
- flask-cors

## Notes

- The synthetic data generator creates realistic patterns where fraudulent transactions tend to have:
  - Higher transaction amounts
  - Odd transaction times (late night/early morning)
  - Weekend transactions
  - Location mismatches
  - Higher transaction frequency
  - Newer accounts
  - More previous chargebacks

- For production use, replace synthetic data with real transaction data.

- The models use class balancing to handle the imbalanced nature of fraud detection (few fraud cases).

## License

This project is provided as-is for educational and research purposes.

## Author

Created for UPI fraud detection using machine learning techniques.

