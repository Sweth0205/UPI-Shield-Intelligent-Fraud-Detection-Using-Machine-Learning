"""
Machine Learning models for UPI fraud detection
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
import numpy as np

class FraudDetectionModels:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        self.trained_models = {}
        self.model_scores = {}
    
    def train_all(self, X_train, y_train):
        """
        Train all models
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        """
        print("Training models...")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"[OK] {name} trained successfully")
        
        print("\n" + "=" * 50)
        print("All models trained successfully!")
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Parameters:
        -----------
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
            
        Returns:
        --------
        dict: Evaluation metrics for all models
        """
        results = {}
        
        print("\nEvaluating models...")
        print("=" * 80)
        
        for name, model in self.trained_models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm
            }
            
            # Print results
            print(f"\n{name.upper().replace('_', ' ')}")
            print("-" * 80)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  True Negatives:  {cm[0][0]}")
            print(f"  False Positives: {cm[0][1]}")
            print(f"  False Negatives: {cm[1][0]}")
            print(f"  True Positives:  {cm[1][1]}")
        
        print("\n" + "=" * 80)
        self.model_scores = results
        return results
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best performing model based on a metric
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison (default: 'f1_score')
            
        Returns:
        --------
        tuple: (model_name, model_object)
        """
        if not self.model_scores:
            raise ValueError("Models not evaluated yet. Call evaluate_all() first.")
        
        best_model_name = max(
            self.model_scores.keys(),
            key=lambda x: self.model_scores[x][metric]
        )
        
        return best_model_name, self.trained_models[best_model_name]
    
    def predict(self, model_name, X):
        """
        Make predictions using a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : np.array
            Features to predict on
            
        Returns:
        --------
        np.array: Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name, X):
        """
        Get prediction probabilities using a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : np.array
            Features to predict on
            
        Returns:
        --------
        np.array: Prediction probabilities
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        return self.trained_models[model_name].predict_proba(X)
    
    def save_models(self, directory='models'):
        """
        Save all trained models to disk
        
        Parameters:
        -----------
        directory : str
            Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filepath = os.path.join(directory, f'{name}.pkl')
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save scores
        scores_path = os.path.join(directory, 'model_scores.pkl')
        joblib.dump(self.model_scores, scores_path)
        print(f"Saved model scores to {scores_path}")
    
    def load_model(self, model_name, directory='models'):
        """
        Load a specific model from disk
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
        directory : str
            Directory containing models
        """
        filepath = os.path.join(directory, f'{model_name}.pkl')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"Loaded {model_name} from {filepath}")

