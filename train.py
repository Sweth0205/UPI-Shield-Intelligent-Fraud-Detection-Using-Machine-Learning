"""
Main training script for UPI fraud detection models
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_data
from models import FraudDetectionModels

def plot_results(results, save_path='results/model_comparison.png'):
    """
    Plot comparison of model performance
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model evaluation results
    save_path : str
        Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[name][metric] for name in model_names]
        bars = ax.bar(model_names, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim([0, 1])
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")

def main():
    print("=" * 80)
    print("UPI FRAUD DETECTION - MODEL TRAINING")
    print("=" * 80)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    
    # Initialize models
    fraud_models = FraudDetectionModels()
    
    # Train all models
    fraud_models.train_all(X_train, y_train)
    
    # Evaluate all models
    results = fraud_models.evaluate_all(X_test, y_test)
    
    # Get best model
    best_model_name, best_model = fraud_models.get_best_model(metric='f1_score')
    print(f"\n{'='*80}")
    print(f"Best Model: {best_model_name.upper().replace('_', ' ')}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"{'='*80}")
    
    # Save all models
    fraud_models.save_models('models')
    
    # Plot results
    plot_results(results)
    
    print("\nTraining completed successfully!")
    print("\nNext steps:")
    print("1. Use predict.py to make predictions on new transactions")
    print("2. Check results/model_comparison.png for model comparison")

if __name__ == "__main__":
    main()

