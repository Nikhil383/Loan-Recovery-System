import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.utils.data_generator import generate_loan_data
from src.models.loan_recovery_model import LoanRecoveryModel

def train_and_save_model(data_path=None, model_type='random_forest', tune_hyperparameters=False):
    """
    Train a loan recovery model and save it to disk.

    Parameters:
    -----------
    data_path : str, optional
        Path to the loan data CSV file, by default None
        If None, generates synthetic data
    model_type : str, optional
        Type of model to train, by default 'random_forest'
    tune_hyperparameters : bool, optional
        Whether to tune hyperparameters, by default False

    Returns:
    --------
    dict
        Dictionary containing model performance metrics
    """
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load or generate data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
    else:
        print("Generating synthetic loan data")
        data = generate_loan_data(n_samples=1000)

        # Save generated data
        data_path = 'data/loan_data.csv'
        data.to_csv(data_path, index=False)
        print(f"Saved generated data to {data_path}")

    # Print data summary
    print(f"\nData shape: {data.shape}")
    print(f"Recovery rate: {data['recovery_status'].mean() * 100:.2f}%")

    # Train model
    print(f"\nTraining {model_type} model...")
    model = LoanRecoveryModel(model_type=model_type)
    metrics = model.train(data, tune_hyperparameters=tune_hyperparameters)

    # Print performance metrics
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    for label, values in metrics['classification_report'].items():
        if label in ['0', '1']:
            label_name = 'Not Recovered' if label == '0' else 'Recovered'
            print(f"{label_name}:")
            print(f"  Precision: {values['precision']:.4f}")
            print(f"  Recall: {values['recall']:.4f}")
            print(f"  F1-score: {values['f1-score']:.4f}")

    # Save model
    model_path = f"models/loan_recovery_{model_type}.pkl"
    model.save_model(model_path)
    print(f"\nSaved model to {model_path}")

    # Plot feature importance if available
    if 'feature_importance' in metrics:
        fig = model.plot_feature_importance(top_n=10)
        fig_path = f"models/feature_importance_{model_type}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved feature importance plot to {fig_path}")

    return metrics

if __name__ == "__main__":
    # Train only Random Forest model
    print(f"\n{'='*50}")
    print(f"Training Random Forest Model")
    print(f"{'='*50}")
    train_and_save_model(model_type='random_forest', tune_hyperparameters=True)
