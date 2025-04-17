import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.data_processor import LoanDataProcessor

class LoanRecoveryModel:
    """
    Machine learning model for predicting loan recovery.
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the loan recovery model.

        Parameters:
        -----------
        model_type : str, optional
            Type of model to use, by default 'random_forest'
            Only 'random_forest' is supported
        """
        self.model_type = 'random_forest'  # Always use Random Forest
        self.model = None
        self.processor = LoanDataProcessor()

        # Initialize the Random Forest model
        self.model = RandomForestClassifier(random_state=42)

    def train(self, data, target_column='recovery_status', test_size=0.2, tune_hyperparameters=False):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        data : pandas.DataFrame
            The training data
        target_column : str, optional
            The name of the target column, by default 'recovery_status'
        test_size : float, optional
            Proportion of data to use for testing, by default 0.2
        tune_hyperparameters : bool, optional
            Whether to perform hyperparameter tuning, by default False

        Returns:
        --------
        dict
            Dictionary containing model performance metrics
        """
        # Prepare data
        X, y = self.processor.prepare_data(data, target_column)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Preprocess the data
        X_train_processed = self.processor.fit_transform(X_train)
        X_test_processed = self.processor.transform(X_test)

        # Tune hyperparameters if requested
        if tune_hyperparameters:
            self._tune_hyperparameters(X_train_processed, y_train)

        # Train the model
        self.model.fit(X_train_processed, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_processed)
        y_prob = self.model.predict_proba(X_test_processed)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': self.model.score(X_test_processed, y_test),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.processor.get_feature_names()
            metrics['feature_importance'] = dict(zip(feature_names, self.model.feature_importances_))

        return metrics

    def predict(self, data):
        """
        Make predictions on new data.

        Parameters:
        -----------
        data : pandas.DataFrame
            The data to make predictions on

        Returns:
        --------
        numpy.ndarray
            Array of predicted probabilities of recovery
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Prepare data
        if 'recovery_status' in data.columns:
            X, _ = self.processor.prepare_data(data)
        else:
            X = self.processor.prepare_data(data)

        # Preprocess the data
        X_processed = self.processor.transform(X)

        # Make predictions
        return self.model.predict_proba(X_processed)[:, 1]

    def save_model(self, model_path, processor_path=None):
        """
        Save the trained model and preprocessor to disk.

        Parameters:
        -----------
        model_path : str
            Path to save the model
        processor_path : str, optional
            Path to save the preprocessor, by default None
            If None, will use model_path with '_processor' appended
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Save the model
        joblib.dump(self.model, model_path)

        # Save the preprocessor
        if processor_path is None:
            processor_path = model_path.replace('.pkl', '_processor.pkl')

        joblib.dump(self.processor, processor_path)

    @classmethod
    def load_model(cls, model_path, processor_path=None):
        """
        Load a trained model and preprocessor from disk.

        Parameters:
        -----------
        model_path : str
            Path to the saved model
        processor_path : str, optional
            Path to the saved preprocessor, by default None
            If None, will use model_path with '_processor' appended

        Returns:
        --------
        LoanRecoveryModel
            The loaded model
        """
        # Create a new instance
        instance = cls()

        # Load the model
        instance.model = joblib.load(model_path)

        # Load the preprocessor
        if processor_path is None:
            processor_path = model_path.replace('.pkl', '_processor.pkl')

        instance.processor = joblib.load(processor_path)

        return instance

    def _tune_hyperparameters(self, X_train, y_train):
        """
        Perform hyperparameter tuning for Random Forest model.

        Parameters:
        -----------
        X_train : numpy.ndarray
            The processed training features
        y_train : numpy.ndarray
            The training target values
        """
        # Random Forest hyperparameters
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_

    def plot_feature_importance(self, top_n=10):
        """
        Plot feature importance for the trained model.

        Parameters:
        -----------
        top_n : int, optional
            Number of top features to display, by default 10

        Returns:
        --------
        matplotlib.figure.Figure
            The feature importance plot
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances.")

        # Get feature names and importances
        feature_names = self.processor.get_feature_names()
        importances = self.model.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1]

        # Take top N features
        indices = indices[:top_n]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top {} Feature Importances'.format(top_n))
        plt.tight_layout()

        return fig

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix for model predictions.

        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels

        Returns:
        --------
        matplotlib.figure.Figure
            The confusion matrix plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Not Recovered', 'Recovered'])
        ax.set_yticklabels(['Not Recovered', 'Recovered'])
        plt.tight_layout()

        return fig
