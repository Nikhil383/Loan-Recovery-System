import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class LoanDataProcessor:
    """
    Class for preprocessing loan data for machine learning models.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.preprocessor = None
        self.categorical_features = ['gender', 'employment_status', 'payment_history']
        self.numerical_features = ['age', 'annual_income', 'credit_score', 'loan_amount', 
                                  'interest_rate', 'loan_term', 'days_past_due', 
                                  'previous_defaults', 'monthly_payment', 'debt_to_income']
        
    def fit(self, X):
        """
        Fit the preprocessor on the training data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The training data
            
        Returns:
        --------
        self : LoanDataProcessor
            The fitted processor
        """
        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Fit the preprocessor
        self.preprocessor.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using the fitted preprocessor.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The data to transform
            
        Returns:
        --------
        numpy.ndarray
            The transformed data
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")
        
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X):
        """
        Fit the preprocessor and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The data to fit and transform
            
        Returns:
        --------
        numpy.ndarray
            The transformed data
        """
        return self.fit(X).transform(X)
    
    def get_feature_names(self):
        """
        Get the names of the transformed features.
        
        Returns:
        --------
        list
            List of feature names after transformation
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")
        
        # Get feature names from the column transformer
        feature_names = []
        
        # Get numerical feature names (these stay the same)
        feature_names.extend(self.numerical_features)
        
        # Get categorical feature names (these are expanded by one-hot encoding)
        categorical_features = self.preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(
            self.categorical_features)
        feature_names.extend(categorical_features)
        
        return feature_names
    
    def prepare_data(self, data, target_column='recovery_status'):
        """
        Prepare data for model training or prediction.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to prepare
        target_column : str, optional
            The name of the target column, by default 'recovery_status'
            
        Returns:
        --------
        tuple
            (X, y) if target_column is in data, otherwise just X
        """
        # Drop customer_id as it's not a feature
        if 'customer_id' in data.columns:
            data = data.drop('customer_id', axis=1)
        
        if target_column in data.columns:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            return X, y
        else:
            return data
