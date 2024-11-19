#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GameDataLoader:
    """Data loader for Excel-based game data"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self):
        """Load data from Excel file"""
        print("Loading game data from Excel...")
        try:
            self.raw_data = pd.read_excel(self.file_path, sheet_name=None)
            print(f"Successfully loaded {len(self.raw_data)} sheets")
            return self.raw_data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def process_data(self):
        """Process and structure the data for training"""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        processed_data = {}
        
        # Process each sheet
        for sheet_name, df in self.raw_data.items():
            print(f"Processing sheet: {sheet_name}")
            
            # Clean and structure data
            cleaned_df = self._clean_sheet_data(df)
            if cleaned_df is not None:
                processed_data[sheet_name] = cleaned_df
        
        self.processed_data = processed_data
        return processed_data
    
    def _clean_sheet_data(self, df):
        """Clean and structure sheet data"""
        try:
            # Remove any completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Forward fill any missing values in header rows
            if df.columns.dtype == 'object':
                df.columns = df.columns.fillna(method='ffill')
            
            return df
        except Exception as e:
            print(f"Error cleaning sheet data: {str(e)}")
            return None

class ModelTrainer:
    """Model trainer using Excel data"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.models = {}
        self.scalers = {}
        
    def prepare_training_data(self):
        """Prepare data for training"""
        if self.data_loader.processed_data is None:
            print("No processed data available. Please process data first.")
            return None
            
        training_data = {}
        
        for sheet_name, df in self.data_loader.processed_data.items():
            print(f"Preparing training data from sheet: {sheet_name}")
            
            # Extract relevant features and targets
            features, targets = self._extract_features_targets(df)
            
            if features is not None and targets is not None:
                training_data[sheet_name] = {
                    'features': features,
                    'targets': targets
                }
        
        return training_data
    
    def train_models(self, training_data):
        """Train models using prepared data"""
        if training_data is None:
            print("No training data available.")
            return
            
        for sheet_name, data in training_data.items():
            print(f"Training models for: {sheet_name}")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data['features'])
            
            # Train models for each target
            for target_col in data['targets'].columns:
                print(f"Training model for target: {target_col}")
                
                model = self._train_single_model(
                    X_scaled,
                    data['targets'][target_col]
                )
                
                self.models[f"{sheet_name}_{target_col}"] = model
                self.scalers[f"{sheet_name}_{target_col}"] = scaler
    
    def _extract_features_targets(self, df):
        """Extract features and targets from dataframe"""
        try:
            # Identify feature and target columns based on data structure
            # This needs to be customized based on your Excel structure
            feature_cols = [col for col in df.columns if 'input' in col.lower()]
            target_cols = [col for col in df.columns if 'output' in col.lower()]
            
            if not feature_cols or not target_cols:
                print("Could not identify feature/target columns")
                return None, None
            
            features = df[feature_cols]
            targets = df[target_cols]
            
            return features, targets
            
        except Exception as e:
            print(f"Error extracting features/targets: {str(e)}")
            return None, None
    
    def _train_single_model(self, X, y):
        """Train a single model"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X, y)
        return model

class PredictionSystem:
    """System for making predictions using trained models"""
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
        
    def predict(self, input_data, model_key):
        """Make predictions using trained model"""
        if model_key not in self.model_trainer.models:
            print(f"No trained model found for: {model_key}")
            return None
            
        model = self.model_trainer.models[model_key]
        scaler = self.model_trainer.scalers[model_key]
        
        # Scale input data
        X_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(X_scaled)
        
        return prediction

#%%
# Example usage
if __name__ == "__main__":
    # Initialize data loader
    data_loader = GameDataLoader("game_data/w01_MBA_c5_out.xlsx")
    
    # Load and process data
    raw_data = data_loader.load_data()
    processed_data = data_loader.process_data()
    
    # Initialize and train models
    trainer = ModelTrainer(data_loader)
    training_data = trainer.prepare_training_data()
    trainer.train_models(training_data)
    
    # Initialize prediction system
    predictor = PredictionSystem(trainer)
    
    # Print available models
    print("\nTrained Models:")
    for model_key in trainer.models.keys():
        print(f"- {model_key}")
    
    # Create visualization of model performance
    plt.figure(figsize=(15, 5))
    
    for i, (model_key, model) in enumerate(trainer.models.items(), 1):
        plt.subplot(1, len(trainer.models), i)
        
        # Get feature importances
        importances = model.feature_importances_
        features = range(len(importances))
        
        plt.bar(features, importances)
        plt.title(f'Feature Importance\n{model_key}')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.show()