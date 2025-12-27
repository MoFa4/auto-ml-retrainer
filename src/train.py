"""
Model training script with MLflow tracking
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import yaml
import joblib
import json
from datetime import datetime
import os

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path):
    """Load training data"""
    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_features(df, target_column):
    """Split features and target"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_model(X_train, y_train, params):
    """Train Random Forest model"""
    print("🎯 Training Random Forest model...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print("✅ Model training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2_score': float(r2_score(y_test, y_pred))
    }
    
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R² Score: {metrics['r2_score']:.4f}")
    
    return metrics

def save_model(model, metrics, config):
    """Save model and metadata"""
    # Ensure models directory exists
    os.makedirs(config['paths']['models'], exist_ok=True)
    
    model_path = config['paths']['current_model']
    metadata_path = config['paths']['metadata']
    
    # Save model
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'model_type': config['model']['type'],
        'params': config['params']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved to: {metadata_path}")
    
    return metadata

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🚀 STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Setup MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        # Load data
        df = load_data(config['paths']['data'])
        
        # Prepare features
        X, y = prepare_features(df, config['model']['target_column'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['model']['test_size'],
            random_state=config['model']['random_state']
        )
        
        print(f"📊 Train set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Train model
        model = train_model(X_train, y_train, config['params'])
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log to MLflow
        mlflow.log_params(config['params'])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        metadata = save_model(model, metrics, config)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"📈 Final R² Score: {metrics['r2_score']:.4f}")
        print(f"📁 Model saved at: {config['paths']['current_model']}")
        
        return model, metrics

if __name__ == "__main__":
    main()