"""
Model evaluation and comparison utilities
"""
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import yaml

def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_current_model_metadata():
    """Load metadata of current production model"""
    config = load_config()
    metadata_path = config['paths']['metadata']
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📋 Current model metadata loaded")
        print(f"   R² Score: {metadata['metrics']['r2_score']:.4f}")
        print(f"   Timestamp: {metadata['timestamp']}")
        return metadata
    except FileNotFoundError:
        print("⚠️  No current model found. This will be the first model.")
        return None

def evaluate_on_data(model, X, y):
    """Evaluate model on given data"""
    y_pred = model.predict(X)
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2_score': float(r2_score(y, y_pred))
    }
    
    return metrics

def compare_models(new_metrics, current_metadata, config):
    """
    Compare new model with current model
    Returns: (should_deploy: bool, reason: str)
    """
    print("\n" + "=" * 60)
    print("🔍 MODEL COMPARISON")
    print("=" * 60)
    
    # If no current model exists, deploy the new one
    if current_metadata is None:
        print("✅ No existing model. Deploying new model.")
        return True, "First model deployment"
    
    current_r2 = current_metadata['metrics']['r2_score']
    new_r2 = new_metrics['r2_score']
    improvement_threshold = config['thresholds']['improvement_threshold']
    
    print(f"📊 Current Model R²: {current_r2:.4f}")
    print(f"📊 New Model R²:     {new_r2:.4f}")
    print(f"📈 Improvement:      {(new_r2 - current_r2):.4f}")
    print(f"🎯 Required:         {improvement_threshold:.4f}")
    
    # Check if new model meets minimum threshold
    min_r2 = config['thresholds']['min_r2_score']
    if new_r2 < min_r2:
        reason = f"New model R² ({new_r2:.4f}) below minimum threshold ({min_r2:.4f})"
        print(f"\n❌ {reason}")
        return False, reason
    
    # Check if new model is better than current
    improvement = new_r2 - current_r2
    if improvement >= improvement_threshold:
        reason = f"New model improved by {improvement:.4f} (≥ {improvement_threshold:.4f})"
        print(f"\n✅ {reason}")
        return True, reason
    else:
        reason = f"Improvement {improvement:.4f} below threshold ({improvement_threshold:.4f})"
        print(f"\n❌ {reason}")
        return False, reason

if __name__ == "__main__":
    # Quick test
    metadata = load_current_model_metadata()
    if metadata:
        print(f"\n✅ Current model found with R² = {metadata['metrics']['r2_score']:.4f}")
    else:
        print("\n⚠️  No current model found")