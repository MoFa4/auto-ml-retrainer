"""
Data preparation script for California Housing dataset
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os

def prepare_initial_data():
    """
    Fetch California Housing dataset and prepare training data
    """
    print("📦 Fetching California Housing dataset...")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/new_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("✅ Directories created/verified")
    
    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📊 Features: {list(df.columns[:-1])}")
    print(f"🎯 Target: {df.columns[-1]}")
    
    # Create a smaller "new data" sample for testing retraining
    # We'll use 20% as "new data" to simulate new data arriving
    train_data, new_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save full dataset as training data
    data_path = 'data/training_data.csv'
    train_data.to_csv(data_path, index=False)
    print(f"💾 Training data saved to: {data_path}")
    
    # Save the new data sample
    new_data_path = 'data/new_data/new_batch_1.csv'
    new_data.to_csv(new_data_path, index=False)
    print(f"💾 New data sample saved to: {new_data_path}")
    
    print("\n✅ Data preparation complete!")
    print(f"   - Training data: {train_data.shape[0]} rows")
    print(f"   - New data batch: {new_data.shape[0]} rows")
    
    return train_data, new_data

if __name__ == "__main__":
    prepare_initial_data()