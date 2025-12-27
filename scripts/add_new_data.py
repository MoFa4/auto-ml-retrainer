"""
Script to simulate new data arrival
Useful for testing the automated pipeline
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

def generate_new_data_batch(num_samples=1000):
    """Generate a new batch of data for testing"""
    
    print(f"🎲 Generating {num_samples} new data samples...")
    
    # Load full dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Take a random sample
    new_batch = df.sample(n=num_samples, random_state=int(datetime.now().timestamp()))
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/new_data/batch_{timestamp}.csv"
    
    # Ensure directory exists
    os.makedirs('data/new_data', exist_ok=True)
    
    # Save
    new_batch.to_csv(filename, index=False)
    
    print(f"✅ New data batch saved: {filename}")
    print(f"📊 Samples: {num_samples}")
    print(f"💡 Now commit and push this file to trigger the pipeline!")
    
    return filename

if __name__ == "__main__":
    generate_new_data_batch(1000)