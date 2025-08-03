import pandas as pd
import numpy as np
import os

def create_creditcard_dataset():
    """Create a synthetic credit card fraud dataset similar to the real one"""
    print("🔧 Creating synthetic credit card fraud dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Dataset parameters (similar to real dataset)
    n_samples = 284807
    n_fraud = 492
    
    print(f"📊 Generating {n_samples:,} transactions with {n_fraud} fraud cases...")
    
    # Create synthetic data
    data = {}
    
    # Generate V1-V28 features (PCA components)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Add Amount and Time features
    data['Amount'] = np.random.exponential(88, n_samples)
    data['Time'] = np.arange(n_samples)
    
    # Create target variable (fraud = 1, legitimate = 0)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    data['Class'] = [0] * n_samples
    for idx in fraud_indices:
        data['Class'][idx] = 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Make fraud transactions have different patterns
    for i in range(1, 29):
        df.loc[df['Class'] == 1, f'V{i}'] += np.random.normal(0, 0.5, n_fraud)
    
    # Save to CSV
    df.to_csv("data/creditcard.csv", index=False)
    
    print("✅ Synthetic dataset created and saved!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📊 Fraud rate: {(df['Class'].sum() / len(df) * 100):.2f}%")
    print(f"📊 Total transactions: {len(df):,}")
    print(f"📊 Fraud cases: {df['Class'].sum():,}")
    print(f"📊 Legitimate cases: {(df['Class'] == 0).sum():,}")
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create dataset
    df = create_creditcard_dataset()
    
    print("\n🎉 Dataset ready! You can now use it in your fraud detection app.")
    print("📁 File location: data/creditcard.csv")
    
    # Show sample data
    print("\n📈 Sample data:")
    print(df.head()) 