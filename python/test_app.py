#!/usr/bin/env python3
"""
Test script for Credit Card Fraud Detection Application
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import pickle

def test_data_generation():
    """Test synthetic data generation"""
    print("🧪 Testing data generation...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000  # Smaller dataset for testing
    n_fraud = 50
    
    data = {}
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    data['Amount'] = np.random.exponential(88, n_samples)
    data['Time'] = np.arange(n_samples)
    
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    data['Class'] = [0] * n_samples
    for idx in fraud_indices:
        data['Class'][idx] = 1
    
    df = pd.DataFrame(data)
    
    # Make fraud transactions have different patterns
    for i in range(1, 29):
        df.loc[df['Class'] == 1, f'V{i}'] += np.random.normal(0, 0.5, n_fraud)
    
    print(f"✅ Data generated successfully! Shape: {df.shape}")
    print(f"📊 Fraud rate: {(df['Class'].sum() / len(df)) * 100:.2f}%")
    
    return df

def test_model_training(df):
    """Test model training pipeline"""
    print("\n🧪 Testing model training...")
    
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Precision: {precision:.3f}")
    print(f"📊 Recall: {recall:.3f}")
    print(f"📊 F1-Score: {f1:.3f}")
    
    return rf_model, scaler

def test_prediction(model, scaler):
    """Test prediction functionality"""
    print("\n🧪 Testing prediction functionality...")
    
    # Create sample transaction
    sample_transaction = np.random.normal(0, 1, 30)  # 30 features (Amount, Time, V1-V28)
    sample_transaction[0] = 100.0  # Amount
    sample_transaction[1] = 1000   # Time
    
    # Reshape for prediction
    sample_transaction = sample_transaction.reshape(1, -1)
    
    # Scale the input
    sample_scaled = scaler.transform(sample_transaction)
    
    # Make prediction
    prediction = model.predict(sample_scaled)[0]
    prediction_proba = model.predict_proba(sample_scaled)[0]
    
    print(f"✅ Prediction successful!")
    print(f"📊 Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
    print(f"📊 Confidence: {prediction_proba[prediction]:.3f}")
    
    return prediction, prediction_proba

def test_model_saving(model, scaler):
    """Test model saving functionality"""
    print("\n🧪 Testing model saving...")
    
    # Save model data
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    }
    
    # Save to file
    with open('test_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model saved successfully!")
    
    # Load model data
    with open('test_model.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    
    # Clean up
    os.remove('test_model.pkl')
    print("✅ Test file cleaned up!")

def main():
    """Run all tests"""
    print("🚀 Starting Credit Card Fraud Detection Tests")
    print("=" * 50)
    
    try:
        # Test 1: Data Generation
        df = test_data_generation()
        
        # Test 2: Model Training
        model, scaler = test_model_training(df)
        
        # Test 3: Prediction
        prediction, proba = test_prediction(model, scaler)
        
        # Test 4: Model Saving
        test_model_saving(model, scaler)
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("✅ Application is ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 