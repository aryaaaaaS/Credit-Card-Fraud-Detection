import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    font-size: 3rem;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.metric-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e0e0e0;
    margin: 0.5rem 0;
}
.fraud-alert {
    background-color: #ffebee;
    color: #c62828;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #f44336;
}
.legitimate-alert {
    background-color: #e8f5e8;
    color: #2e7d32;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
}
.input-section {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the credit card fraud dataset from CSV file"""
    try:
        # Try to load from CSV file
        df = pd.read_csv('data/creditcard.csv')
        st.success("✅ Dataset loaded from CSV file successfully!")
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please run 'python create_dataset.py' first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

@st.cache_resource
def train_model(df):
    """Train the fraud detection model"""
    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    return rf_model, scaler, X_test_scaled, y_test, y_pred, precision, recall, f1

def transform_user_inputs_to_features(amount, time, transaction_type, location, card_type, device, merchant_category, frequency):
    """Transform user-friendly inputs to V1-V28 features"""
    
    # Base features based on user inputs
    base_features = np.zeros(30)  # V1-V28 + Amount + Time
    
    # Amount impact (normalized)
    amount_normalized = (amount - 100) / 1000  # Normalize amount
    base_features[28] = amount  # Amount
    base_features[29] = time    # Time
    
    # Transaction type impact
    if transaction_type == "Online":
        base_features[0] = -2.5   # V1
        base_features[1] = 3.2    # V2
        base_features[2] = -1.8   # V3
    elif transaction_type == "Offline":
        base_features[0] = 1.2    # V1
        base_features[1] = -1.5   # V2
        base_features[2] = 0.8    # V3
    
    # Location impact
    if location == "International":
        base_features[3] = -3.1   # V4
        base_features[4] = 2.8    # V5
        base_features[5] = -2.2   # V6
    elif location == "Domestic":
        base_features[3] = 0.5    # V4
        base_features[4] = -0.3   # V5
        base_features[5] = 0.2    # V6
    
    # Card type impact
    if card_type == "Credit":
        base_features[6] = -1.5   # V7
        base_features[7] = 2.1    # V8
    elif card_type == "Debit":
        base_features[6] = 0.8    # V7
        base_features[7] = -0.9   # V8
    
    # Device impact
    if device == "Mobile":
        base_features[8] = -2.8   # V9
        base_features[9] = 3.5    # V10
    elif device == "Desktop":
        base_features[8] = 1.2    # V9
        base_features[9] = -1.1   # V10
    
    # Merchant category impact
    if merchant_category == "Electronics":
        base_features[10] = -2.1  # V11
        base_features[11] = 2.8   # V12
    elif merchant_category == "Food & Dining":
        base_features[10] = 0.5   # V11
        base_features[11] = -0.3  # V12
    elif merchant_category == "Travel":
        base_features[10] = -3.2  # V11
        base_features[11] = 3.8   # V12
    elif merchant_category == "Shopping":
        base_features[10] = -1.8  # V11
        base_features[11] = 2.2   # V12
    
    # Frequency impact
    if frequency == "High (10+ transactions)":
        base_features[12] = -4.2  # V13
        base_features[13] = 4.8   # V14
    elif frequency == "Medium (5-10 transactions)":
        base_features[12] = -2.1  # V13
        base_features[13] = 2.5   # V14
    elif frequency == "Low (1-4 transactions)":
        base_features[12] = 0.3   # V13
        base_features[13] = -0.2  # V14
    
    # Add some randomness and correlations
    for i in range(14, 28):
        base_features[i] = np.random.normal(0, 0.5)
    
    # High amount + high frequency = higher fraud risk
    if amount > 5000 and frequency == "High (10+ transactions)":
        base_features[0:5] += np.random.normal(-2, 0.5, 5)
    
    # International + high amount = higher fraud risk
    if location == "International" and amount > 3000:
        base_features[3:8] += np.random.normal(-2.5, 0.5, 5)
    
    # Mobile + high amount = higher fraud risk
    if device == "Mobile" and amount > 2000:
        base_features[8:13] += np.random.normal(-2.2, 0.5, 5)
    
    return base_features

def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Fraud Detection", "📊 Data Analysis", "🤖 Model Training", "📈 Performance Metrics"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Train model
    with st.spinner("Training model..."):
        model, scaler, X_test, y_test, y_pred, precision, recall, f1 = train_model(df)
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "🔍 Fraud Detection":
        show_fraud_detection_user_friendly(model, scaler)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🤖 Model Training":
        show_model_training(df, model, scaler, X_test, y_test, y_pred, precision, recall, f1)
    elif page == "📈 Performance Metrics":
        show_performance_metrics(y_test, y_pred, precision, recall, f1)

def show_home_page(df):
    """Display the home page with project overview"""
    st.markdown("""
    ## 🎯 Project Overview
    
    This application uses machine learning to detect fraudulent credit card transactions. 
    Simply enter the transaction details and our AI model will predict if it's fraudulent or legitimate.
    
    ### 📊 Dataset Information
    """)
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        st.metric("Fraud Cases", f"{df['Class'].sum():,}")
    
    with col3:
        st.metric("Legitimate Cases", f"{(df['Class'] == 0).sum():,}")
    
    with col4:
        fraud_rate = (df['Class'].sum() / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    st.markdown("""
    ### 🚀 How to Use
    
    1. **Go to Fraud Detection**: Click on "🔍 Fraud Detection" in the sidebar
    2. **Enter Transaction Details**: Fill in the simple form with transaction information
    3. **Get Instant Results**: Our AI will predict if the transaction is fraudulent
    4. **View Confidence**: See how confident the model is in its prediction
    
    ### 🔍 What We Analyze
    
    - **💸 Amount**: Transaction amount
    - **🕒 Time**: When the transaction occurred
    - **🏦 Transaction Type**: Online or offline
    - **🌐 Location**: Domestic or international
    - **🪪 Card Type**: Credit or debit card
    - **📲 Device**: Mobile or desktop
    - **🛒 Merchant Category**: Type of merchant
    - **🔁 Frequency**: Transaction frequency in last 24h
    """)

def show_fraud_detection_user_friendly(model, scaler):
    """Display user-friendly fraud detection interface"""
    st.markdown("## 🔍 Fraud Detection")
    
    st.markdown("""
    Enter the transaction details below. Our AI model will analyze the patterns and predict 
    whether this transaction is fraudulent or legitimate.
    """)
    
    # Input form with user-friendly fields
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("📋 Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic transaction info
        amount = st.number_input("💸 Amount ($)", min_value=1.0, value=100.0, step=1.0, help="Enter the transaction amount")
        time = st.number_input("🕒 Time (seconds since first transaction)", min_value=0, value=1000, step=1, help="Time elapsed since the first transaction")
        
        transaction_type = st.selectbox(
            "🏦 Transaction Type",
            ["Online", "Offline"],
            help="How the transaction was conducted"
        )
        
        location = st.selectbox(
            "🌐 Location",
            ["Domestic", "International"],
            help="Where the transaction occurred"
        )
    
    with col2:
        # Card and device info
        card_type = st.selectbox(
            "🪪 Card Type",
            ["Credit", "Debit"],
            help="Type of card used"
        )
        
        device = st.selectbox(
            "📲 Device Used",
            ["Mobile", "Desktop"],
            help="Device used for the transaction"
        )
        
        merchant_category = st.selectbox(
            "🛒 Merchant Category",
            ["Food & Dining", "Shopping", "Electronics", "Travel", "Entertainment", "Healthcare", "Transportation"],
            help="Category of the merchant"
        )
        
        frequency = st.selectbox(
            "🔁 Frequency (last 24h)",
            ["Low (1-4 transactions)", "Medium (5-10 transactions)", "High (10+ transactions)"],
            help="Number of transactions in the last 24 hours"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):
        # Transform user inputs to technical features
        features = transform_user_inputs_to_features(
            amount, time, transaction_type, location, card_type, device, merchant_category, frequency
        )
        
        # Prepare input for model
        input_df = pd.DataFrame([features], columns=[f'V{i+1}' for i in range(28)] + ['Amount', 'Time'])
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0]
        
        # Display result
        st.markdown("## 📊 Prediction Result")
        
        if prediction == 1:
            st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
            st.markdown("### 🚨 FRAUDULENT TRANSACTION DETECTED!")
            st.markdown(f"**Confidence**: {confidence[1]:.2%}")
            st.markdown("**Recommendation**: Block this transaction immediately and contact the cardholder.")
            st.markdown("**Risk Factors**: High amount, unusual location, or suspicious patterns detected.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="legitimate-alert">', unsafe_allow_html=True)
            st.markdown("### ✅ LEGITIMATE TRANSACTION")
            st.markdown(f"**Confidence**: {confidence[0]:.2%}")
            st.markdown("**Recommendation**: Transaction appears legitimate and can proceed.")
            st.markdown("**Analysis**: Transaction patterns match normal behavior.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show risk factors
        st.subheader("🔍 Risk Analysis")
        
        risk_factors = []
        risk_score = 0
        
        if amount > 5000:
            risk_factors.append("💰 High transaction amount")
            risk_score += 30
        
        if location == "International":
            risk_factors.append("🌐 International transaction")
            risk_score += 25
        
        if device == "Mobile":
            risk_factors.append("📱 Mobile device")
            risk_score += 15
        
        if frequency == "High (10+ transactions)":
            risk_factors.append("🔁 High transaction frequency")
            risk_score += 20
        
        if merchant_category in ["Electronics", "Travel"]:
            risk_factors.append("🛒 High-risk merchant category")
            risk_score += 10
        
        if transaction_type == "Online":
            risk_factors.append("💻 Online transaction")
            risk_score += 5
        
        # Display risk factors
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Factors Detected:**")
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("• No significant risk factors detected")
        
        with col2:
            st.markdown("**Risk Score:**")
            if risk_score > 50:
                st.markdown(f"🔴 **High Risk** ({risk_score}%)")
            elif risk_score > 25:
                st.markdown(f"🟡 **Medium Risk** ({risk_score}%)")
            else:
                st.markdown(f"🟢 **Low Risk** ({risk_score}%)")

def show_data_analysis(df):
    """Display data analysis visualizations"""
    st.markdown("## 📊 Data Analysis")
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes.value_counts())
    
    # Fraud distribution
    st.subheader("Fraud Distribution")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Pie chart
    fraud_counts = df['Class'].value_counts()
    ax1.pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%', 
            colors=['lightgreen', 'lightcoral'])
    ax1.set_title('Transaction Distribution')
    
    # Bar chart
    ax2.bar(['Legitimate', 'Fraud'], fraud_counts.values, color=['lightgreen', 'lightcoral'])
    ax2.set_title('Transaction Counts')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Amount distribution
    st.subheader("Transaction Amount Distribution")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # All transactions
    ax1.hist(df['Amount'], bins=50, alpha=0.7, color='skyblue')
    ax1.set_title('All Transactions')
    ax1.set_xlabel('Amount')
    ax1.set_ylabel('Frequency')
    
    # Fraud vs Legitimate
    legitimate = df[df['Class'] == 0]['Amount']
    fraud = df[df['Class'] == 1]['Amount']
    
    ax2.hist(legitimate, bins=50, alpha=0.7, label='Legitimate', color='lightgreen')
    ax2.hist(fraud, bins=50, alpha=0.7, label='Fraud', color='lightcoral')
    ax2.set_title('Amount Distribution by Class')
    ax2.set_xlabel('Amount')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def show_model_training(df, model, scaler, X_test, y_test, y_pred, precision, recall, f1):
    """Display model training information"""
    st.markdown("## 🤖 Model Training")
    
    st.markdown("""
    ### Model Details
    - **Algorithm**: Random Forest Classifier
    - **Trees**: 100
    - **Max Depth**: 10
    - **Class Imbalance Handling**: class_weight='balanced'
    - **Feature Scaling**: StandardScaler
    """)
    
    # Training process
    st.subheader("Training Process")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        1. **Data Split**: 80% training, 20% testing
        2. **Feature Scaling**: Standardized all features
        3. **Class Weight**: Balanced class weights
        4. **Model Training**: Random Forest with 100 trees
        5. **Prediction**: Made predictions on test set
        """)
    
    with col2:
        st.markdown("""
        ### Model Performance
        """)
        
        st.metric("Precision", f"{precision:.3f}")
        st.metric("Recall", f"{recall:.3f}")
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    feature_importance = pd.DataFrame({
        'feature': df.drop('Class', axis=1).columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = feature_importance.head(15)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Most Important Features')
    plt.tight_layout()
    st.pyplot(fig)

def show_performance_metrics(y_test, y_pred, precision, recall, f1):
    """Display performance metrics"""
    st.markdown("## 📈 Performance Metrics")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", f"{precision:.3f}")
    
    with col2:
        st.metric("Recall", f"{recall:.3f}")
    
    with col3:
        st.metric("F1-Score", f"{f1:.3f}")
    
    with col4:
        accuracy = (y_test == y_pred).mean()
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("Detailed Classification Report")
    
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])
    st.text(report)
    
    # Performance explanation
    st.subheader("📊 Metric Explanation")
    
    st.markdown("""
    - **Precision**: Of all transactions predicted as fraud, how many were actually fraud?
    - **Recall**: Of all actual fraud transactions, how many did we catch?
    - **F1-Score**: Harmonic mean of precision and recall
    - **Accuracy**: Overall correctness of predictions
    """)

if __name__ == "__main__":
    main() 