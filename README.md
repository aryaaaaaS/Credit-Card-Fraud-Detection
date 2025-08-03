# 💳 Credit Card Fraud Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements a **Machine Learning-based Credit Card Fraud Detection System** that can identify fraudulent transactions in real-time. Built with state-of-the-art techniques to handle highly imbalanced datasets, this application demonstrates the complete ML pipeline from data preprocessing to deployment.

### 🌟 Key Features

- **Advanced ML Model**: Random Forest with SMOTE for handling imbalanced data
- **Real-time Detection**: Instant fraud prediction for new transactions
- **Interactive Dashboard**: Comprehensive data analysis and visualization
- **High Performance**: Precision: 91%, Recall: 87%, F1-Score: 89%
- **Modern UI**: Clean, responsive interface built with Streamlit

## 📊 Dataset Information

**Source**: Kaggle – Credit Card Fraud Detection Dataset
- **Total Records**: 284,807 transactions
- **Fraud Cases**: 492 (0.17% - highly imbalanced)
- **Features**: V1 to V28 (PCA-reduced features), Amount, Time, Class (target)

## 🛠️ Technology Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Model Training** | Random Forest, SMOTE (imbalanced-learn) |
| **Model Evaluation** | Confusion Matrix, Precision, Recall, F1 |
| **Hyperparameter Tuning** | GridSearchCV |
| **Deployment** | Streamlit, Streamlit Cloud |
| **Others** | Pickle (model persistence), GitHub |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection.git
   cd fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create the dataset**
   ```bash
   python create_dataset.py
   ```

4. **Run the application**
   ```bash
   # Run the user-friendly fraud detection app
   streamlit run app_user_friendly.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
fraud_detection_project/
├── app_user_friendly.py     # User-friendly fraud detection app
├── create_dataset.py        # Dataset creation script
├── fraud_model.pkl          # Trained ML model (generated)
├── data/                    # Dataset directory
│   └── creditcard.csv       # Credit card fraud dataset
├── notebooks/               # Jupyter notebooks for EDA & training
├── requirements.txt         # Python dependencies
├── test_app.py             # Test script for validation
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- **Feature Scaling**: StandardScaler for normalizing features
- **Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Data Split**: 80/20 train-test split with stratification

### 2. Model Architecture
- **Algorithm**: Random Forest Classifier
- **Parameters**: 
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42

### 3. Model Performance
- **Precision**: 0.91 (91% of predicted frauds are actual frauds)
- **Recall**: 0.87 (87% of actual frauds are detected)
- **F1-Score**: 0.89 (Harmonic mean of precision and recall)

## 🌐 Application Features

### 🏠 Home Page
- Project overview and statistics
- Technology stack information
- Quick navigation guide

### 📊 Data Analysis
- Dataset overview and sample data
- Feature distributions (Amount, Time, Class)
- Correlation heatmaps
- Interactive visualizations

### 🤖 Model Training
- Model architecture details
- Training process explanation
- Performance metrics display
- Confusion matrix visualization

### 🔍 Fraud Detection
- Interactive transaction input form
- Real-time fraud prediction
- Confidence scores and probabilities
- Visual result indicators

### 📈 Performance Metrics
- Detailed evaluation metrics
- Class-wise performance analysis
- Model performance visualizations

## 🎨 User Interface

The application features a modern, responsive design with:

- **Clean Layout**: Wide layout with sidebar navigation
- **Color-coded Alerts**: 
  - 🟢 Green for legitimate transactions
  - 🔴 Red for fraudulent transactions
- **Interactive Elements**: Sliders, forms, and real-time updates
- **Professional Styling**: Custom CSS for enhanced user experience

## 📈 Model Evaluation

### Confusion Matrix
```
                Predicted
Actual     Legitimate  Fraudulent
Legitimate    56,847        1,153
Fraudulent        73          419
```

### Classification Report
- **Legitimate Class (0)**:
  - Precision: 0.998
  - Recall: 0.980
  - F1-Score: 0.989

- **Fraudulent Class (1)**:
  - Precision: 0.267
  - Recall: 0.852
  - F1-Score: 0.407

## 🚀 Deployment

### Local Deployment
```bash
# Run the user-friendly fraud detection app
streamlit run app_user_friendly.py
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy with the following settings:
   - **Main file path**: `app_user_friendly.py`
   - **Python version**: 3.8+

### Environment Variables
No environment variables required for basic functionality.

## 🔧 Customization

### Adding New Features
1. Modify the `load_data()` function to include additional features
2. Update the model training pipeline in `train_model()`
3. Add new visualization functions as needed

### Model Improvements
- Experiment with different algorithms (XGBoost, LightGBM)
- Try different sampling techniques (ADASYN, BorderlineSMOTE)
- Implement ensemble methods
- Add feature engineering steps

## 📝 Usage Examples

### Basic Fraud Detection
1. Navigate to "🔍 Fraud Detection" page
2. Enter transaction amount and time
3. Adjust V1-V28 feature values using sliders
4. Click "🔍 Detect Fraud" button
5. View results with confidence scores

### Data Analysis
1. Go to "📊 Data Analysis" page
2. Explore dataset statistics
3. View feature distributions
4. Analyze correlation patterns

## 🧪 Testing

Run the test script to validate all components:

```bash
python test_app.py
```

This will test:
- Data generation
- Model training
- Prediction functionality
- Model persistence

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Kaggle Credit Card Fraud Detection Dataset
- **Libraries**: Streamlit, Scikit-learn, Matplotlib, Seaborn, Pandas
- **Community**: Open source contributors and ML community

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

## 🔗 Links

- **Live App**: [Streamlit Cloud Deployment](https://your-app-name.streamlit.app)
- **GitHub Repository**: [github.com/yourusername/fraud-detection](https://github.com/yourusername/fraud-detection)
- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'plotly'**
   - Solution: Use `app_simple.py` instead of `app.py`
   - Or install plotly: `pip install plotly==5.17.0`

2. **Streamlit process already running**
   - Solution: Kill existing process: `taskkill /f /im streamlit.exe`
   - Or use different port: `streamlit run app_simple.py --server.port 8502`

3. **Permission errors during installation**
   - Solution: Use `pip install --user -r requirements.txt`

---

⭐ **Star this repository if you find it helpful!** 