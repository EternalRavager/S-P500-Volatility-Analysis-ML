# Stock Volatility Predictor 📈

A Machine Learning application that predicts stock market volatility using Random Forest classification and real-time technical analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Model Training](#model-training)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project predicts whether a stock will experience **High** or **Low** volatility based on historical price data and technical indicators. The model is trained on **5 years of real market data** from **S&P 500 / Fortune 500 companies** (fetched from slickcharts.com) and uses 11 carefully engineered features.

### Key Highlights

- **Fortune 500 Training Data**: Model trained on real stock data from S&P 500 companies
- **5 Years of History**: Comprehensive training covering multiple market cycles
- **Fast Predictions**: Real-time predictions using 90 days of recent data for optimal speed
- **11 Technical Indicators**: Including moving averages, momentum, and volume analysis
- **Cross-Validated**: Model performance verified using 5-fold cross-validation

---

## ✨ Features

- ✅ Real-time stock data fetching via Yahoo Finance
- ✅ Fortune 500 / S&P 500 company data for training (from slickcharts.com)
- ✅ 11 technical indicators (MA-5, MA-20, ROC, Price Range, Volume Change, etc.)
- ✅ Gold price correlation analysis for market sentiment
- ✅ Feature importance visualization
- ✅ Confidence scoring for predictions
- ✅ Dynamic model metrics (loaded from JSON, not hard-coded)
- ✅ Professional web interface

---

## 📁 Project Structure

```
Stock_Volatility_ML/
│
├── src/                              # Source code
│   ├── app.py                        # Flask web application
│   ├── train_real_data.py            # Training with Fortune 500 data (RECOMMENDED)
│   ├── generate_sample_data.py       # Generate synthetic data (testing only)
│   └── retrain_enhanced_model.py     # Legacy training script
│
├── models/                           # Trained models
│   ├── volatility_model.pkl          # Random Forest model (trained on Fortune 500)
│   └── model_metrics.json            # Performance metrics & feature importance
│
├── data/                             # Datasets
│   └── processed/
│       └── ml_preprocessed_data.csv  # Preprocessed training data
│
├── templates/                        # HTML templates
│   └── index.html                    # Web interface
│
├── tests/                            # Test scripts
│   ├── test_api.py                   # API endpoint tests
│   ├── validate_all.py               # Comprehensive validation
│   └── verify_training.py            # Training verification
│
├── docs/                             # Documentation
│   └── README.md                     # This file
│
├── ML.ipynb                          # Jupyter notebook (exploration & analysis)
├── Viva_Prep_for_Prof_QnA.md         # Q&A preparation notes
├── run.py                            # Application entry point
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git ignore rules
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python run.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

### 4. Make a Prediction

1. Enter a stock ticker (e.g., `AAPL`, `GOOGL`, `TSLA`)
2. Click **"Predict Volatility"**
3. View the prediction with confidence scores and feature importance

---

## 🎓 Model Training

### Two Training Approaches

| Script | Data Source | Purpose | When to Use |
|--------|-------------|---------|-------------|
| `src/train_real_data.py` | Fortune 500 / S&P 500 (Real) | Production model | **RECOMMENDED** for best accuracy |
| `src/generate_sample_data.py` | Synthetic | Testing/Demo | Quick testing without API calls |

### Training with Real Fortune 500 Data (Recommended)

```bash
python src/train_real_data.py
```

**What it does:**
1. Fetches S&P 500 / Fortune 500 ticker list from slickcharts.com
2. Downloads 5 years of historical data for each stock via Yahoo Finance
3. Fetches Gold futures data (GC=F) for market correlation
4. Calculates 11 technical indicators (MA-5, MA-20, ROC, etc.)
5. Calculates volatility labels based on rolling standard deviation
6. Trains Random Forest classifier with 100 trees
7. Evaluates with 5-fold cross-validation
8. Saves model and comprehensive metrics to `models/` directory

**Time Required:** ~5-10 minutes (depends on number of stocks and internet speed)

**Configuration Options** (in `train_real_data.py`):
```python
MAX_STOCKS = 50      # Number of stocks to use (None = all ~500 stocks)
YEARS_OF_DATA = 5    # Years of historical data
N_ESTIMATORS = 100   # Number of Random Forest trees
MAX_DEPTH = 10       # Maximum tree depth
```

### Training with Synthetic Data (Testing Only)

```bash
python src/generate_sample_data.py
python src/retrain_enhanced_model.py
```

⚠️ **Note:** Synthetic data is for testing purposes only. Use `train_real_data.py` for production models with better accuracy.

---

## 📊 Model Performance

### Current Model Statistics

These metrics are from training on **5 years of real Fortune 500 stock data** (  results):

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 64.5% | Correct predictions on test set |
| **F1-Score** | 63.8% | Harmonic mean of precision/recall |
| **ROC-AUC** | 0.71 | Area under ROC curve |
| **CV Accuracy** | 63.2% ± 1.8% | 5-fold cross-validation |

### Confusion Matrix

```
              Predicted
              Low    High
Actual Low    1245    312
       High    298   1189
```

### Top 5 Feature Importances

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | MA_20 | 14.95% | 20-day moving average |
| 2 | Close | 12.30% | Closing price |
| 3 | MA_5 | 10.45% | 5-day moving average |
| 4 | ROC | 9.95% | Rate of Change (momentum) |
| 5 | Price_Range | 9.20% | High - Low (intraday range) |

### Training Data vs. Prediction Data

| Aspect | Training | Prediction |
|--------|----------|------------|
| **Data Source** | Fortune 500 / S&P 500 companies | Any valid ticker |
| **Time Period** | 5 years | 90 days |
| **Purpose** | Learn volatility patterns | Real-time inference |
| **Speed** | 5-10 minutes | 2-3 seconds |

**Why 90-Day Data for Predictions?**

The model is **trained on 5 years of data** for robust learning, but **predictions use only 90 days** because:

1. **Speed**: Fetching 90 days takes 2-3 seconds vs. 30+ seconds for 5 years
2. **Relevance**: Recent data better reflects current market conditions
3. **Feature Calculation**: Technical indicators only need ~20 days of history
4. **User Experience**: Fast predictions improve usability

The model learns patterns from 5 years but applies them to the latest 90-day snapshot.

---

## 🔧 Technical Details

### Data Pipeline

```
1. Fetch Fortune 500 tickers from slickcharts.com
         ↓
2. Download 5 years of OHLCV data from Yahoo Finance
         ↓
3. Fetch Gold futures (GC=F) for market sentiment
         ↓
4. Calculate technical indicators (MA, ROC, etc.)
         ↓
5. Calculate volatility labels (rolling std dev)
         ↓
6. Train Random Forest (100 trees, max_depth=10)
         ↓
7. Evaluate with 5-fold cross-validation
         ↓
8. Save model and metrics
```

### Feature Engineering

All features are calculated from OHLCV (Open, High, Low, Close, Volume) data:

| Feature | Description | Calculation |
|---------|-------------|-------------|
| Open | Opening price | Raw value |
| High | Highest price | Raw value |
| Low | Lowest price | Raw value |
| Close | Closing price | Raw value |
| Volume | Trading volume | Raw value |
| Gold_Close | Gold futures price | From GC=F ticker |
| MA_5 | 5-day moving average | `Close.rolling(5).mean()` |
| MA_20 | 20-day moving average | `Close.rolling(20).mean()` |
| ROC | Rate of Change | `(Close - Close[10]) / Close[10] * 100` |
| Price_Range | Intraday range | `High - Low` |
| Volume_Change | Volume momentum | `Volume.pct_change() * 100` |

### Volatility Label Calculation

```python
# Daily returns
Returns = Close.pct_change()

# 20-day rolling volatility (standard deviation of returns)
Volatility = Returns.rolling(20).std()

# Binary classification based on median threshold
Label = 1 if Volatility > median(Volatility) else 0
```

### Model Architecture

```
Random Forest Classifier
├── n_estimators: 100 trees
├── max_depth: 10
├── min_samples_split: 10
├── min_samples_leaf: 5
├── class_weight: balanced
└── random_state: 42
```

---

## 📡 API Reference

### `GET /`

Returns the web interface.

### `POST /predict`

Make a volatility prediction for a stock.

**Request:**
```json
{
  "ticker": "AAPL"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "prediction": "Low Volatility",
  "confidence": 0.73,
  "probability": {
    "low": 0.73,
    "high": 0.27
  },
  "current_price": 178.25,
  "price_change": 1.25,
  "feature_importance": {
    "MA_20": 0.1495,
    "Close": 0.1230,
    ...
  },
  "model_accuracy": 0.645,
  "date": "2024-12-27"
}
```

### `GET /model-info`

Get model metadata and performance metrics.

**Response:**
```json
{
  "accuracy": 0.645,
  "f1_score": 0.638,
  "trained_date": "2024-12-27 10:30:00",
  "model_type": "Random Forest",
  "data_source": "Yahoo Finance (Real Data) - Fortune 500"
}
```

---

## 🧪 Testing

### Run All Tests

```bash
python tests/validate_all.py
```

### Test API Endpoints

```bash
# Start the server first
python run.py

# In another terminal
python tests/test_api.py
```

### Verify Model Training

```bash
python tests/verify_training.py
```

---

## 🔮 Future Improvements

- [ ] **LSTM Deep Learning Model** - Capture temporal patterns
- [ ] **News Sentiment Analysis** - Incorporate market sentiment
- [ ] **Multi-Stock Portfolio** - Analyze portfolio volatility
- [ ] **Docker Deployment** - Containerized deployment
- [ ] **CI/CD Pipeline** - Automated testing and deployment
- [ ] **Backtesting Framework** - Historical performance analysis
- [ ] **Real-time WebSocket** - Live price updates
- [ ] **Hyperparameter Tuning** - Optimize model parameters with GridSearch

---

## 📦 Requirements

```
flask==3.0.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
yfinance==0.2.28
requests==2.31.0
beautifulsoup4==4.12.0
```

---

## 📝 Notes for Future Reference

### Key Files to Understand

1. **`src/app.py`** - Main Flask application with prediction logic
2. **`src/train_real_data.py`** - Primary training script using Fortune 500 data (matches ML.ipynb)
3. **`models/model_metrics.json`** - All model statistics (dynamically loaded)
4. **`ML.ipynb`** - Detailed exploration, analysis, and model comparison

### Important Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Random Forest** | Best accuracy among tested models (RF, Logistic Regression, SVM) |
| **11 Features** | Optimal balance between accuracy and overfitting |
| **Gold Correlation** | Gold is a market fear indicator (flight to safety) |
| **90-Day Prediction** | Speed vs. accuracy trade-off for real-time use |
| **Fortune 500 Training** | Diverse sectors ensure robust pattern learning |
| **5-Year History** | Covers multiple market cycles (bull/bear markets) |

### Retraining the Model

If market conditions change significantly or you want to update with latest data:

```bash
# Recommended: Train on Fortune 500 real data
python src/train_real_data.py

# Check new metrics
cat models/model_metrics.json

# Restart the app to load new model
python run.py
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not loading | Check `models/volatility_model.pkl` exists |
| Metrics not showing | Verify `models/model_metrics.json` is valid JSON |
| Training fails | Check internet connection for Yahoo Finance |
| Prediction timeout | Increase yfinance timeout in `app.py` |

---

## 👤 Author

Abhishek

---

## 📄 License

This project is for educational and demonstration purposes.

---

*Last Updated: December 2025*
