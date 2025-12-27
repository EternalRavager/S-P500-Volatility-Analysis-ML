# Stock Volatility Predictor

Machine Learning application for predicting stock market volatility using Random Forest and real-time technical indicators.

## Features
- ✅ Real-time stock data from Yahoo Finance
- ✅ 11 technical indicators (MA-5, MA-20, ROC, Price Range, Volume Change)
- ✅ Gold price correlation analysis
- ✅ Feature importance visualization
- ✅ Confidence scoring for predictions
- ✅ Dynamic model metrics (accuracy, F1-score)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
python run.py
```

### 3. Access Web Interface
- Open browser: **http://localhost:5000**
- Enter stock ticker (e.g., AAPL, GOOGL, TSLA)
- View volatility prediction with confidence scores

## Project Structure
```
Stock_Volatility_ML/
├── src/                          # Source code
│   ├── app.py                    # Flask web application
│   ├── retrain_enhanced_model.py # Model training script
│   └── generate_sample_data.py   # Sample data generator
├── models/                       # Trained models
│   ├── volatility_model.pkl      # Random Forest model
│   └── model_metrics.json        # Performance metrics
├── data/                         # Training datasets
│   └── ml_preprocessed_data.csv
├── templates/                    # HTML templates
│   └── index.html
├── tests/                        # Test scripts
│   ├── test_api.py
│   ├── validate_all.py
│   └── verify_training.py
├── docs/                         # Documentation
│   └── README.md
├── run.py                        # Application entry point
└── requirements.txt              # Python dependencies
```

## Model Information
- **Algorithm**: Random Forest Classifier (100 trees)
- **Features**: 11 technical indicators + gold correlation
- **Performance**: Metrics loaded from `models/model_metrics.json`
  - Accuracy: ~55.7%
  - F1-Score: ~54.7%
- **Training Data**: 1000+ samples with engineered features

## Testing

### Validate All Components
```bash
python tests/validate_all.py
```

### Test API Endpoints
```bash
python tests/test_api.py
```

### Verify Model Training
```bash
python tests/verify_training.py
```

## Retraining Model

To retrain with updated data:

```bash
python src/retrain_enhanced_model.py
```

This will:
1. Generate/load training data from `data/ml_preprocessed_data.csv`
2. Train Random Forest model with 100 estimators
3. Save model to `models/volatility_model.pkl`
4. Save performance metrics to `models/model_metrics.json`
5. Display feature importance rankings

## API Endpoints

### `GET /`
Main web interface with interactive UI

### `POST /predict`
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
  "confidence": 0.87,
  "feature_importance": {...},
  "current_price": 178.25,
  "date": "2024-12-27"
}
```

## Feature Engineering

Input features calculated from 90 days of historical data:

| Feature | Description |
|---------|-------------|
| Open, High, Low, Close | Daily OHLC prices |
| Volume | Trading volume |
| Gold_Close | Gold futures price (GC=F) |
| MA_5 | 5-day moving average |
| MA_20 | 20-day moving average |
| ROC | Rate of Change (momentum) |
| Price_Range | High - Low (intraday range) |
| Volume_Change | % change in volume |

## Technical Details

### Model Training Process
1. Load preprocessed data (`ml_preprocessed_data.csv`)
2. Split into 80% train, 20% test
3. Train Random Forest with 100 trees
4. Evaluate on test set
5. Save model and metrics

### Prediction Pipeline
1. Fetch 90 days historical data via yfinance
2. Calculate 11 technical indicators
3. Normalize features
4. Predict volatility class
5. Return confidence scores

## Future Enhancements
- [ ] LSTM deep learning model
- [ ] News sentiment analysis integration
- [ ] Multi-stock portfolio analysis
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Real-time WebSocket updates
- [ ] Backtesting framework

## Requirements
- Python 3.8+
- Flask 3.0.0
- scikit-learn 1.3.0
- pandas 2.1.0
- numpy 1.24.3
- yfinance 0.2.28
- joblib 1.3.2

## Development Workflow

### Adding New Features
1. Update feature engineering in `retrain_enhanced_model.py`
2. Retrain model to generate new metrics
3. Update `app.py` if new features require different preprocessing
4. Run validation tests

### Running Tests Before Deployment
```bash
# Validate all components
python tests/validate_all.py

# Test API functionality
python tests/test_api.py

# Verify model predictions
python tests/verify_training.py
```

## Troubleshooting

### Model Not Found Error
```bash
# Retrain the model
python src/retrain_enhanced_model.py
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Port Already in Use
```python
# Edit run.py and change port
app.run(debug=True, host='0.0.0.0', port=5001)
```

## License
Educational project for machine learning demonstration.

## Author
Developed for academic purposes - Stock Market Volatility Prediction using ML

## Acknowledgments
- Yahoo Finance for real-time stock data
- scikit-learn for machine learning framework
- Flask for web framework
