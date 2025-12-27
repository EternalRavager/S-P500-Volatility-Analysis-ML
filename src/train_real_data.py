"""
Stock Volatility ML - Real Data Training Script
================================================

This script trains the volatility prediction model using REAL stock market data
fetched from Yahoo Finance. It aligns with the approach used in ML.ipynb.

Data Source: 
    - Stock List: Fortune 500 / S&P 500 companies from slickcharts.com (matching ML.ipynb)
    - Price Data: Yahoo Finance (yfinance)
    - Period: 5 years of historical data
    - Features: 11 technical indicators

Usage:
    python src/train_real_data.py

For testing with synthetic data instead, use:
    python src/generate_sample_data.py

Author: Abhishek
Last Updated: December 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_auc_score
)
import joblib
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (Matching ML.ipynb)
# =============================================================================

# Training configuration
YEARS_OF_DATA = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Number of stocks to use (set to None to use all Fortune 500)
# Using top 50 for faster training, set to None for all ~500 stocks
MAX_STOCKS = 50

# Feature columns (must match app.py)
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Gold_Close',
    'MA_5', 'MA_20', 'ROC', 'Price_Range', 'Volume_Change'
]


# =============================================================================
# FORTUNE 500 DATA FETCHING (Matching ML.ipynb)
# =============================================================================

def get_fortune500_tickers():
    """
    Fetch Fortune 500 / S&P 500 company tickers from slickcharts.com
    This matches the approach used in ML.ipynb
    
    Returns:
        List of stock ticker symbols
    """
    print("   Fetching S&P 500 list from slickcharts.com...")
    
    url = "https://www.slickcharts.com/sp500"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})
        
        if table is None:
            print("   ⚠ Could not find table, using fallback list...")
            return get_fallback_tickers()
        
        tickers = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                ticker = cols[2].text.strip()
                # Clean ticker (some have dots like BRK.B)
                ticker = ticker.replace('.', '-')
                tickers.append(ticker)
        
        if tickers:
            print(f"   ✓ Found {len(tickers)} S&P 500 companies")
            return tickers
        else:
            print("   ⚠ No tickers found, using fallback list...")
            return get_fallback_tickers()
            
    except Exception as e:
        print(f"   ⚠ Error fetching from slickcharts: {e}")
        print("   Using fallback list...")
        return get_fallback_tickers()


def get_fallback_tickers():
    """
    Fallback list of major stocks if web scraping fails
    Includes diverse sectors for robust training (86 stocks)
    """
    return [
        # Technology (15)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
        'ORCL', 'ACN', 'IBM', 'INTC', 'AMD',
        # Finance (12)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
        'PNC', 'TFC',
        # Healthcare (10)
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
        # Consumer Discretionary (10)
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG',
        # Consumer Staples (8)
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL',
        # Energy (6)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',
        # Industrials (8)
        'UPS', 'HON', 'UNP', 'BA', 'CAT', 'GE', 'RTX', 'LMT',
        # Communication (5)
        'DIS', 'NFLX', 'CMCSA', 'VZ', 'T',
        # Utilities (4)
        'NEE', 'DUK', 'SO', 'D',
        # Real Estate (4)
        'AMT', 'PLD', 'CCI', 'EQIX',
        # Materials (4)
        'LIN', 'APD', 'SHW', 'FCX'
    ]


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        stock = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
        if not stock.empty and len(stock) > 100:  # Need at least 100 days
            # Flatten multi-level columns from yfinance
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.droplevel(1)
            stock['Ticker'] = ticker
            return stock
    except Exception as e:
        pass  # Silently fail for individual stocks
    return None


def fetch_gold_data(start_date, end_date):
    """
    Fetch Gold futures data for market sentiment correlation
    Matches ML.ipynb approach using GC=F ticker
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with Gold closing prices
    """
    try:
        gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
        if not gold.empty:
            # Flatten multi-level columns from yfinance
            if isinstance(gold.columns, pd.MultiIndex):
                gold.columns = gold.columns.droplevel(1)
            return gold[['Close']].rename(columns={'Close': 'Gold_Close'})
    except Exception as e:
        print(f"   ⚠ Error fetching Gold data: {e}")
    return None


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS (Matching ML.ipynb)
# =============================================================================

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for the dataset
    These match the features used in ML.ipynb and app.py
    
    Features calculated:
        - MA_5: 5-day Simple Moving Average
        - MA_20: 20-day Simple Moving Average
        - ROC: Rate of Change (momentum)
        - Price_Range: High - Low (intraday volatility)
        - Volume_Change: Percentage change in volume
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional technical indicator columns
    """
    df = df.copy()
    
    # Moving Averages (matching ML.ipynb)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Rate of Change - 10-day momentum (matching ML.ipynb)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Price Range - intraday volatility proxy
    df['Price_Range'] = df['High'] - df['Low']
    
    # Volume Change - trading activity change
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    
    # Drop rows with NaN values from rolling calculations
    df = df.dropna()
    
    return df


def calculate_volatility_labels(df, window=20):
    """
    Calculate volatility labels based on rolling standard deviation of returns
    Modified to create 3-class labels (Low/Medium/High) to match app.py expectations
    
    Args:
        df: DataFrame with price data
        window: Rolling window for volatility calculation (default: 20 days)
    
    Returns:
        DataFrame with Volatility_Label column (0=Low, 1=Medium, 2=High)
    """
    df = df.copy()
    
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate rolling volatility (standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    
    # Create 3-class labels using percentiles
    # Low: bottom 33%, Medium: middle 33%, High: top 33%
    low_threshold = df['Volatility'].quantile(0.33)
    high_threshold = df['Volatility'].quantile(0.67)
    
    df['Volatility_Label'] = 1  # Default to Medium
    df.loc[df['Volatility'] <= low_threshold, 'Volatility_Label'] = 0  # Low
    df.loc[df['Volatility'] > high_threshold, 'Volatility_Label'] = 2  # High
    
    # Drop NaN values
    df = df.dropna()
    
    return df


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_model(X_train, y_train):
    """
    Train Random Forest classifier with parameters matching ML.ipynb
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, X_full, y_full):
    """
    Comprehensive model evaluation with cross-validation
    Matches evaluation approach in ML.ipynb
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_full: Full dataset features (for CV)
        y_full: Full dataset labels (for CV)
    
    Returns:
        Dictionary with all evaluation metrics
    """
    # Basic predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    try:
        # For multi-class, use ovr (one-vs-rest) strategy
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.5
    
    # Cross-validation (3-fold for speed) - simplified from ML.ipynb
    print("   Running 3-fold cross-validation (faster)...")
    cv_scores = cross_val_score(model, X_full, y_full, cv=3, scoring='accuracy', n_jobs=-1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics, y_pred


# =============================================================================
# MAIN TRAINING PIPELINE (Matching ML.ipynb)
# =============================================================================

def main():
    """
    Main training pipeline - matches ML.ipynb workflow
    
    Steps:
        1. Fetch Fortune 500 tickers from slickcharts.com
        2. Download 5 years of historical data from Yahoo Finance
        3. Fetch Gold data for correlation
        4. Calculate technical indicators
        5. Calculate volatility labels
        6. Train Random Forest model
        7. Evaluate with 5-fold cross-validation
        8. Save model and metrics
    """
    print("=" * 70)
    print("STOCK VOLATILITY ML - REAL DATA TRAINING")
    print("Aligned with ML.ipynb methodology")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Date range (5 years - matching ML.ipynb)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
    
    # =========================================================================
    # STEP 1: Get Fortune 500 Tickers (Matching ML.ipynb)
    # =========================================================================
    print(f"\n📋 Step 1/7: Getting S&P 500 / Fortune 500 company list...")
    
    all_tickers = get_fortune500_tickers()
    
    # Limit stocks if specified
    if MAX_STOCKS and MAX_STOCKS < len(all_tickers):
        tickers = all_tickers[:MAX_STOCKS]
        print(f"   Using top {MAX_STOCKS} stocks for training")
        print(f"   (Set MAX_STOCKS = None in script to use all {len(all_tickers)} stocks)")
    else:
        tickers = all_tickers
        print(f"   Using all {len(tickers)} stocks")
    
    print(f"\n📊 Configuration:")
    print(f"   • Stocks: {len(tickers)} companies")
    print(f"   • Period: {start_date.date()} to {end_date.date()} ({YEARS_OF_DATA} years)")
    print(f"   • Test Size: {TEST_SIZE * 100}%")
    print(f"   • Model: Random Forest ({N_ESTIMATORS} trees, max_depth={MAX_DEPTH})")
    
    # =========================================================================
    # STEP 2: Fetch Stock Data (Matching ML.ipynb)
    # =========================================================================
    print(f"\n📥 Step 2/7: Fetching stock data from Yahoo Finance...")
    print(f"   This may take a few minutes for {len(tickers)} stocks...")
    
    all_stock_data = []
    successful = 0
    failed = 0
    
    fetch_start = time.time()
    
    for i, ticker in enumerate(tickers, 1):
        # Progress indicator every 10 stocks
        if i % 10 == 0 or i == len(tickers):
            elapsed = time.time() - fetch_start
            print(f"   Progress: {i}/{len(tickers)} stocks ({successful} successful, {failed} failed) - {elapsed:.1f}s")
        
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None:
            all_stock_data.append(data)
            successful += 1
        else:
            failed += 1
    
    fetch_time = time.time() - fetch_start
    
    if not all_stock_data:
        print("\n❌ ERROR: Failed to fetch any stock data!")
        return
    
    combined_df = pd.concat(all_stock_data, ignore_index=False)
    print(f"\n   ✓ Fetched {successful}/{len(tickers)} stocks in {fetch_time:.1f}s")
    print(f"   ✓ Combined dataset: {len(combined_df):,} rows")
    
    # =========================================================================
    # STEP 3: Fetch Gold Data (Matching ML.ipynb)
    # =========================================================================
    print(f"\n🥇 Step 3/7: Fetching Gold futures data (GC=F)...")
    
    gold_data = fetch_gold_data(start_date, end_date)
    if gold_data is not None:
        print(f"   ✓ Gold data: {len(gold_data):,} days")
        combined_df = combined_df.join(gold_data, how='left')
        combined_df['Gold_Close'] = combined_df['Gold_Close'].fillna(method='ffill').fillna(method='bfill')
    else:
        print("   ⚠ Using fallback Gold values")
        combined_df['Gold_Close'] = 1900.0
    
    # =========================================================================
    # STEP 4: Calculate Technical Indicators (Matching ML.ipynb)
    # =========================================================================
    print(f"\n⚙️  Step 4/7: Calculating technical indicators...")
    
    combined_df = calculate_technical_indicators(combined_df)
    print(f"   ✓ Features: MA_5, MA_20, ROC, Price_Range, Volume_Change")
    print(f"   ✓ Dataset after feature engineering: {len(combined_df):,} rows")
    
    # =========================================================================
    # STEP 5: Calculate Volatility Labels (Matching ML.ipynb)
    # =========================================================================
    print(f"\n📈 Step 5/7: Calculating volatility labels...")
    
    combined_df = calculate_volatility_labels(combined_df)
    
    high_vol = combined_df['Volatility_Label'].sum()
    low_vol = len(combined_df) - high_vol
    print(f"   ✓ High Volatility: {high_vol:,} ({high_vol/len(combined_df)*100:.1f}%)")
    print(f"   ✓ Low Volatility: {low_vol:,} ({low_vol/len(combined_df)*100:.1f}%)")
    print(f"   ✓ Final dataset: {len(combined_df):,} samples")
    
    # =========================================================================
    # STEP 6: Train Model (Matching ML.ipynb)
    # =========================================================================
    print(f"\n🌲 Step 6/7: Training Random Forest model...")
    
    # Prepare features and labels
    X = combined_df[FEATURE_COLUMNS].values
    y = combined_df['Volatility_Label'].values
    
    # Split data (80/20 - matching ML.ipynb)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Test samples: {len(X_test):,}")
    
    # Train
    train_start = time.time()
    model = train_model(X_train, y_train)
    train_time = time.time() - train_start
    print(f"   ✓ Model trained in {train_time:.2f} seconds")
    
    # =========================================================================
    # STEP 7: Evaluate Model (Matching ML.ipynb)
    # =========================================================================
    print(f"\n📊 Step 7/7: Evaluating model performance...")
    
    metrics, y_pred = evaluate_model(model, X_test, y_test, X, y)
    
    # Feature importance
    feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    
    print(f"\n{'=' * 70}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'=' * 70}")
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   • Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   • F1-Score:     {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"   • ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"   • CV Accuracy:  {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Low Volatility', 'Medium Volatility', 'High Volatility']))
    
    print(f"\n🔢 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                    Predicted")
    print(f"                    Low    Medium  High")
    print(f"   Actual Low      {cm[0][0]:5d}  {cm[0][1]:6d} {cm[0][2]:5d}")
    print(f"          Medium   {cm[1][0]:5d}  {cm[1][1]:6d} {cm[1][2]:5d}")
    print(f"          High     {cm[2][0]:5d}  {cm[2][1]:6d} {cm[2][2]:5d}")
    
    print(f"\n⭐ Top 5 Feature Importances:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(sorted_features[:5], 1):
        print(f"   {i}. {feat:15s}: {imp:.4f} ({imp*100:.2f}%)")
    
    # =========================================================================
    # SAVE MODEL AND METRICS
    # =========================================================================
    print(f"\n💾 Saving model and metrics...")
    
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH = BASE_DIR / "models" / "volatility_model.pkl"
    METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
    
    # Get list of successful tickers
    successful_tickers = list(combined_df['Ticker'].unique()) if 'Ticker' in combined_df.columns else tickers[:successful]
    
    # Prepare comprehensive metrics
    total_time = time.time() - start_time
    
    full_metrics = {
        "accuracy": metrics['accuracy'],
        "f1_score": metrics['f1_score'],
        "roc_auc": metrics['roc_auc'],
        "cv_accuracy_mean": metrics['cv_accuracy_mean'],
        "cv_accuracy_std": metrics['cv_accuracy_std'],
        "feature_importance": feature_importance,
        "confusion_matrix": metrics['confusion_matrix'],
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "Random Forest",
        "model_params": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": RANDOM_STATE
        },
        "data_info": {
            "source": "Yahoo Finance (Real Data) - S&P 500 from slickcharts.com",
            "period_years": YEARS_OF_DATA,
            "stocks_attempted": len(tickers),
            "stocks_successful": successful,
            "stocks_used": successful_tickers[:20] if len(successful_tickers) > 20 else successful_tickers,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "total_samples": int(len(X))
        },
        "training_time_seconds": round(train_time, 2),
        "fetch_time_seconds": round(fetch_time, 2),
        "total_time_seconds": round(total_time, 2),
        "notes": "Model trained on 5 years of real S&P 500 stock data (matching ML.ipynb). Predictions use 90-day window for speed."
    }
    
    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    print(f"   ✓ Metrics saved to: {METRICS_PATH}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"   ✓ Model saved to: {MODEL_PATH}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"   • Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   • Stocks used: {successful} companies")
    print(f"   • Total samples: {len(X):,}")
    print(f"   • Model accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"\n   📁 Files saved:")
    print(f"      • Model: {MODEL_PATH}")
    print(f"      • Metrics: {METRICS_PATH}")
    print(f"\n   🚀 Next steps:")
    print(f"      1. Run 'python run.py' to start the application")
    print(f"      2. Open http://localhost:5000 in your browser")
    print(f"      3. Enter a stock ticker to get volatility predictions")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
