import os
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify

# Project paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "volatility_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
TEMPLATE_DIR = BASE_DIR / "templates"

# Initialize Flask app
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Features expected by the Random Forest model
FEATURE_NAMES = [
    "Open", "High", "Low", "Close", "Volume", "Gold_Close",
    "MA_5", "MA_20", "ROC", "Price_Range", "Volume_Change"
]

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    FEATURE_IMPORTANCE = model.feature_importances_
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠ Warning: Model not found at {MODEL_PATH}. Please run the training script first.")
    model = None
    FEATURE_IMPORTANCE = np.zeros(len(FEATURE_NAMES))

# Load metrics from JSON file instead of hard-coding
try:
    with open(METRICS_PATH, 'r') as f:
        MODEL_INFO = json.load(f)
        # Add additional fields for backward compatibility
        MODEL_INFO["algorithm"] = MODEL_INFO.get("model_type", "Random Forest Classifier")
        MODEL_INFO["n_estimators"] = MODEL_INFO.get("n_estimators", 100)
        MODEL_INFO["features_count"] = len(FEATURE_NAMES)
        print(f"✓ Loaded model metrics: Accuracy={MODEL_INFO['accuracy']:.4f}, F1={MODEL_INFO['f1_score']:.4f}")
except FileNotFoundError:
    print(f"⚠ model_metrics.json not found, using fallback values")
    MODEL_INFO = {
        "algorithm": "Random Forest Classifier",
        "n_estimators": 100,
        "features_count": len(FEATURE_NAMES),
        "accuracy": 0.5567,
        "f1_score": 0.5472,
        "note": "Using default metrics - retrain model to update"
    }


def calculate_technical_indicators(stock_data):
    """
    Computes technical indicators used as features for the ML model.
    I chose these specific indicators to capture both trend (MA) and momentum (ROC).
    """
    df = stock_data.copy()
    df = df.sort_values("Date")

    # Moving Averages to smooth out price noise
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    # Rate of Change (Momentum) - how fast did price change over 5 days?
    df["ROC"] = df["Close"].pct_change(periods=5) * 100

    # Intraday Volatility
    df["Price_Range"] = df["High"] - df["Low"]

    # Volume Spike Detection
    df["Volume_Change"] = df["Volume"].pct_change() * 100

    latest = df.iloc[-1]

    # Handle potential NaNs (e.g., if a stock is new and lacks history)
    return {
        "ma_5": float(latest["MA_5"]) if not pd.isna(latest["MA_5"]) else None,
        "ma_20": float(latest["MA_20"]) if not pd.isna(latest["MA_20"]) else None,
        "roc": float(latest["ROC"]) if not pd.isna(latest["ROC"]) else None,
        "price_range": float(latest["Price_Range"]),
        "volume_change": float(latest["Volume_Change"]) if not pd.isna(latest["Volume_Change"]) else None,
        # Simple heuristic to determine trend direction for the UI
        "trend": "Bullish" if latest["MA_5"] > latest["MA_20"] else "Bearish"
    }


def get_latest_data(ticker):
    """
    Fetches real-time data from Yahoo Finance.
    Returns a dictionary of features formatted for the model.
    """
    try:
        # Fetch 90 days of data to ensure we have enough for 20-day Moving Averages
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period="90d")

        if stock_data.empty:
            return None, "No data found for this ticker. Please check the symbol."

        stock_data.reset_index(inplace=True)

        # Apply the same feature engineering used during training
        # Note: We duplicate this logic here to ensure the live data matches the training data schema exactly.
        stock_data["MA_5"] = stock_data["Close"].rolling(window=5).mean()
        stock_data["MA_20"] = stock_data["Close"].rolling(window=20).mean()
        stock_data["ROC"] = stock_data["Close"].pct_change(periods=5) * 100
        stock_data["Price_Range"] = stock_data["High"] - stock_data["Low"]
        stock_data["Volume_Change"] = stock_data["Volume"].pct_change() * 100

        latest_stock = stock_data.iloc[-1]

        # Fetch Gold prices as a macro-economic correlation feature
        gold = yf.Ticker("GC=F")
        gold_data = gold.history(period="90d")
        gold_data.reset_index(inplace=True)

        # Match gold price to stock date, fallback to latest available if mismatch
        latest_date = latest_stock["Date"]
        gold_row = gold_data[gold_data["Date"] == latest_date]

        if gold_row.empty:
            gold_value = gold_data.iloc[-1]["Close"]
        else:
            gold_value = gold_row.iloc[0]["Close"]

        # Calculate historical volatility for comparison
        # Annualized volatility = Standard Deviation of Returns * Sqrt(252 trading days)
        returns = stock_data["Close"].pct_change()
        historical_volatility = returns.std() * np.sqrt(252)

        return {
            "ticker": ticker,
            "date": latest_date.strftime("%Y-%m-%d"),
            "open": float(latest_stock["Open"]),
            "high": float(latest_stock["High"]),
            "low": float(latest_stock["Low"]),
            "close": float(latest_stock["Close"]),
            "volume": float(latest_stock["Volume"]),
            "gold_close": float(gold_value),
            # Use .get with default 0 to handle NaNs safely
            "ma_5": float(latest_stock["MA_5"]) if not pd.isna(latest_stock["MA_5"]) else 0,
            "ma_20": float(latest_stock["MA_20"]) if not pd.isna(latest_stock["MA_20"]) else 0,
            "roc": float(latest_stock["ROC"]) if not pd.isna(latest_stock["ROC"]) else 0,
            "price_range": float(latest_stock["Price_Range"]),
            "volume_change": float(latest_stock["Volume_Change"]) if not pd.isna(latest_stock["Volume_Change"]) else 0,
            "technical_indicators": calculate_technical_indicators(stock_data),
            "historical_volatility": float(historical_volatility),
            "price_change_pct": float((latest_stock["Close"] - latest_stock["Open"]) / latest_stock["Open"] * 100)
        }, None

    except Exception as e:
        return None, str(e)


def predict_volatility(data):
    """
    Runs the Random Forest inference.
    Returns prediction label and confidence scores.
    """
    if model is None:
        raise ValueError("Model is not loaded. Check server logs.")

    # Structure input exactly as the model expects
    feature_values = [[
        data["open"], data["high"], data["low"],
        data["close"], data["volume"], data["gold_close"],
        data["ma_5"], data["ma_20"],
        data["roc"], data["price_range"],
        data["volume_change"]
    ]]

    feature_vector = pd.DataFrame(feature_values, columns=FEATURE_NAMES)

    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]

    # Mapping numeric predictions back to human-readable labels
    labels = {0: "Low", 1: "Medium", 2: "High"}

    return {
        "prediction": labels[prediction],
        "prediction_code": int(prediction),
        "confidence": float(max(probabilities)),
        "probabilities": {
            "low": float(probabilities[0]),
            "medium": float(probabilities[1]),
            "high": float(probabilities[2])
        }
    }


def get_feature_contributions(data):
    """
    Maps feature importance scores to the current data values.
    Helps explain *why* the model made a decision.
    """
    feature_values = [
        data["open"], data["high"], data["low"], data["close"],
        data["volume"], data["gold_close"],
        data["ma_5"], data["ma_20"],
        data["roc"], data["price_range"],
        data["volume_change"]
    ]

    contributions = []
    for name, value, importance in zip(FEATURE_NAMES, feature_values, FEATURE_IMPORTANCE):
        contributions.append({
            "name": name,
            "value": value,
            "importance": float(importance),
            "contribution": float(importance * 100)
        })

    # Sort by importance so the most impactful features appear first
    contributions.sort(key=lambda x: x["importance"], reverse=True)
    return contributions


@app.route("/")
def home():
    return render_template("index.html", model_info=MODEL_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        ticker = request.json.get("ticker", "").strip().upper()

        if not ticker:
            return jsonify({"error": "Please enter a ticker symbol"}), 400

        data, error = get_latest_data(ticker)

        if error:
            return jsonify({"error": error}), 400

        result = predict_volatility(data)
        feature_contributions = get_feature_contributions(data)

        # User-friendly interpretations
        interpretations = {
            "Low": "Stable movement expected. Suitable for conservative strategies.",
            "Medium": "Moderate fluctuations likely. Standard risk level.",
            "High": "Significant swings expected. High risk, potential for high reward."
        }

        response = {
            "success": True,
            "ticker": data["ticker"],
            "date": data["date"],
            "stock_data": {
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"],
                "price_change_pct": data["price_change_pct"]
            },
            "gold_price": data["gold_close"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "interpretation": interpretations[result["prediction"]],
            "technical_indicators": data["technical_indicators"],
            "historical_volatility": data["historical_volatility"],
            "feature_contributions": feature_contributions,
            "model_info": MODEL_INFO
        }
        return jsonify(response)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": "Internal server error during analysis."}), 500


if __name__ == "__main__":
    # Debug mode is on for development, but would be off in production
    app.run(debug=True, port=5000)