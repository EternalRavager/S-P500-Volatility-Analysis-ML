import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Configuration
# Using relative paths to ensure this runs on any machine
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "ml_preprocessed_data.csv"
MODEL_OUTPUT = BASE_DIR / "models" / "volatility_model.pkl"
METRICS_OUTPUT = BASE_DIR / "models" / "model_metrics.json"

print("=" * 70)
print("Stock Volatility Model - Retraining Pipeline")
print("=" * 70)


def load_and_engineer_features(filepath):
    """
    Loads raw data and generates technical indicators.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find data file at: {filepath}")

    print("\n[1/5] Loading and preprocessing data...")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    print(f"   ✓ Loaded {len(df)} rows")

    # Feature Engineering
    # I'm adding moving averages to help the model detect trends
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # Rate of Change (Momentum)
    df['ROC'] = df['Close'].pct_change(periods=5) * 100

    # Intraday Volatility (High-Low spread)
    df['Price_Range'] = df['High'] - df['Low']

    # Volume Change (Liquidity shocks)
    df['Volume_Change'] = df['Volume'].pct_change() * 100

    # Cleaning: Rolling windows create NaNs at the start of the dataset
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Outlier Removal
    # Clipping values to 3 standard deviations to prevent extreme market events 
    # (like flash crashes) from skewing the model's baseline logic.
    for col in ['ROC', 'Volume_Change']:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)

    print(f"   ✓ Feature engineering complete. Final shape: {df.shape}")
    return df


def train_model(df):
    """
    Trains the Random Forest Classifier.
    """
    print("\n[2/5] Preparing training sets...")

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Gold_Close',
        'MA_5', 'MA_20', 'ROC', 'Price_Range', 'Volume_Change'
    ]

    X = df[feature_cols]
    y = df['Volatility_Label']  # Assuming this column exists in your preprocessed CSV

    # Splitting 80/20 standard split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   ✓ Training samples: {len(X_train)}")
    print(f"   ✓ Testing samples: {len(X_test)}")

    print("\n[3/5] Training Random Forest...")
    # Using 100 trees for a balance between accuracy and inference speed
    clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
    clf.fit(X_train, y_train)
    print("   ✓ Training complete")

    return clf, X_test, y_test, feature_cols


def evaluate_model(clf, X_test, y_test, feature_cols):
    """
    Prints performance metrics and feature importance.
    """
    print("\n[4/5] Evaluating performance...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"   ✓ Accuracy: {acc:.4f}")
    print(f"   ✓ F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    print("\nFeature Importance:")
    # Mapping importance scores to feature names to see what drives predictions
    importances = dict(zip(feature_cols, clf.feature_importances_))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, score) in enumerate(sorted_features, 1):
        print(f"   {rank}. {name:15s} : {score:.4f}")
    
    return acc, f1, importances


if __name__ == "__main__":
    try:
        # Pipeline execution
        data = load_and_engineer_features(DATA_FILE)
        model, X_test, y_test, features = train_model(data)
        accuracy, f1, feature_importance_dict = evaluate_model(model, X_test, y_test, features)

        # Create metrics dictionary
        model_metrics = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "feature_importance": {k: float(v) for k, v in feature_importance_dict.items()},
            "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "Random Forest",
            "n_estimators": 100,
            "training_samples": len(data)
        }

        # Save metrics
        print(f"\n[5/5] Saving model and metrics...")
        with open(METRICS_OUTPUT, 'w') as f:
            json.dump(model_metrics, f, indent=2)
        print(f"   ✓ Metrics saved to {METRICS_OUTPUT}")
        
        # Save model
        joblib.dump(model, MODEL_OUTPUT)
        print(f"   ✓ Model saved to {MODEL_OUTPUT}")
        print("\n   ✓ Done! Pipeline finished successfully.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Make sure 'ml_preprocessed_data.csv' is in the data/ folder.")