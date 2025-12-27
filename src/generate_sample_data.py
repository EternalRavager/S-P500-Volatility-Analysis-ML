"""
Generate sample preprocessed data for model training.
This creates synthetic stock data with realistic patterns.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating sample training data...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2 years of daily data
n_days = 500
start_date = datetime.now() - timedelta(days=n_days)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Sample tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'WMT']

all_data = []

for ticker in tickers:
    # Generate realistic price data with random walk
    base_price = np.random.uniform(50, 300)
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = base_price * np.exp(np.cumsum(returns))

    for i in range(n_days):
        # Daily OHLC with realistic spreads
        close = prices[i]
        daily_volatility = np.random.uniform(0.01, 0.03)

        high = close * (1 + np.random.uniform(0, daily_volatility))
        low = close * (1 - np.random.uniform(0, daily_volatility))
        open_price = np.random.uniform(low, high)

        # Volume (millions of shares)
        volume = np.random.uniform(1e6, 50e6)

        # Gold price (correlated with market sentiment)
        gold_close = 1800 + np.random.normal(0, 50) + (i * 0.1)  # Slight upward trend

        all_data.append({
            'Date': dates[i],
            'Ticker': ticker,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
            'Gold_Close': gold_close,
        })

# Create DataFrame
df = pd.DataFrame(all_data)

# Calculate volatility (standard deviation of returns over 20-day window)
df = df.sort_values(['Ticker', 'Date'])
df['Returns'] = df.groupby('Ticker')['Close'].pct_change()
df['Volatility'] = df.groupby('Ticker')['Returns'].transform(
    lambda x: x.rolling(window=20, min_periods=1).std()
)

# Create volatility labels (Low, Medium, High)
def label_volatility(volatility_series):
    """Classify volatility into three categories using percentiles."""
    low_threshold = volatility_series.quantile(0.33)
    high_threshold = volatility_series.quantile(0.67)

    labels = []
    for vol in volatility_series:
        if vol < low_threshold:
            labels.append(0)  # Low
        elif vol < high_threshold:
            labels.append(1)  # Medium
        else:
            labels.append(2)  # High
    return labels

df['Volatility_Label'] = df.groupby('Ticker')['Volatility'].transform(label_volatility)

# Remove rows with NaN (from rolling calculations)
df = df.dropna()

# Select final columns to match expected format
final_columns = [
    'Date', 'Ticker', 'Open', 'High', 'Low', 'Close',
    'Volume', 'Gold_Close', 'Volatility_Label'
]
df = df[final_columns]

# Save to CSV
output_file = "ml_preprocessed_data.csv"
df.to_csv(output_file, index=False)

print(f"✓ Generated {len(df)} rows of sample data")
print(f"✓ Saved to: {output_file}")
print(f"\nData summary:")
print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  - Tickers: {df['Ticker'].nunique()}")
print(f"  - Label distribution:")
for label, name in [(0, 'Low'), (1, 'Medium'), (2, 'High')]:
    count = (df['Volatility_Label'] == label).sum()
    pct = count / len(df) * 100
    print(f"    {name}: {count} ({pct:.1f}%)")

