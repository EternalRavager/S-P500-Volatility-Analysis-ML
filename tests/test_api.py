import requests
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_TICKER = "AAPL"

print("=" * 60)
print("Integration Test: Volatility Prediction API")
print("=" * 60)


def test_prediction_endpoint():
    print(f"Target: {BASE_URL}/predict")
    print(f"Ticker: {TEST_TICKER}")
    print("-" * 30)

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"ticker": TEST_TICKER},
            timeout=10
        )
        duration = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success! (Time: {duration:.2f}s)")

            # Verify payload structure
            print("\n[Response Payload Analysis]")
            print(f"   Prediction: {data['prediction']} (Confidence: {data['confidence']:.2f})")
            print(f"   Date:       {data['date']}")
            print(f"   Gold Price: ${data['gold_price']:.2f}")

            # Check for feature contributions
            if 'feature_contributions' in data:
                top_feature = data['feature_contributions'][0]
                print(f"   Top Driver: {top_feature['name']} ({top_feature['contribution']:.1f}%)")

            print("\n✓ API is functioning correctly.")
            return True
        else:
            print(f"✗ Failed. Status Code: {response.status_code}")
            print(f"Error Message: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("\n✗ Connection Error!")
        print("   Is the Flask app running? Open a separate terminal and run 'python app.py'")
        return False
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_prediction_endpoint()
    if not success:
        sys.exit(1)