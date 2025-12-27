"""
Entry point for Stock Volatility Predictor
Run: python run.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("Stock Volatility Predictor - Starting Server")
    print("=" * 60)
    print("Open your browser at: http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
