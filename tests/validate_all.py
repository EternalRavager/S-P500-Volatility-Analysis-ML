#!/usr/bin/env python
"""
One-command validation script - Tests everything at once
Run this to verify the entire project is working
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def check_file(filepath, description):
    """Check if a file exists and report its size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
        print(f"   ✓ {description}: {size_str}")
        return True
    else:
        print(f"   ✗ {description}: MISSING")
        return False

def test_import(module_name):
    """Test if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"   ✓ {module_name}")
        return True
    except ImportError:
        print(f"   ✗ {module_name} - NOT INSTALLED")
        return False

def test_model():
    """Test if model can be loaded and used"""
    try:
        import joblib
        model = joblib.load('rf_volatility_model.pkl')
        print(f"   ✓ Model loads successfully")
        print(f"      - Features: {model.n_features_in_}")
        print(f"      - Classes: {model.n_classes_}")
        print(f"      - Estimators: {model.n_estimators}")
        return True
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        return False

def main():
    print("=" * 70)
    print("STOCK VOLATILITY PREDICTOR - VALIDATION CHECKLIST")
    print("=" * 70)

    all_ok = True

    # Check Python version
    print("\n[1/5] Python Environment:")
    print(f"   ✓ Python {sys.version.split()[0]}")

    # Check dependencies
    print("\n[2/5] Required Dependencies:")
    deps = ['pandas', 'numpy', 'sklearn', 'joblib', 'flask', 'yfinance', 'requests']
    for dep in deps:
        if not test_import(dep):
            all_ok = False

    # Check files
    print("\n[3/5] Required Files:")
    files_to_check = [
        ('app.py', 'Flask Application'),
        ('retrain_enhanced_model.py', 'Training Script'),
        ('rf_volatility_model.pkl', 'Trained Model'),
        ('ml_preprocessed_data.csv', 'Training Data'),
        ('templates/index.html', 'Web Interface'),
    ]

    for filepath, desc in files_to_check:
        if not check_file(filepath, desc):
            all_ok = False

    # Test model
    print("\n[4/5] Model Validation:")
    if not test_model():
        all_ok = False

    # Summary
    print("\n[5/5] Quick Test:")
    try:
        import joblib
        import numpy as np

        model = joblib.load('rf_volatility_model.pkl')
        # Create dummy input
        X_dummy = np.random.random((1, 11))
        prediction = model.predict(X_dummy)[0]
        proba = model.predict_proba(X_dummy)[0]

        labels = ['Low', 'Medium', 'High']
        print(f"   ✓ Test prediction: {labels[prediction]} ({proba[prediction]*100:.1f}% confidence)")
    except Exception as e:
        print(f"   ✗ Prediction test failed: {e}")
        all_ok = False

    # Final verdict
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ ALL CHECKS PASSED - System is ready!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run: python app.py")
        print("  2. Open: http://localhost:5000")
        print("  3. Test with ticker: AAPL")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Please review errors above")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("  • Missing dependencies? Run: pip install -r requirements.txt")
        print("  • Missing model? Run: python retrain_enhanced_model.py")
        print("  • Missing data? Run: python generate_sample_data.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())

