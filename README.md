# Stock Volatility Predictor 

A Machine Learning application that forecasts stock market volatility using Random Forest classification and real-time technical analysis.

## Project Overview

This project aims to help traders identify potential risks by classifying stocks into **Low**, **Medium**, or **High** volatility categories. Unlike simple price tracking, this tool analyzes market sentiment, momentum, and macroeconomic factors (like Gold prices) to generate a probabilistic risk assessment.

I built this to explore the intersection of quantitative finance and supervised learning, specifically focusing on how technical indicators (Moving Averages, ROC) influence model decision-making.

## Tech Stack

* **Core:** Python 3.9+
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Processing:** Pandas, NumPy
* **Market Data:** Yahoo Finance API (`yfinance`)
* **Backend:** Flask (REST API)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Async/Await)

## Key Features

* **Real-Time Analysis:** Fetches live market data on-demand; no stale databases.
* **Feature Engineering:** Calculates 5-day/20-day Moving Averages, Price Range, and Rate of Change (ROC) dynamically.
* **Macro Correlation:** Includes Gold Futures (GC=F) as a feature to gauge market fear/sentiment.
* **Explainable AI:** The UI displays "Feature Contributions," showing exactly which factors (e.g., Volume vs. Trend) influenced the specific prediction.
* **Robust Pipeline:** Includes scripts for data preprocessing, model retraining, and frontend generation.

## Project Structure

```text
├── app.py                     # Main Flask application & API endpoints
├── retrain_enhanced_model.py  # ML Pipeline: Loads data -> Engineers Features -> Trains Model
├── build_frontend.py          # Generates the HTML templates
├── test_api.py                # Integration test script
├── rf_volatility_model.pkl    # Serialized Random Forest Model
└── templates/
    └── index.html             # The User Interface
```
Future Improvements
If I had more time, I would expand this project by:

Implementing an LSTM (Deep Learning) model to compare against the Random Forest.

Adding a sentiment analysis feature using news headlines via NLTK.

Dockerizing the application for easier deployment.

How to Run
1. Install Dependencies
```Bash
pip install flask pandas numpy scikit-learn yfinance joblib requests
```
2. Generate the Frontend
Run this once to build the HTML interface.

```Bash
python build_frontend.py
```
3. Start the Server
```Bash
python app.py
```
You should see: Running on http://127.0.0.1:5000

4. Access the App
Open your browser and navigate to http://localhost:5000.

5. (Optional) Run Tests
Open a second terminal window and run:

```Bash
python test_api.py
```
---

### How to Run and Verify (For You)

Here is your exact checklist to get this running and verify it works before your interview.

1.  **Setup the Frontend:**
    Run the build script to create the `templates` folder and `index.html`.
    ```bash
    python build_frontend.py
    ```
    *Output:* `✓ Successfully generated frontend at: templates/index.html`

2.  **Verify Model Existence:**
    Make sure `rf_volatility_model.pkl` is in the same folder. If you don't have it, run `python retrain_enhanced_model.py` (assuming you have the CSV data). *If you do not have the CSV data anymore, let me know, and I can write a script to generate dummy data so the training script works.*

3.  **Start the App:**
    ```bash
    python app.py
    ```
    *Output:*
    ```text
    ✓ Model loaded successfully...
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```

4.  **Verify via Browser:**
    Go to `http://localhost:5000`. Type "AAPL" and click Analyze. You should see the gauge charts animate and data populate.

5.  **Verify via Test Script:**
    Keep `app.py` running. Open a **new** terminal window and run:
    ```bash
    python test_api.py
    ```
    *Output:* `✓ Success! (Time: 0.xxs)` followed by a breakdown of the prediction data.