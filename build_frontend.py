import os

# Output configuration
OUTPUT_DIR = "templates"
OUTPUT_FILE = "index.html"

# HTML Template Content
# I'm generating this programmatically so I can easily inject version numbers
# or styles later without manually editing the HTML file every time.
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Volatility | ML Portfolio</title>
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #34495e;
            --accent: #e67e22;
            --light: #ecf0f1;
            --success: #27ae60;
            --danger: #c0392b;
            --warning: #f39c12;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #f4f6f7;
            color: var(--primary);
            line-height: 1.6;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }

        /* Header Section */
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid #ddd;
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5rem; margin-bottom: 10px; color: var(--primary); }
        .subtitle { font-size: 1.1rem; color: #7f8c8d; font-style: italic; }

        /* Card Layout */
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) { .dashboard { grid-template-columns: 1fr; } }

        .card {
            background: white;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .full-width { grid-column: 1 / -1; }

        h2 { margin-bottom: 20px; font-size: 1.4rem; border-bottom: 2px solid var(--light); padding-bottom: 10px; }

        /* Input Controls */
        .input-group { display: flex; gap: 10px; }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 2px solid #bdc3c7;
            border-radius: 6px;
            font-size: 1rem;
        }
        button {
            padding: 12px 24px;
            background: var(--secondary);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }
        button:hover { background: var(--primary); }
        button:disabled { background: #95a5a6; cursor: not-allowed; }

        /* Examples */
        .tags { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px; }
        .tag {
            background: var(--light);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9rem;
            cursor: pointer;
            border: 1px solid #bdc3c7;
            transition: all 0.2s;
        }
        .tag:hover { background: var(--secondary); color: white; border-color: var(--secondary); }

        /* Results & Indicators */
        .prediction-box {
            text-align: center;
            padding: 30px;
            border-radius: 8px;
            background: var(--light);
            border: 2px solid transparent;
        }
        .prediction-box.low { background: #d5f4e6; border-color: var(--success); color: var(--success); }
        .prediction-box.medium { background: #fef5e7; border-color: var(--warning); color: var(--warning); }
        .prediction-box.high { background: #fadbd8; border-color: var(--danger); color: var(--danger); }

        .result-value { font-size: 3rem; font-weight: 800; margin: 10px 0; }

        /* Grid for key-values */
        .kv-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 15px;
        }
        .kv-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid var(--secondary);
        }
        .kv-label { font-size: 0.8rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
        .kv-val { font-size: 1.1rem; font-weight: bold; color: var(--primary); }

        /* Progress Bars */
        .bar-container { margin-bottom: 15px; }
        .bar-header { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 5px; }
        .bar-bg { background: #ecf0f1; height: 10px; border-radius: 5px; overflow: hidden; }
        .bar-fill { height: 100%; transition: width 1s ease; }

        /* Loading Spinner */
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #f3f3f3;
            border-top: 4px solid var(--secondary); border-radius: 50%;
            animation: spin 1s linear infinite; margin: 0 auto 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .hidden { display: none; }
        .error-msg { 
            background: #fadbd8; color: #c0392b; padding: 15px; 
            border-radius: 6px; margin-top: 20px; display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Stock Volatility Predictor</h1>
            <p class="subtitle">Machine Learning Analysis using Random Forest & Technical Indicators</p>
        </header>

        <div class="card full-width">
            <h2>Market Analysis</h2>
            <div class="input-group">
                <input type="text" id="tickerInput" placeholder="Enter Ticker (e.g., AAPL, NVDA, TSLA)" onkeyup="if(event.key==='Enter') runAnalysis()">
                <button id="analyzeBtn" onclick="runAnalysis()">Analyze Volatility</button>
            </div>

            <div class="tags">
                <span class="tag" onclick="setTicker('AAPL')">AAPL</span>
                <span class="tag" onclick="setTicker('MSFT')">MSFT</span>
                <span class="tag" onclick="setTicker('GOOGL')">GOOGL</span>
                <span class="tag" onclick="setTicker('TSLA')">TSLA</span>
                <span class="tag" onclick="setTicker('NVDA')">NVDA</span>
                <span class="tag" onclick="setTicker('AMZN')">AMZN</span>
            </div>

            <div id="errorBox" class="error-msg"></div>
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Fetching real-time data & computing indicators...</p>
            </div>
        </div>

        <div id="resultsArea" class="hidden">
            <div class="dashboard">
                <div>
                    <div class="card">
                        <h2>Prediction</h2>
                        <div id="predBox" class="prediction-box">
                            <div>Predicted Volatility Level</div>
                            <div id="predValue" class="result-value">-</div>
                            <div id="confValue" style="font-weight: 600">-</div>
                        </div>
                        <div id="interpretation" style="margin-top: 15px; font-style: italic; color: #555;"></div>
                    </div>

                    <div class="card">
                        <h2>Model Confidence</h2>
                        <div id="probBars"></div>
                    </div>
                </div>

                <div>
                    <div class="card">
                        <h2>Market Data (Live)</h2>
                        <div class="kv-grid">
                            <div class="kv-item"><div class="kv-label">Price</div><div class="kv-val" id="price">-</div></div>
                            <div class="kv-item"><div class="kv-label">Change</div><div class="kv-val" id="change">-</div></div>
                            <div class="kv-item"><div class="kv-label">Volume</div><div class="kv-val" id="volume">-</div></div>
                            <div class="kv-item"><div class="kv-label">Gold Price</div><div class="kv-val" id="gold">-</div></div>
                        </div>
                    </div>

                    <div class="card">
                        <h2>Technical Indicators</h2>
                        <div class="kv-grid">
                            <div class="kv-item"><div class="kv-label">MA (5-Day)</div><div class="kv-val" id="ma5">-</div></div>
                            <div class="kv-item"><div class="kv-label">MA (20-Day)</div><div class="kv-val" id="ma20">-</div></div>
                            <div class="kv-item"><div class="kv-label">R.O.C.</div><div class="kv-val" id="roc">-</div></div>
                            <div class="kv-item"><div class="kv-label">Price Range</div><div class="kv-val" id="range">-</div></div>
                        </div>
                    </div>

                    <div class="card">
                        <h2>Feature Importance</h2>
                        <p style="font-size: 0.85rem; margin-bottom: 10px; color: #666;">Top factors driving this specific prediction:</p>
                        <div id="featImp"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setTicker(t) {
            document.getElementById('tickerInput').value = t;
            runAnalysis();
        }

        async function runAnalysis() {
            const ticker = document.getElementById('tickerInput').value.trim();
            if (!ticker) return;

            // UI State Management
            const btn = document.getElementById('analyzeBtn');
            const load = document.getElementById('loading');
            const err = document.getElementById('errorBox');
            const res = document.getElementById('resultsArea');

            btn.disabled = true;
            load.style.display = 'block';
            err.style.display = 'none';
            res.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker: ticker })
                });

                const data = await response.json();

                if (!response.ok) throw new Error(data.error || 'Prediction failed');

                renderResults(data);
                res.style.display = 'block';

            } catch (e) {
                err.textContent = "Error: " + e.message;
                err.style.display = 'block';
            } finally {
                btn.disabled = false;
                load.style.display = 'none';
            }
        }

        function renderResults(data) {
            // 1. Prediction Box
            const box = document.getElementById('predBox');
            box.className = 'prediction-box ' + data.prediction.toLowerCase();
            document.getElementById('predValue').textContent = data.prediction;
            document.getElementById('confValue').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            document.getElementById('interpretation').textContent = data.interpretation;

            // 2. Probabilities
            const probs = data.probabilities;
            const colors = { low: '#27ae60', medium: '#f39c12', high: '#c0392b' };
            let probHtml = '';
            ['Low', 'Medium', 'High'].forEach(level => {
                const key = level.toLowerCase();
                const val = (probs[key] * 100).toFixed(1);
                probHtml += `
                    <div class="bar-container">
                        <div class="bar-header"><span>${level}</span><span>${val}%</span></div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width:${val}%; background:${colors[key]}"></div>
                        </div>
                    </div>`;
            });
            document.getElementById('probBars').innerHTML = probHtml;

            // 3. Market Data
            document.getElementById('price').textContent = '$' + data.stock_data.close.toFixed(2);
            document.getElementById('change').textContent = data.stock_data.price_change_pct.toFixed(2) + '%';
            document.getElementById('volume').textContent = (data.stock_data.volume / 1000000).toFixed(1) + 'M';
            document.getElementById('gold').textContent = '$' + data.gold_price.toFixed(0);

            // 4. Tech Indicators
            const tech = data.technical_indicators;
            document.getElementById('ma5').textContent = tech.ma_5 ? '$' + tech.ma_5.toFixed(2) : 'N/A';
            document.getElementById('ma20').textContent = tech.ma_20 ? '$' + tech.ma_20.toFixed(2) : 'N/A';
            document.getElementById('roc').textContent = tech.roc ? tech.roc.toFixed(2) : 'N/A';
            document.getElementById('range').textContent = '$' + tech.price_range.toFixed(2);

            // 5. Feature Importance (Top 4)
            const topFeats = data.feature_contributions.slice(0, 4);
            document.getElementById('featImp').innerHTML = topFeats.map(f => `
                <div style="display:flex; justify-content:space-between; margin-bottom:5px; font-size:0.9rem;">
                    <span>${f.name}</span>
                    <span style="font-weight:bold">${f.contribution.toFixed(1)}%</span>
                </div>
                <div class="bar-bg" style="height:6px; margin-bottom:10px;">
                    <div class="bar-fill" style="width:${f.contribution}%; background:#34495e;"></div>
                </div>
            `).join('');
        }
    </script>
</body>
</html>
"""

# Ensure directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# Write file
full_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
with open(full_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✓ Successfully generated frontend at: {full_path}")