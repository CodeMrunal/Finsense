# FinSense - Hybrid Artificial Intelligence System for Financial Intelligence

A production-ready, modular Python system for financial market analysis, price forecasting, and risk assessment using Machine Learning and Deep Learning models.

## 🚀 Features

- **Data Ingestion**: Fetch historical and live financial market data from Yahoo Finance
- **Feature Engineering**: Comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **ML Models**: Random Forest and XGBoost for price forecasting
- **Deep Learning**: LSTM models for time series forecasting
- **Risk Metrics**: VaR, CVaR, Sharpe Ratio, Sortino Ratio, Beta, Maximum Drawdown
- **Explainability**: SHAP values and feature importance for model interpretability
- **Personal Finance Advisor**: AI-powered chatbot with price predictions and creative financial advice (FREE - no API costs!)
- **REST API**: FastAPI-based RESTful API for programmatic access
- **Dashboard**: Interactive Streamlit dashboard for visualization and analysis
- **AEFDS**: Adaptive model selection + decision engine + explainable recommendation chatbot

## Novelty of the Project

FinSense now includes a research-oriented architecture called **Adaptive Explainable Financial Decision System (AEFDS)**:

- **Adaptive Model Selection**: The system automatically selects a model family from market regime signals (trend, volatility, RSI) using `AdaptiveModelSelector`.
- **Decision Engine**: A weighted scoring engine transforms predictive and market signals into explicit **BUY / SELL / HOLD** outputs with confidence and risk.
- **Explainable AI Layer**: Chatbot recommendations are structured and grounded in indicators, model outputs, and top feature-importance drivers.

End-to-end pipeline:

`Data -> Feature Engineering -> ML Models -> Adaptive Selector -> Decision Engine -> Chatbot -> UI`

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FinSense
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Create a `.env` file for API keys:
```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## 📁 Project Structure

```
FinSense/
├── config/                 # Configuration settings
│   ├── __init__.py
│   └── settings.py
├── src/                    # Source code
│   ├── data/              # Data ingestion
│   │   ├── __init__.py
│   │   └── data_ingestion.py
│   ├── features/          # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/            # ML models
│   │   ├── __init__.py
│   │   ├── ml_models.py
│   │   └── lstm_model.py
│   ├── risk/              # Risk metrics
│   │   ├── __init__.py
│   │   └── risk_metrics.py
│   ├── explainability/    # Model explainability
│   │   ├── __init__.py
│   │   └── explainability.py
│   └── services/          # Business logic
│       ├── __init__.py
│       ├── forecasting_service.py
│       └── risk_service.py
├── api/                   # FastAPI application
│   ├── __init__.py
│   └── main.py
├── dashboard/             # Streamlit dashboard
│   └── app.py
├── data/                  # Data storage (created automatically)
├── models/                # Trained models (created automatically)
├── logs/                  # Log files (created automatically)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment (optional, for chatbot):**
```bash
copy config\env.example .env  # Windows
# OR
cp config/env.example .env  # Mac/Linux
# Then edit .env and add your OPENAI_API_KEY
```

### Running the Project

**Option 1: Run Backend API**
```bash
python run_backend.py
```
Access API at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

**Option 2: Run Streamlit Dashboard**
```bash
streamlit run dashboard/app.py
```
Access dashboard at `http://localhost:8501`

**Option 3: Run Both (Recommended)**
- Terminal 1: `python run_backend.py`
- Terminal 2: `streamlit run dashboard/app.py`

**For detailed instructions, see:** `RUN_GUIDE.md` or `START_HERE.md`

### 3. Example Usage

#### Train a Model

```python
from src.services.forecasting_service import ForecastingService

service = ForecastingService()
result = service.train_ml_model(symbol="AAPL", model_type="xgboost")
print(f"Test R²: {result['metrics']['test_r2']:.4f}")
```

#### Make Predictions

```python
predictions = service.predict(symbol="AAPL", model_type="xgboost", forecast_days=7)
print(f"Forecasts: {predictions['forecasts']}")
```

#### Risk Analysis

```python
from src.services.risk_service import RiskService

risk_service = RiskService()
risk_analysis = risk_service.analyze_risk(
    symbol="AAPL",
    benchmark_symbol="^GSPC",  # S&P 500
    period="1y"
)
print(f"Sharpe Ratio: {risk_analysis['metrics']['sharpe_ratio']:.2f}")
```

## 📡 API Endpoints

### Predictions
- `POST /api/v1/predict` - Generate price forecasts
- `POST /api/v1/train/ml` - Train ML model
- `POST /api/v1/train/lstm` - Train LSTM model

### Risk Analysis
- `POST /api/v1/risk/analyze` - Perform risk analysis

### Features
- `GET /api/v1/features/importance` - Get feature importance
- `GET /api/v1/data/info` - Get market information

See `http://localhost:8000/docs` for interactive API documentation.

## 🧪 Model Training

### ML Models (XGBoost, Random Forest)

```python
from src.services.forecasting_service import ForecastingService

service = ForecastingService()
result = service.train_ml_model(
    symbol="AAPL",
    model_type="xgboost",  # or "random_forest"
    period="2y"
)
```

### LSTM Model

```python
result = service.train_lstm_model(
    symbol="AAPL",
    period="2y"
)
```

## 📊 Feature Engineering

The system automatically generates comprehensive features including:

- **Moving Averages**: SMA, EMA
- **Momentum Indicators**: RSI, Stochastic, ADX
- **Trend Indicators**: MACD
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, Volume Ratio
- **Statistical Features**: Rolling mean, std, min, max
- **Lag Features**: Historical price lags

## ⚠️ Risk Metrics

The system calculates:

- **Volatility**: Standard deviation of returns
- **VaR**: Value at Risk (95% and 99%)
- **CVaR**: Conditional Value at Risk
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Beta**: Market sensitivity
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Tracking Error**: Deviation from benchmark
- **Information Ratio**: Active return per unit of tracking error

## 🔍 Explainability

The system provides model interpretability through:

- **SHAP Values**: Explain individual predictions
- **Feature Importance**: Understand which features drive predictions
- **Gradient-based Explanations**: For LSTM models

## 💬 Personal Finance Advisor

The system includes an enhanced **Personal Finance Advisor** chatbot that provides:

- **Price Predictions**: Ask "Predict AAPL price for next 7 days" to get AI-powered forecasts
- **Financial Education**: Learn concepts like volatility, Sharpe ratio, RSI, MACD with creative explanations
- **Investment Advice**: Get personalized recommendations based on risk analysis
- **Creative Responses**: Real-world analogies and visual indicators for easy understanding
- **FREE**: No API costs required - works completely locally!

### Example Usage

```python
from chatbot.personal_finance_advisor import PersonalFinanceAdvisor

advisor = PersonalFinanceAdvisor()

# Price prediction
answer = advisor.chat("Predict AAPL price for next 7 days")
print(answer)

# Financial education
answer = advisor.chat("What is volatility?")
print(answer)

# Investment advice
answer = advisor.chat("Should I invest in TSLA?")
print(answer)
```

**Try it in the Dashboard:** Go to "Live & Chatbot" page and start chatting!

For detailed guide, see: `PERSONAL_ADVISOR_GUIDE.md`

## 🎨 Dashboard Features

The Streamlit dashboard includes:

1. **Dashboard**: Real-time stock data visualization
2. **Data Explorer**: Feature engineering and data exploration
3. **Model Training**: Train and evaluate models
4. **Live & Chatbot**: Real-time prices, predictions, and Personal Finance Advisor
4. **Forecasting**: Generate price predictions
5. **Risk Analysis**: Comprehensive risk metrics
6. **Feature Importance**: Visualize model features

## 🔧 Configuration

Edit `config/settings.py` or create a `.env` file to customize:

- API settings (host, port)
- Model hyperparameters
- Feature engineering parameters
- Risk metrics parameters
- Data source settings

## 📝 Logging

Logs are automatically saved to `logs/` directory with rotation and retention policies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This system is for educational and research purposes only. Financial predictions are inherently uncertain, and past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

## 🙏 Acknowledgments

- Built with Python, TensorFlow, scikit-learn, XGBoost
- Data provided by Yahoo Finance
- Visualization with Plotly and Streamlit

## 📧 Support

For issues and questions, please open an issue on GitHub.

