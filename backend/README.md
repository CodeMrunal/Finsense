# FinSense Backend API

Modular FastAPI backend for the Financial Intelligence System.

## Structure

```
backend/
├── main.py              # Main FastAPI application
├── schemas.py          # Pydantic request/response models
├── dependencies.py     # Shared service dependencies
└── routes/
    ├── predict.py      # Prediction endpoints
    ├── risk.py         # Risk analysis endpoints
    └── explain.py      # LLM explanation endpoints
```

## Running the Server

```bash
python run_backend.py
```

Or directly:
```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Prediction Endpoints

#### POST `/predict`
Predict stock prices using ML or LSTM models.

**Request:**
```json
{
  "symbol": "AAPL",
  "model_type": "xgboost",
  "forecast_days": 7,
  "use_lstm": false
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "model_type": "xgboost",
  "forecast_days": 7,
  "forecasts": [150.5, 151.2, 150.8, ...],
  "forecast_dates": ["2024-01-15", "2024-01-16", ...],
  "confidence_interval": {
    "lower": [148.0, 148.5, ...],
    "upper": [153.0, 153.5, ...]
  },
  "timestamp": "2024-01-14T10:30:00"
}
```

#### GET `/predict/live/{symbol}`
Get live price for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 150.25,
  "timestamp": "2024-01-14T10:30:00",
  "ohlc": {
    "open": 150.0,
    "high": 150.5,
    "low": 149.8,
    "close": 150.25
  },
  "volume": 1000000
}
```

### 2. Risk Analysis Endpoints

#### POST `/risk`
Analyze financial risk metrics.

**Request:**
```json
{
  "symbol": "AAPL",
  "benchmark_symbol": "^GSPC",
  "period": "1y",
  "start_date": null,
  "end_date": null
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "benchmark": "^GSPC",
  "period": "1y",
  "volatility": 0.25,
  "sharpe_ratio": 1.5,
  "sortino_ratio": 1.8,
  "var_95": -0.02,
  "cvar_95": -0.03,
  "max_drawdown": -0.15,
  "beta": 1.2,
  "trend_direction": "upward",
  "metrics": {
    "mean_return": 0.12,
    "total_return": 15.5,
    "price_range": {
      "min": 140.0,
      "max": 180.0,
      "current": 150.25
    }
  },
  "timestamp": "2024-01-14T10:30:00"
}
```

#### GET `/risk/{symbol}`
Quick risk analysis.

**Query Parameters:**
- `period`: Analysis period (default: "1y")

### 3. Explanation Endpoints

#### POST `/explain`
Get LLM-powered explanation for financial questions.

**Request:**
```json
{
  "question": "What does a Sharpe ratio of 1.5 mean?",
  "symbol": "AAPL",
  "context": {
    "AAPL": {
      "volatility": 0.25,
      "sharpe_ratio": 1.5,
      "current_price": 150.0
    }
  },
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "A Sharpe ratio of 1.5 indicates that...",
  "question": "What does a Sharpe ratio of 1.5 mean?",
  "symbol": "AAPL",
  "sources": [
    {
      "content": "The Sharpe ratio measures...",
      "source": "document_1",
      "type": "financial_knowledge"
    }
  ],
  "tokens_used": 250,
  "confidence": "high",
  "timestamp": "2024-01-14T10:30:00"
}
```

#### GET `/explain/health`
Check RAG chatbot availability.

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

- `OPENAI_API_KEY`: Required for `/explain` endpoint
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## Error Handling

All endpoints return standardized error responses:

```json
{
  "error": "Error message",
  "detail": "Additional details",
  "timestamp": "2024-01-14T10:30:00"
}
```

## Features

- ✅ Modular route design
- ✅ Pydantic request/response validation
- ✅ Comprehensive error handling
- ✅ CORS support
- ✅ Service dependency injection
- ✅ Logging and monitoring
- ✅ Interactive API documentation













