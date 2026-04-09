"""
FastAPI REST API for FinSense Financial Intelligence System.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger
import sys

from config import settings
from src.services.forecasting_service import ForecastingService
from src.services.risk_service import RiskService
from src.explainability.explainability import ModelExplainability

# Configure logging
logger.add(
    settings.LOGS_DIR / "api.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.LOG_LEVEL
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Financial Intelligence System API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
forecasting_service = ForecastingService()
risk_service = RiskService()
explainability = ModelExplainability()


# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    model_type: Optional[str] = "xgboost"
    forecast_days: Optional[int] = 1
    use_lstm: Optional[bool] = False


class TrainingRequest(BaseModel):
    symbol: str
    model_type: Optional[str] = "xgboost"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class RiskAnalysisRequest(BaseModel):
    symbol: str
    benchmark_symbol: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = "1y"


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FinSense Financial Intelligence API",
        "version": settings.API_VERSION,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/predict")
async def predict(request: PredictionRequest):
    """
    Make price predictions for a symbol.
    
    Args:
        request: Prediction request with symbol and model parameters
    
    Returns:
        Predictions and metadata
    """
    try:
        result = forecasting_service.predict(
            symbol=request.symbol,
            model_type=request.model_type,
            forecast_days=request.forecast_days,
            use_lstm=request.use_lstm
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/train/ml")
async def train_ml_model(request: TrainingRequest):
    """
    Train ML model for a symbol.
    
    Args:
        request: Training request with symbol and parameters
    
    Returns:
        Training results and metrics
    """
    try:
        result = forecasting_service.train_ml_model(
            symbol=request.symbol,
            model_type=request.model_type,
            start_date=request.start_date,
            end_date=request.end_date
        )
        return {
            'symbol': request.symbol,
            'model_type': request.model_type,
            'metrics': result['metrics'],
            'feature_importance': result['feature_importance']
        }
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/train/lstm")
async def train_lstm_model(request: TrainingRequest):
    """
    Train LSTM model for a symbol.
    
    Args:
        request: Training request with symbol and parameters
    
    Returns:
        Training results and metrics
    """
    try:
        result = forecasting_service.train_lstm_model(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )
        return {
            'symbol': request.symbol,
            'model_type': 'lstm',
            'metrics': result['metrics']
        }
    except Exception as e:
        logger.error(f"LSTM training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/risk/analyze")
async def analyze_risk(request: RiskAnalysisRequest):
    """
    Perform risk analysis for a symbol.
    
    Args:
        request: Risk analysis request
    
    Returns:
        Risk metrics and analysis
    """
    try:
        result = risk_service.analyze_risk(
            symbol=request.symbol,
            benchmark_symbol=request.benchmark_symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            period=request.period
        )
        return result
    except Exception as e:
        logger.error(f"Risk analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/features/importance")
async def get_feature_importance(
    symbol: str = Query(..., description="Stock ticker symbol"),
    model_type: str = Query("xgboost", description="Model type")
):
    """
    Get feature importance for a trained model.
    
    Args:
        symbol: Stock ticker symbol
        model_type: Type of model
    
    Returns:
        Feature importance data
    """
    try:
        importance_df = forecasting_service.get_feature_importance(
            symbol=symbol,
            model_type=model_type
        )
        return {
            'symbol': symbol,
            'model_type': model_type,
            'feature_importance': importance_df.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/info")
async def get_market_info(symbol: str = Query(..., description="Stock ticker symbol")):
    """
    Get market information for a symbol.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Market information
    """
    try:
        from src.data.data_ingestion import DataIngestion
        data_ingestion = DataIngestion()
        info = data_ingestion.get_market_info(symbol)
        return info
    except Exception as e:
        logger.error(f"Market info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )













