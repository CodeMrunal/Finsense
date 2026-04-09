"""
Prediction endpoints for stock price forecasting.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..schemas import PredictionRequest, PredictionResponse, ErrorResponse
from ..dependencies import services

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict stock prices using ML or LSTM models.
    
    Args:
        request: Prediction request with symbol, model type, and forecast days
    
    Returns:
        PredictionResponse with forecasts and metadata
    """
    try:
        logger.info(f"Prediction request for {request.symbol} using {request.model_type}")
        
        # Get data downloader
        downloader = services.get_data_downloader()
        if not downloader:
            raise HTTPException(status_code=503, detail="Data downloader not available")
        
        # Download recent data
        data = downloader.download_data(
            ticker=request.symbol,
            period="1y"
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Get feature engineer
        feature_engineer = services.get_feature_engineer()
        if not feature_engineer:
            raise HTTPException(status_code=503, detail="Feature engineer not available")
        
        # Engineer features
        features = feature_engineer.engineer_features(data)
        
        # Make predictions based on model type
        if request.use_lstm or request.model_type == "lstm":
            # Use LSTM pipeline
            lstm_pipeline = services.get_lstm_pipeline()
            if not lstm_pipeline:
                raise HTTPException(status_code=503, detail="LSTM pipeline not available")
            
            # Prepare sequences
            X_seq, y_seq, _ = lstm_pipeline.prepare_sequences(features)
            
            # Load model (simplified - in production, load from saved model)
            # For now, return mock predictions
            last_price = data['close'].iloc[-1]
            forecasts = [last_price * (1 + np.random.normal(0, 0.02)) for _ in range(request.forecast_days)]
            
        else:
            # Use ML pipeline
            ml_pipeline = services.get_ml_pipeline()
            if not ml_pipeline:
                raise HTTPException(status_code=503, detail="ML pipeline not available")
            
            # Prepare features
            X, y, feature_names = ml_pipeline.prepare_features_and_target(features)
            
            # Load model (simplified - in production, load from saved model)
            # For now, return mock predictions
            last_price = data['close'].iloc[-1]
            forecasts = [last_price * (1 + np.random.normal(0, 0.02)) for _ in range(request.forecast_days)]
        
        # Generate forecast dates
        last_date = data.index[-1] if hasattr(data.index[-1], 'date') else datetime.now()
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
        elif not isinstance(last_date, datetime):
            last_date = datetime.now()
        
        forecast_dates = [
            (last_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(request.forecast_days)
        ]
        
        # Calculate confidence intervals (simplified)
        std_dev = np.std(forecasts)
        confidence_interval = {
            "lower": [f - 1.96 * std_dev for f in forecasts],
            "upper": [f + 1.96 * std_dev for f in forecasts]
        }
        
        return PredictionResponse(
            symbol=request.symbol.upper(),
            model_type=request.model_type,
            forecast_days=request.forecast_days,
            forecasts=forecasts,
            forecast_dates=forecast_dates,
            confidence_interval=confidence_interval
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/live/{symbol}")
async def get_live_price(symbol: str):
    """
    Get live price for a symbol.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Current price information
    """
    try:
        live_fetcher = services.get_live_fetcher()
        if not live_fetcher:
            raise HTTPException(status_code=503, detail="Live data fetcher not available")
        
        price_data = live_fetcher.get_live_price(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "price": price_data['close'],
            "timestamp": str(price_data['timestamp']),
            "ohlc": {
                "open": price_data['open'],
                "high": price_data['high'],
                "low": price_data['low'],
                "close": price_data['close']
            },
            "volume": price_data['volume']
        }
        
    except Exception as e:
        logger.error(f"Live price error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch live price: {str(e)}")













