"""
Forecasting service that orchestrates data ingestion, feature engineering,
model training, and prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger

from src.data.data_ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineering
from src.models.ml_models import MLForecaster, AdaptiveModelSelector, MLDirectionClassifier
from src.models.lstm_model import LSTMForecaster
from src.models.decision_engine import compute_decision
from config import settings


class ForecastingService:
    """Service for end-to-end forecasting pipeline."""
    
    def __init__(self):
        """Initialize forecasting service."""
        self.data_ingestion = DataIngestion()
        self.feature_engineering = FeatureEngineering()
        self.adaptive_selector = AdaptiveModelSelector()
        self.logger = logger

    def _extract_market_regime_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract trend/volatility/RSI signals from engineered dataframe."""
        closes = data["close"].tail(20) if "close" in data.columns else pd.Series(dtype=float)
        trend = 0.0
        if len(closes) >= 2 and closes.iloc[0] != 0:
            trend = float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0])

        volatility = 0.2
        if "volatility" in data.columns and not data["volatility"].dropna().empty:
            volatility = float(data["volatility"].dropna().iloc[-1])

        rsi = 50.0
        if "rsi" in data.columns and not data["rsi"].dropna().empty:
            rsi = float(data["rsi"].dropna().iloc[-1])

        trend_direction = "upward" if trend > 0.01 else ("downward" if trend < -0.01 else "stable")
        return {
            "trend": trend,
            "trend_direction": trend_direction,
            "volatility": volatility,
            "rsi": rsi,
        }
    
    def prepare_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch and prepare data for modeling.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            period: Period to fetch
        
        Returns:
            DataFrame with engineered features
        """
        # Fetch data
        self.logger.info(f"Fetching data for {symbol}...")
        data = self.data_ingestion.fetch_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        
        # Validate data
        if not self.data_ingestion.validate_data(data):
            raise ValueError(f"Data validation failed for {symbol}")
        
        # Engineer features
        self.logger.info("Engineering features...")
        features = self.feature_engineering.engineer_features(data)
        
        return features
    
    def train_ml_model(
        self,
        symbol: str,
        model_type: str = "xgboost",
        adaptive_mode: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> Dict:
        """
        Train ML model for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            model_type: Type of model ('random_forest', 'xgboost', 'extra_trees', 'gradient_boosting')
            start_date: Start date
            end_date: End date
            period: Period to fetch if dates not provided
        
        Returns:
            Dictionary with model and metrics
        """
        # Prepare data
        data = self.prepare_data(symbol, start_date, end_date, period)
        
        regime = self._extract_market_regime_signals(data)
        selected_model = model_type
        if adaptive_mode:
            selected_model = self.adaptive_selector.select_model(
                trend=regime["trend"],
                volatility=regime["volatility"],
                rsi=regime["rsi"]
            )

        # Initialize model
        model = MLForecaster(model_type=selected_model)
        
        # Prepare training data
        X, y = model.prepare_data(data, target_column='close')
        
        # Train model
        metrics = model.train(X, y)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Save model
        model_path = settings.MODELS_DIR / f"{symbol}_{selected_model}_model.pkl"
        model.save_model(str(model_path))
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'selected_model': selected_model,
            'adaptive_mode': adaptive_mode,
            'market_regime': regime,
            'top_features': feature_importance.head(3)['feature'].tolist(),
            'model_path': str(model_path)
        }
    
    def train_lstm_model(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> Dict:
        """
        Train LSTM model for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            period: Period to fetch if dates not provided
        
        Returns:
            Dictionary with model and metrics
        """
        # Prepare data
        data = self.prepare_data(symbol, start_date, end_date, period)
        
        # Initialize model
        model = LSTMForecaster()
        
        # Prepare training data
        X_train, y_train, X_test, y_test = model.prepare_data(data, target_column='close')
        
        # Train model
        metrics = model.train(X_train, y_train, X_test, y_test)
        
        # Save model
        model_path = settings.MODELS_DIR / f"{symbol}_lstm_model.h5"
        model.save_model(str(model_path))
        
        return {
            'model': model,
            'metrics': metrics,
            'model_path': str(model_path)
        }
    
    def predict(
        self,
        symbol: str,
        model_type: str = "xgboost",
        forecast_days: int = 1,
        use_lstm: bool = False,
        adaptive_mode: bool = False,
        sentiment: str = "neutral"
    ) -> Dict:
        """
        Make predictions for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            model_type: Type of ML model ('random_forest', 'xgboost', 'extra_trees', 'gradient_boosting')
            forecast_days: Number of days to forecast
            use_lstm: Whether to use LSTM model
        
        Returns:
            Dictionary with predictions
        """
        # Prepare data
        data = self.prepare_data(symbol, period="1y")
        
        if use_lstm:
            # Load LSTM model
            model_path = settings.MODELS_DIR / f"{symbol}_lstm_model.h5"
            model = LSTMForecaster()
            model.load_model(str(model_path))
            
            # Prepare data for LSTM
            X_train, y_train, X_test, y_test = model.prepare_data(data, target_column='close')
            
            # Get last sequence
            last_sequence = X_test[-1] if len(X_test) > 0 else X_train[-1]
            
            # Forecast
            forecasts = model.forecast_future(last_sequence, steps=forecast_days)
            
            return {
                'symbol': symbol,
                'forecasts': forecasts.tolist(),
                'model_type': 'lstm'
            }
        
        else:
            # Load ML model
            regime = self._extract_market_regime_signals(data)
            selected_model = model_type
            if adaptive_mode:
                selected_model = self.adaptive_selector.select_model(
                    trend=regime["trend"],
                    volatility=regime["volatility"],
                    rsi=regime["rsi"]
                )

            preferred_model = selected_model
            candidate_models = [preferred_model, model_type, "xgboost", "random_forest", "gradient_boosting", "extra_trees"]
            # Keep order while removing duplicates
            seen = set()
            candidate_models = [m for m in candidate_models if not (m in seen or seen.add(m))]

            chosen_model = None
            model_path: Optional[Path] = None
            for candidate in candidate_models:
                candidate_path = settings.MODELS_DIR / f"{symbol}_{candidate}_model.pkl"
                if candidate_path.exists():
                    chosen_model = candidate
                    model_path = candidate_path
                    break

            # If no saved model exists, train the preferred model once and continue.
            if chosen_model is None:
                self.logger.warning(
                    f"No trained model file found for {symbol}. Auto-training {preferred_model}."
                )
                train_result = self.train_ml_model(
                    symbol=symbol,
                    model_type=preferred_model,
                    adaptive_mode=False,
                    period="2y"
                )
                chosen_model = train_result.get("selected_model", preferred_model)
                model_path = settings.MODELS_DIR / f"{symbol}_{chosen_model}_model.pkl"

            model = MLForecaster(model_type=selected_model)
            model.model_type = chosen_model
            model.load_model(str(model_path))
            
            # Prepare data
            X, _ = model.prepare_data(data, target_column='close')
            
            # Predict
            predictions = model.predict(X[-forecast_days:])
            pred_value = float(np.mean(predictions)) if len(predictions) else 0.0
            current_price = float(data["close"].iloc[-1]) if "close" in data.columns else pred_value
            expected_return = ((pred_value - current_price) / current_price * 100.0) if current_price else 0.0
            ma_value = float(data["sma_20"].dropna().iloc[-1]) if "sma_20" in data.columns and not data["sma_20"].dropna().empty else current_price
            decision = compute_decision(
                prediction=expected_return,
                rsi=regime["rsi"],
                sentiment=sentiment,
                volatility=regime["volatility"],
                trend=regime["trend_direction"]
            )
            top_features = model.get_feature_importance().head(3)["feature"].tolist()
            
            return {
                'symbol': symbol,
                'forecasts': predictions.tolist(),
                'model_type': chosen_model,
                'selected_model': chosen_model,
                'adaptive_mode': adaptive_mode,
                'decision': decision,
                'context': {
                    "prediction": round(expected_return, 2),
                    "trend": regime["trend_direction"],
                    "rsi": round(regime["rsi"], 2),
                    "moving_average": round(ma_value, 2),
                    "sentiment": sentiment,
                    "volatility": round(regime["volatility"], 4),
                    "top_features": top_features
                }
            }
    
    def get_feature_importance(
        self,
        symbol: str,
        model_type: str = "xgboost"
    ) -> pd.DataFrame:
        """
        Get feature importance for a trained model.
        
        Args:
            symbol: Stock ticker symbol
            model_type: Type of model
        
        Returns:
            DataFrame with feature importance
        """
        model_path = settings.MODELS_DIR / f"{symbol}_{model_type}_model.pkl"
        model = MLForecaster(model_type=model_type)
        model.load_model(str(model_path))
        
        return model.get_feature_importance()

    def train_direction_model(
        self,
        symbol: str,
        model_type: str = "random_forest",
        period: str = "2y",
        forecast_horizon: int = 5,
        confidence_threshold: float = 0.60,
        hyperparameter_tuning: bool = True
    ) -> Dict:
        """
        Train directional classification model (UP/DOWN) with time-series integrity.
        """
        raw = self.data_ingestion.fetch_historical_data(symbol=symbol, period=period)
        features = self.feature_engineering.engineer_directional_features(raw)

        clf = MLDirectionClassifier(model_type=model_type)
        X, y = clf.prepare_data(
            features,
            target_column="close",
            forecast_horizon=forecast_horizon
        )
        metrics = clf.train(
            X,
            y,
            hyperparameter_tuning=hyperparameter_tuning,
            confidence_threshold=confidence_threshold
        )
        fi = clf.get_feature_importance()
        model_path = settings.MODELS_DIR / f"{symbol}_{model_type}_direction_model.pkl"
        clf.save_model(str(model_path))

        return {
            "model": clf,
            "metrics": metrics,
            "feature_importance": fi.to_dict("records"),
            "model_path": str(model_path),
            "forecast_horizon": forecast_horizon,
            "confidence_threshold": confidence_threshold,
        }

    def predict_direction(
        self,
        symbol: str,
        model_type: str = "random_forest",
        period: str = "1y",
        forecast_horizon: int = 5,
        confidence_threshold: float = 0.60
    ) -> Dict:
        """
        Predict direction with threshold-based BUY/SELL/HOLD decision.
        """
        raw = self.data_ingestion.fetch_historical_data(symbol=symbol, period=period)
        features = self.feature_engineering.engineer_directional_features(raw)
        clf = MLDirectionClassifier(model_type=model_type)
        model_path = settings.MODELS_DIR / f"{symbol}_{model_type}_direction_model.pkl"
        clf.load_model(str(model_path))

        X, _ = clf.prepare_data(features, target_column="close", forecast_horizon=forecast_horizon)
        pred = clf.predict_with_decision(X[-1:], confidence_threshold=confidence_threshold)
        fi = clf.get_feature_importance().head(3)["feature"].tolist()

        return {
            "symbol": symbol,
            "model_type": model_type,
            "forecast_horizon": forecast_horizon,
            "probability_up": float(pred["proba_up"][0]),
            "decision": str(pred["decision"][0]),
            "top_features": fi,
        }

