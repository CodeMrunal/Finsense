"""
Example script: Train models and make predictions.
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.forecasting_service import ForecastingService
from src.services.risk_service import RiskService
from loguru import logger

# Configure logger
logger.add("logs/example.log", rotation="10 MB", retention="7 days")


def main():
    """Main example function."""
    symbol = "AAPL"
    
    # Initialize services
    forecasting_service = ForecastingService()
    risk_service = RiskService()
    
    print(f"\n{'='*60}")
    print(f"FinSense Example: Training and Prediction for {symbol}")
    print(f"{'='*60}\n")
    
    # 1. Train ML Model
    print("1. Training XGBoost model...")
    try:
        ml_result = forecasting_service.train_ml_model(
            symbol=symbol,
            model_type="xgboost",
            period="2y"
        )
        print(f"   ✓ Model trained successfully!")
        print(f"   Test R²: {ml_result['metrics']['test_r2']:.4f}")
        print(f"   Test RMSE: ${ml_result['metrics']['test_rmse']:.2f}")
        print(f"   Top 5 Features:")
        for i, feat in enumerate(ml_result['feature_importance'][:5], 1):
            print(f"      {i}. {feat['feature']}: {feat['importance']:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print()
    
    # 2. Train LSTM Model
    print("2. Training LSTM model...")
    try:
        lstm_result = forecasting_service.train_lstm_model(
            symbol=symbol,
            period="2y"
        )
        print(f"   ✓ LSTM model trained successfully!")
        print(f"   Test R²: {lstm_result['metrics']['test_r2']:.4f}")
        print(f"   Test RMSE: ${lstm_result['metrics']['test_rmse']:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print()
    
    # 3. Make Predictions
    print("3. Making predictions...")
    try:
        predictions = forecasting_service.predict(
            symbol=symbol,
            model_type="xgboost",
            forecast_days=7
        )
        print(f"   ✓ Predictions generated!")
        print(f"   Next 7 days forecasts:")
        for i, pred in enumerate(predictions['forecasts'], 1):
            print(f"      Day {i}: ${pred:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print()
    
    # 4. Risk Analysis
    print("4. Performing risk analysis...")
    try:
        risk_analysis = risk_service.analyze_risk(
            symbol=symbol,
            benchmark_symbol="^GSPC",  # S&P 500
            period="1y"
        )
        metrics = risk_analysis['metrics']
        print(f"   ✓ Risk analysis completed!")
        print(f"   Volatility: {metrics['volatility']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   VaR (95%): {metrics['var_95']:.2%}")
        print(f"   Maximum Drawdown: {metrics['max_drawdown']['max_drawdown']:.2%}")
        if 'beta' in metrics:
            print(f"   Beta: {metrics['beta']:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Example completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()













