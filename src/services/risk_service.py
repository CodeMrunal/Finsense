"""
Risk analysis service that calculates financial risk metrics.
"""
import pandas as pd
from typing import Dict, Optional
from loguru import logger

from src.data.data_ingestion import DataIngestion
from src.risk.risk_metrics import RiskMetrics


class RiskService:
    """Service for risk analysis."""
    
    def __init__(self):
        """Initialize risk service."""
        self.data_ingestion = DataIngestion()
        self.risk_metrics = RiskMetrics()
        self.logger = logger
    
    def analyze_risk(
        self,
        symbol: str,
        benchmark_symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> Dict:
        """
        Perform comprehensive risk analysis for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            benchmark_symbol: Optional benchmark symbol (e.g., '^GSPC' for S&P 500)
            start_date: Start date
            end_date: End date
            period: Period to analyze
        
        Returns:
            Dictionary with risk metrics
        """
        # Fetch data
        self.logger.info(f"Analyzing risk for {symbol}...")
        data = self.data_ingestion.fetch_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        
        prices = data['close']
        
        # Fetch benchmark if provided
        market_prices = None
        if benchmark_symbol:
            try:
                benchmark_data = self.data_ingestion.fetch_historical_data(
                    symbol=benchmark_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
                market_prices = benchmark_data['close']
            except Exception as e:
                self.logger.warning(f"Could not fetch benchmark data: {str(e)}")
        
        # Calculate all risk metrics
        metrics = self.risk_metrics.calculate_all_metrics(prices, market_prices)
        
        return {
            'symbol': symbol,
            'benchmark': benchmark_symbol,
            'metrics': metrics,
            'analysis_date': pd.Timestamp.now().isoformat()
        }













