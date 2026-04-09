"""
Data ingestion module for fetching historical and live financial market data.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from loguru import logger
from config import settings


class DataIngestion:
    """Handles data ingestion from various financial data sources."""
    
    def __init__(self):
        """Initialize data ingestion."""
        self.logger = logger
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data retrieved for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add metadata
            data['symbol'] = symbol
            data['date'] = data.index
            
            self.logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_live_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch latest market data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            DataFrame with latest OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                # Fallback to daily data
                data = ticker.history(period="5d", interval="1d")
            
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data['symbol'] = symbol
            data['date'] = data.index
            
            self.logger.info(f"Fetched live data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching live data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to fetch
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data_dict[symbol] = self.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
            except Exception as e:
                self.logger.warning(f"Skipping {symbol}: {str(e)}")
                continue
        
        return data_dict
    
    def get_market_info(self, symbol: str) -> Dict:
        """
        Get additional market information for a symbol.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with market information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'pe_ratio': info.get('trailingPE', 0),
            }
        except Exception as e:
            self.logger.error(f"Error fetching market info for {symbol}: {str(e)}")
            return {}
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            True if data is valid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if data.empty:
            return False
        
        for col in required_columns:
            if col not in data.columns:
                return False
        
        # Check for missing values
        if data[required_columns].isnull().sum().sum() > len(data) * 0.1:
            self.logger.warning("More than 10% missing values detected")
            return False
        
        return True













