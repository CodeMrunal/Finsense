"""
Historical Stock Market Data Downloader using Yahoo Finance.

This module provides functionality to download historical stock market data,
handle missing values, and save to CSV files in a reusable format.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataDownloader:
    """
    Download and save historical stock market data from Yahoo Finance.
    
    Features:
    - Download data for single or multiple tickers
    - Flexible date range input
    - Automatic missing value handling
    - Save to CSV with configurable options
    - Data validation and error handling
    """
    
    def __init__(self, output_dir: Union[str, Path] = "data/raw"):
        """
        Initialize the historical data downloader.
        
        Args:
            output_dir: Directory to save CSV files. Defaults to "data/raw"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")
    
    def download_data(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        prepost: bool = False,
        actions: bool = True
    ) -> pd.DataFrame:
        """
        Download historical stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
            start_date: Start date in 'YYYY-MM-DD' format or datetime object.
                       If None and period is None, defaults to 1 year ago.
            end_date: End date in 'YYYY-MM-DD' format or datetime object.
                     If None, defaults to today.
            period: Period to fetch. Options: '1d', '5d', '1mo', '3mo', '6mo',
                   '1y', '2y', '5y', '10y', 'ytd', 'max'. If provided,
                   start_date and end_date are ignored.
            interval: Data interval. Options: '1m', '2m', '5m', '15m', '30m',
                     '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
                     Default: '1d' (daily)
            auto_adjust: Adjust all OHLC automatically. Default: True
            prepost: Include pre/post market data. Default: False
            actions: Download stock splits and dividends. Default: True
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        
        Raises:
            ValueError: If ticker is invalid or no data is retrieved
            Exception: For network or other errors
        """
        try:
            logger.info(f"Downloading data for ticker: {ticker}")
            
            # Convert string dates to datetime if needed
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Set defaults if not provided
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None and period is None:
                # Default to 1 year if nothing specified
                start_date = end_date - timedelta(days=365)
                period = None
            
            # Download data using yfinance
            ticker_obj = yf.Ticker(ticker)
            
            if period:
                logger.info(f"Downloading {period} of data for {ticker}")
                data = ticker_obj.history(
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    prepost=prepost,
                    actions=actions
                )
            else:
                logger.info(f"Downloading data from {start_date.date()} to {end_date.date()} for {ticker}")
                data = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    prepost=prepost,
                    actions=actions
                )
            
            # Validate data
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {ticker}. Please check the ticker symbol.")
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Standardize column names (lowercase, replace spaces with underscores)
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add metadata columns
            data['ticker'] = ticker
            data['date'] = data.index
            data = data.reset_index(drop=False)
            
            # Reorder columns: date, ticker, then OHLCV data
            cols = ['date', 'ticker'] + [c for c in data.columns if c not in ['date', 'ticker']]
            data = data[cols]
            
            logger.info(f"Successfully downloaded {len(data)} records for {ticker}")
            logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            raise
    
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the downloaded data.
        
        Args:
            data: DataFrame with potential missing values
            method: Method to handle missing values. Options:
                   - 'forward_fill': Forward fill (default)
                   - 'backward_fill': Backward fill
                   - 'interpolate': Linear interpolation
                   - 'drop': Drop rows with missing values
                   - 'zero': Fill with zeros
        
        Returns:
            DataFrame with missing values handled
        """
        if data.isnull().sum().sum() == 0:
            logger.debug("No missing values detected")
            return data
        
        missing_count = data.isnull().sum().sum()
        logger.info(f"Handling {missing_count} missing values using method: {method}")
        
        data_cleaned = data.copy()
        
        if method == "forward_fill":
            data_cleaned = data_cleaned.ffill()
            # Fill any remaining NaN at the beginning with backward fill
            data_cleaned = data_cleaned.bfill()
        
        elif method == "backward_fill":
            data_cleaned = data_cleaned.bfill()
            # Fill any remaining NaN at the end with forward fill
            data_cleaned = data_cleaned.ffill()
        
        elif method == "interpolate":
            data_cleaned = data_cleaned.interpolate(method='linear')
            # Fill any remaining NaN with forward/backward fill
            data_cleaned = data_cleaned.ffill().bfill()
        
        elif method == "drop":
            data_cleaned = data_cleaned.dropna()
            logger.info(f"Dropped {len(data) - len(data_cleaned)} rows with missing values")
        
        elif method == "zero":
            data_cleaned = data_cleaned.fillna(0)
        
        else:
            logger.warning(f"Unknown method '{method}', using forward_fill")
            data_cleaned = data_cleaned.ffill().bfill()
        
        # Verify no missing values remain
        remaining_missing = data_cleaned.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Warning: {remaining_missing} missing values still remain after handling")
        
        return data_cleaned
    
    def save_to_csv(
        self,
        data: pd.DataFrame,
        ticker: Optional[str] = None,
        filename: Optional[str] = None,
        index: bool = False,
        date_format: str = "%Y-%m-%d"
    ) -> Path:
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            ticker: Ticker symbol (used for default filename if filename not provided)
            filename: Custom filename. If None, generates filename from ticker and dates
            index: Whether to include index in CSV. Default: False
            date_format: Format for date column. Default: "%Y-%m-%d"
        
        Returns:
            Path to the saved CSV file
        
        Raises:
            ValueError: If data is empty or filename cannot be determined
        """
        if data.empty:
            raise ValueError("Cannot save empty DataFrame to CSV")
        
        # Generate filename if not provided
        if filename is None:
            if ticker is None:
                ticker = data['ticker'].iloc[0] if 'ticker' in data.columns else "unknown"
            
            start_date = data['date'].min().strftime("%Y%m%d")
            end_date = data['date'].max().strftime("%Y%m%d")
            filename = f"{ticker}_{start_date}_{end_date}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = self.output_dir / filename
        
        # Format date column if it exists
        data_to_save = data.copy()
        if 'date' in data_to_save.columns:
            data_to_save['date'] = pd.to_datetime(data_to_save['date']).dt.strftime(date_format)
        
        # Save to CSV
        data_to_save.to_csv(filepath, index=index)
        logger.info(f"Data saved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024:.2f} KB")
        
        return filepath
    
    def download_and_save(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        filename: Optional[str] = None,
        missing_value_method: str = "forward_fill",
        **kwargs
    ) -> tuple[pd.DataFrame, Path]:
        """
        Download data and save to CSV in one step.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format or datetime
            end_date: End date in 'YYYY-MM-DD' format or datetime
            period: Period to fetch (overrides start_date/end_date)
            interval: Data interval. Default: '1d'
            filename: Custom filename for CSV. If None, auto-generated
            missing_value_method: Method to handle missing values
            **kwargs: Additional arguments passed to download_data()
        
        Returns:
            Tuple of (DataFrame, CSV file path)
        """
        # Download data
        data = self.download_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            **kwargs
        )
        
        # Handle missing values with specified method
        if missing_value_method != "forward_fill":  # download_data already uses forward_fill
            data = self._handle_missing_values(data, method=missing_value_method)
        
        # Save to CSV
        filepath = self.save_to_csv(data, ticker=ticker, filename=filename)
        
        return data, filepath
    
    def download_multiple(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        save_to_csv: bool = True,
        **kwargs
    ) -> dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format or datetime
            end_date: End date in 'YYYY-MM-DD' format or datetime
            period: Period to fetch (overrides start_date/end_date)
            interval: Data interval. Default: '1d'
            save_to_csv: Whether to save each ticker's data to CSV. Default: True
            **kwargs: Additional arguments passed to download_data()
        
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Processing ticker {ticker} ({tickers.index(ticker) + 1}/{len(tickers)})")
                
                if save_to_csv:
                    data, _ = self.download_and_save(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        period=period,
                        interval=interval,
                        **kwargs
                    )
                else:
                    data = self.download_data(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        period=period,
                        interval=interval,
                        **kwargs
                    )
                
                results[ticker] = data
                
            except Exception as e:
                logger.error(f"Failed to download data for {ticker}: {str(e)}")
                continue
        
        logger.info(f"Successfully downloaded data for {len(results)}/{len(tickers)} tickers")
        return results


# Convenience function for quick usage
def download_stock_data(
    ticker: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    period: Optional[str] = "1y",
    output_dir: Union[str, Path] = "data/raw",
    save_csv: bool = True
) -> pd.DataFrame:
    """
    Convenience function to quickly download stock data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        period: Period to fetch if dates not provided. Default: '1y'
        output_dir: Directory to save CSV. Default: 'data/raw'
        save_csv: Whether to save to CSV. Default: True
    
    Returns:
        DataFrame with stock data
    """
    downloader = HistoricalDataDownloader(output_dir=output_dir)
    
    if save_csv:
        data, _ = downloader.download_and_save(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
    else:
        data = downloader.download_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
    
    return data


if __name__ == "__main__":
    # Example usage
    downloader = HistoricalDataDownloader(output_dir="data/raw")
    
    # Example 1: Download with date range
    print("Example 1: Downloading AAPL data from 2023-01-01 to 2024-01-01")
    data1, filepath1 = downloader.download_and_save(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    print(f"Downloaded {len(data1)} records")
    print(f"Saved to: {filepath1}\n")
    
    # Example 2: Download with period
    print("Example 2: Downloading MSFT data for 1 year")
    data2, filepath2 = downloader.download_and_save(
        ticker="MSFT",
        period="1y"
    )
    print(f"Downloaded {len(data2)} records")
    print(f"Saved to: {filepath2}\n")
    
    # Example 3: Download multiple tickers
    print("Example 3: Downloading multiple tickers")
    results = downloader.download_multiple(
        tickers=["AAPL", "GOOGL", "MSFT"],
        period="6mo",
        save_to_csv=True
    )
    print(f"Downloaded data for {len(results)} tickers")

