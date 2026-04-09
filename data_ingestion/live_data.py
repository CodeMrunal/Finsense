"""
Live Market Data Ingestion Module using Yahoo Finance.

This module provides functionality to fetch real-time stock prices
with optimization for repeated API calls through caching and rate limiting.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict
from collections import defaultdict
import time
import logging
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveDataFetcher:
    """
    Fetch live market data with optimization for repeated API calls.
    
    Features:
    - Real-time stock price fetching
    - Caching to minimize API calls
    - Rate limiting to prevent API throttling
    - Batch fetching for multiple tickers
    - Automatic retry on failures
    """
    
    def __init__(
        self,
        cache_duration: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the live data fetcher.
        
        Args:
            cache_duration: Cache duration in seconds. Default: 60 seconds
            max_retries: Maximum number of retry attempts. Default: 3
            retry_delay: Delay between retries in seconds. Default: 1.0
            rate_limit_delay: Minimum delay between API calls in seconds. Default: 0.1
        """
        self.cache_duration = cache_duration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        
        # Cache storage: {ticker: (data, timestamp)}
        self._cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._cache_lock = Lock()
        
        # Rate limiting: track last API call time
        self._last_api_call: float = 0.0
        self._rate_limit_lock = Lock()
        
        logger.info(f"LiveDataFetcher initialized with cache_duration={cache_duration}s")
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limiting."""
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_api_call
            
            if time_since_last_call < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last_call
                time.sleep(sleep_time)
            
            self._last_api_call = time.time()
    
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data for ticker is still valid."""
        with self._cache_lock:
            if ticker not in self._cache:
                return False
            
            _, cache_time = self._cache[ticker]
            age = time.time() - cache_time
            
            return age < self.cache_duration
    
    def _get_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get data from cache if valid."""
        with self._cache_lock:
            if self._is_cache_valid(ticker):
                data, _ = self._cache[ticker]
                logger.debug(f"Retrieved {ticker} from cache")
                return data.copy()
            return None
    
    def _update_cache(self, ticker: str, data: pd.DataFrame):
        """Update cache with new data."""
        with self._cache_lock:
            self._cache[ticker] = (data.copy(), time.time())
            logger.debug(f"Updated cache for {ticker}")
    
    def _fetch_from_api(
        self,
        ticker: str,
        interval: str = "1m",
        period: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance API.
        
        Args:
            ticker: Stock ticker symbol
            interval: Data interval. Options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d'
            period: Period to fetch. Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        
        Returns:
            DataFrame with OHLCV data
        
        Raises:
            Exception: If API call fails after retries
        """
        self._wait_for_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching live data for {ticker} (attempt {attempt + 1}/{self.max_retries})")
                
                ticker_obj = yf.Ticker(ticker)
                
                # Fetch recent data (last day for live prices)
                data = ticker_obj.history(period=period, interval=interval)
                
                if data.empty:
                    raise ValueError(f"No data returned for ticker {ticker}")
                
                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                
                # Add metadata
                data['ticker'] = ticker
                data['timestamp'] = data.index
                
                logger.info(f"Successfully fetched {len(data)} records for {ticker}")
                return data
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch data for {ticker} after {self.max_retries} attempts: {str(e)}")
                    raise
    
    def get_live_price(
        self,
        ticker: str,
        use_cache: bool = True,
        interval: str = "1m",
        period: str = "1d"
    ) -> Dict[str, Union[float, datetime, str]]:
        """
        Get the latest live price and OHLC values for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data if available. Default: True
            interval: Data interval for fetching. Default: '1m'
            period: Period to fetch. Default: '1d'
        
        Returns:
            Dictionary with latest price data:
            {
                'ticker': str,
                'timestamp': datetime,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': int,
                'is_cached': bool
            }
        
        Raises:
            Exception: If data cannot be fetched
        """
        ticker = ticker.upper()
        is_cached = False
        
        # Check cache first
        if use_cache:
            cached_data = self._get_from_cache(ticker)
            if cached_data is not None:
                is_cached = True
                data = cached_data
            else:
                data = self._fetch_from_api(ticker, interval=interval, period=period)
                self._update_cache(ticker, data)
        else:
            data = self._fetch_from_api(ticker, interval=interval, period=period)
            self._update_cache(ticker, data)
        
        # Get latest record
        latest = data.iloc[-1]
        
        result = {
            'ticker': ticker,
            'timestamp': latest['timestamp'] if 'timestamp' in latest else data.index[-1],
            'open': float(latest.get('open', 0)),
            'high': float(latest.get('high', 0)),
            'low': float(latest.get('low', 0)),
            'close': float(latest.get('close', 0)),
            'volume': int(latest.get('volume', 0)),
            'is_cached': is_cached
        }
        
        return result
    
    def get_live_ohlc(
        self,
        ticker: str,
        use_cache: bool = True,
        interval: str = "1m",
        period: str = "1d"
    ) -> pd.DataFrame:
        """
        Get latest OHLC (Open, High, Low, Close) values as DataFrame.
        
        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data. Default: True
            interval: Data interval. Default: '1m'
            period: Period to fetch. Default: '1d'
        
        Returns:
            DataFrame with OHLC columns and latest row
        """
        ticker = ticker.upper()
        
        # Check cache first
        if use_cache:
            cached_data = self._get_from_cache(ticker)
            if cached_data is not None:
                data = cached_data
            else:
                data = self._fetch_from_api(ticker, interval=interval, period=period)
                self._update_cache(ticker, data)
        else:
            data = self._fetch_from_api(ticker, interval=interval, period=period)
            self._update_cache(ticker, data)
        
        # Return OHLC columns
        ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in ohlc_columns if col in data.columns]
        
        if 'timestamp' in data.columns:
            result = data[['timestamp'] + available_columns].tail(1)
        else:
            result = data[available_columns].tail(1)
            result.index.name = 'timestamp'
        
        return result
    
    def get_multiple_live_prices(
        self,
        tickers: List[str],
        use_cache: bool = True,
        interval: str = "1m",
        period: str = "1d"
    ) -> Dict[str, Dict[str, Union[float, datetime, str]]]:
        """
        Get live prices for multiple tickers efficiently.
        
        Args:
            tickers: List of ticker symbols
            use_cache: Whether to use cached data. Default: True
            interval: Data interval. Default: '1m'
            period: Period to fetch. Default: '1d'
        
        Returns:
            Dictionary mapping ticker symbols to their live price data
        """
        results = {}
        
        for ticker in tickers:
            try:
                ticker_upper = ticker.upper()
                results[ticker_upper] = self.get_live_price(
                    ticker=ticker_upper,
                    use_cache=use_cache,
                    interval=interval,
                    period=period
                )
            except Exception as e:
                logger.error(f"Failed to fetch live price for {ticker}: {str(e)}")
                results[ticker.upper()] = {
                    'ticker': ticker.upper(),
                    'error': str(e)
                }
        
        return results
    
    def get_recent_data(
        self,
        ticker: str,
        minutes: int = 60,
        use_cache: bool = True,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Get recent market data for the last N minutes.
        
        Args:
            ticker: Stock ticker symbol
            minutes: Number of minutes of data to retrieve. Default: 60
            use_cache: Whether to use cached data. Default: True
            interval: Data interval. Default: '1m'
        
        Returns:
            DataFrame with recent OHLCV data
        """
        ticker = ticker.upper()
        
        # Determine period based on minutes
        if minutes <= 60:
            period = "1d"
        elif minutes <= 390:  # ~6.5 hours (trading day)
            period = "1d"
        else:
            period = "5d"
        
        # Check cache first
        if use_cache:
            cached_data = self._get_from_cache(ticker)
            if cached_data is not None:
                data = cached_data
            else:
                data = self._fetch_from_api(ticker, interval=interval, period=period)
                self._update_cache(ticker, data)
        else:
            data = self._fetch_from_api(ticker, interval=interval, period=period)
            self._update_cache(ticker, data)
        
        # Filter to last N minutes
        if 'timestamp' in data.columns:
            cutoff_time = data['timestamp'].max() - timedelta(minutes=minutes)
            filtered_data = data[data['timestamp'] >= cutoff_time]
        else:
            cutoff_time = data.index.max() - timedelta(minutes=minutes)
            filtered_data = data[data.index >= cutoff_time]
        
        return filtered_data
    
    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cache for a specific ticker or all tickers.
        
        Args:
            ticker: Ticker symbol to clear. If None, clears all cache. Default: None
        """
        with self._cache_lock:
            if ticker:
                ticker_upper = ticker.upper()
                if ticker_upper in self._cache:
                    del self._cache[ticker_upper]
                    logger.info(f"Cleared cache for {ticker_upper}")
            else:
                self._cache.clear()
                logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Dict]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._cache_lock:
            cache_info = {}
            current_time = time.time()
            
            for ticker, (data, cache_time) in self._cache.items():
                age = current_time - cache_time
                is_valid = age < self.cache_duration
                
                cache_info[ticker] = {
                    'cached_at': datetime.fromtimestamp(cache_time).isoformat(),
                    'age_seconds': round(age, 2),
                    'is_valid': is_valid,
                    'records': len(data),
                    'latest_timestamp': str(data['timestamp'].max() if 'timestamp' in data.columns else data.index.max())
                }
            
            return {
                'total_cached_tickers': len(self._cache),
                'cache_duration_seconds': self.cache_duration,
                'tickers': cache_info
            }


# Global instance for convenience
_default_fetcher = None


def get_live_price(
    ticker: str,
    cache_duration: int = 60,
    use_cache: bool = True
) -> Dict[str, Union[float, datetime, str]]:
    """
    Convenience function to get live price for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        cache_duration: Cache duration in seconds. Default: 60
        use_cache: Whether to use cache. Default: True
    
    Returns:
        Dictionary with latest price data
    """
    global _default_fetcher
    
    if _default_fetcher is None:
        _default_fetcher = LiveDataFetcher(cache_duration=cache_duration)
    
    return _default_fetcher.get_live_price(ticker, use_cache=use_cache)


def get_live_ohlc(
    ticker: str,
    cache_duration: int = 60,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Convenience function to get latest OHLC values.
    
    Args:
        ticker: Stock ticker symbol
        cache_duration: Cache duration in seconds. Default: 60
        use_cache: Whether to use cache. Default: True
    
    Returns:
        DataFrame with OHLC data
    """
    global _default_fetcher
    
    if _default_fetcher is None:
        _default_fetcher = LiveDataFetcher(cache_duration=cache_duration)
    
    return _default_fetcher.get_live_ohlc(ticker, use_cache=use_cache)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Live Market Data Fetcher - Example Usage")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = LiveDataFetcher(cache_duration=60, rate_limit_delay=0.2)
    
    # Example 1: Get live price for a single ticker
    print("\n1. Fetching live price for AAPL:")
    price_data = fetcher.get_live_price("AAPL")
    print(f"   Ticker: {price_data['ticker']}")
    print(f"   Timestamp: {price_data['timestamp']}")
    print(f"   Open: ${price_data['open']:.2f}")
    print(f"   High: ${price_data['high']:.2f}")
    print(f"   Low: ${price_data['low']:.2f}")
    print(f"   Close: ${price_data['close']:.2f}")
    print(f"   Volume: {price_data['volume']:,}")
    print(f"   Cached: {price_data['is_cached']}")
    
    # Example 2: Get OHLC DataFrame
    print("\n2. Fetching OHLC DataFrame for MSFT:")
    ohlc_data = fetcher.get_live_ohlc("MSFT")
    print(ohlc_data)
    
    # Example 3: Fetch multiple tickers
    print("\n3. Fetching multiple tickers:")
    multiple_prices = fetcher.get_multiple_live_prices(["AAPL", "GOOGL", "MSFT"])
    for ticker, data in multiple_prices.items():
        if 'error' not in data:
            print(f"   {ticker}: ${data['close']:.2f} (Cached: {data['is_cached']})")
        else:
            print(f"   {ticker}: Error - {data['error']}")
    
    # Example 4: Get recent data (last 30 minutes)
    print("\n4. Fetching last 30 minutes of data for AAPL:")
    recent_data = fetcher.get_recent_data("AAPL", minutes=30)
    print(f"   Records: {len(recent_data)}")
    if len(recent_data) > 0:
        print(f"   Latest Close: ${recent_data['close'].iloc[-1]:.2f}")
    
    # Example 5: Cache information
    print("\n5. Cache Information:")
    cache_info = fetcher.get_cache_info()
    print(f"   Cached tickers: {cache_info['total_cached_tickers']}")
    for ticker, info in cache_info['tickers'].items():
        print(f"   {ticker}: Age={info['age_seconds']}s, Valid={info['is_valid']}")
    
    print("\n" + "=" * 60)













