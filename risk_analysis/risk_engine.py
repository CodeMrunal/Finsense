"""
Financial Risk Analysis Engine.

This module provides reusable functions for computing key financial risk metrics:
- Volatility (historical and rolling)
- Sharpe Ratio (risk-adjusted returns)
- Trend Direction (price trend analysis)

All functions are designed to be standalone and reusable.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_returns(prices: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Calculate simple returns from price series.
    
    Args:
        prices: Price series or array
    
    Returns:
        Series of returns
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    returns = prices.pct_change().dropna()
    return returns


def calculate_log_returns(prices: Union[pd.Series, np.ndarray]) -> pd.Series:
    """
    Calculate logarithmic returns from price series.
    
    Args:
        prices: Price series or array
    
    Returns:
        Series of log returns
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def calculate_volatility(
    prices: Optional[Union[pd.Series, np.ndarray]] = None,
    returns: Optional[Union[pd.Series, np.ndarray]] = None,
    window: Optional[int] = None,
    annualized: bool = True,
    trading_days: int = 252,
    method: str = "standard"
) -> Union[float, pd.Series]:
    """
    Calculate volatility (standard deviation of returns).
    
    Supports both single value and rolling volatility calculations.
    
    Args:
        prices: Price series (if returns not provided)
        returns: Returns series (if prices not provided)
        window: Rolling window size. If None, calculates overall volatility
        annualized: Whether to annualize volatility. Default: True
        trading_days: Number of trading days per year. Default: 252
        method: Calculation method. Options:
               - 'standard': Standard deviation (default)
               - 'realized': Realized volatility (sum of squared returns)
    
    Returns:
        Volatility value (float) or Series (if window specified)
    
    Examples:
        >>> prices = pd.Series([100, 102, 101, 105, 103])
        >>> vol = calculate_volatility(prices, annualized=False)
        >>> rolling_vol = calculate_volatility(prices, window=3, annualized=False)
    """
    # Calculate returns if not provided
    if returns is None:
        if prices is None:
            raise ValueError("Either 'prices' or 'returns' must be provided")
        returns = calculate_returns(prices)
    
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    # Calculate volatility
    if method == "standard":
        if window is None:
            # Overall volatility
            volatility = returns.std()
        else:
            # Rolling volatility
            volatility = returns.rolling(window=window).std()
    
    elif method == "realized":
        if window is None:
            # Overall realized volatility
            volatility = np.sqrt((returns ** 2).sum())
        else:
            # Rolling realized volatility
            volatility = np.sqrt((returns ** 2).rolling(window=window).sum())
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard' or 'realized'")
    
    # Annualize if requested
    if annualized:
        if window is None:
            # For overall volatility, assume daily returns
            volatility = volatility * np.sqrt(trading_days)
        else:
            # For rolling volatility, annualize based on window
            volatility = volatility * np.sqrt(trading_days / window)
    
    return volatility


def calculate_sharpe_ratio(
    prices: Optional[Union[pd.Series, np.ndarray]] = None,
    returns: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.02,
    annualized: bool = True,
    trading_days: int = 252,
    window: Optional[int] = None
) -> Union[float, pd.Series]:
    """
    Calculate Sharpe Ratio (risk-adjusted return).
    
    Sharpe Ratio = (Mean Return - Risk-Free Rate) / Volatility
    
    Args:
        prices: Price series (if returns not provided)
        returns: Returns series (if prices not provided)
        risk_free_rate: Annual risk-free rate. Default: 0.02 (2%)
        annualized: Whether to annualize the ratio. Default: True
        trading_days: Number of trading days per year. Default: 252
        window: Rolling window size. If None, calculates overall Sharpe ratio
    
    Returns:
        Sharpe Ratio value (float) or Series (if window specified)
    
    Examples:
        >>> prices = pd.Series([100, 102, 101, 105, 103, 108, 106])
        >>> sharpe = calculate_sharpe_ratio(prices, risk_free_rate=0.02)
        >>> rolling_sharpe = calculate_sharpe_ratio(prices, window=5, risk_free_rate=0.02)
    """
    # Calculate returns if not provided
    if returns is None:
        if prices is None:
            raise ValueError("Either 'prices' or 'returns' must be provided")
        returns = calculate_returns(prices)
    
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    # Calculate daily risk-free rate
    daily_rf_rate = risk_free_rate / trading_days
    
    # Calculate excess returns
    excess_returns = returns - daily_rf_rate
    
    if window is None:
        # Overall Sharpe ratio
        mean_excess_return = excess_returns.mean()
        volatility = excess_returns.std()
        
        if volatility == 0:
            logger.warning("Zero volatility detected. Returning NaN for Sharpe ratio.")
            return np.nan
        
        sharpe_ratio = mean_excess_return / volatility
        
        if annualized:
            sharpe_ratio = sharpe_ratio * np.sqrt(trading_days)
    
    else:
        # Rolling Sharpe ratio
        mean_excess_return = excess_returns.rolling(window=window).mean()
        volatility = excess_returns.rolling(window=window).std()
        
        sharpe_ratio = mean_excess_return / volatility
        
        if annualized:
            sharpe_ratio = sharpe_ratio * np.sqrt(trading_days)
        
        # Replace inf values with NaN
        sharpe_ratio = sharpe_ratio.replace([np.inf, -np.inf], np.nan)
    
    return sharpe_ratio


def calculate_trend_direction(
    prices: Union[pd.Series, np.ndarray],
    method: str = "linear_regression",
    window: Optional[int] = None,
    threshold: float = 0.0
) -> Union[str, pd.Series]:
    """
    Calculate trend direction (upward, downward, or sideways).
    
    Args:
        prices: Price series
        method: Method to determine trend. Options:
               - 'linear_regression': Linear regression slope (default)
               - 'moving_average': Compare price to moving average
               - 'price_change': Percentage change over period
        window: Window size for calculation. If None, uses entire series
        threshold: Threshold for trend strength. Default: 0.0
                   Positive values require stronger trends
    
    Returns:
        Trend direction:
        - 'upward': Positive trend
        - 'downward': Negative trend
        - 'sideways': No clear trend
        Returns Series if window specified, string otherwise
    
    Examples:
        >>> prices = pd.Series([100, 102, 104, 106, 108])
        >>> trend = calculate_trend_direction(prices)
        >>> rolling_trend = calculate_trend_direction(prices, window=3)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    if method == "linear_regression":
        trend = _trend_linear_regression(prices, window, threshold)
    
    elif method == "moving_average":
        trend = _trend_moving_average(prices, window, threshold)
    
    elif method == "price_change":
        trend = _trend_price_change(prices, window, threshold)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear_regression', 'moving_average', or 'price_change'")
    
    return trend


def _trend_linear_regression(
    prices: pd.Series,
    window: Optional[int],
    threshold: float
) -> Union[str, pd.Series]:
    """Calculate trend using linear regression slope."""
    if window is None:
        # Overall trend
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices.values, 1)[0]
        
        if abs(slope) < threshold:
            return 'sideways'
        elif slope > 0:
            return 'upward'
        else:
            return 'downward'
    
    else:
        # Rolling trend
        def calc_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            return np.polyfit(x, series.values, 1)[0]
        
        slopes = prices.rolling(window=window).apply(calc_slope)
        
        trend_series = pd.Series(index=prices.index, dtype=object)
        trend_series[slopes > threshold] = 'upward'
        trend_series[slopes < -threshold] = 'downward'
        trend_series[(slopes >= -threshold) & (slopes <= threshold)] = 'sideways'
        
        return trend_series


def _trend_moving_average(
    prices: pd.Series,
    window: Optional[int],
    threshold: float
) -> Union[str, pd.Series]:
    """Calculate trend using moving average comparison."""
    if window is None:
        ma_window = min(20, len(prices) // 4)  # Default to 20 or 25% of data
    else:
        ma_window = window
    
    ma = prices.rolling(window=ma_window).mean()
    price_ma_ratio = prices / ma
    
    if window is None:
        # Overall trend
        avg_ratio = price_ma_ratio.mean()
        
        if abs(avg_ratio - 1.0) < threshold:
            return 'sideways'
        elif avg_ratio > 1.0:
            return 'upward'
        else:
            return 'downward'
    
    else:
        # Rolling trend
        trend_series = pd.Series(index=prices.index, dtype=object)
        trend_series[price_ma_ratio > (1 + threshold)] = 'upward'
        trend_series[price_ma_ratio < (1 - threshold)] = 'downward'
        trend_series[(price_ma_ratio >= (1 - threshold)) & (price_ma_ratio <= (1 + threshold))] = 'sideways'
        
        return trend_series


def _trend_price_change(
    prices: pd.Series,
    window: Optional[int],
    threshold: float
) -> Union[str, pd.Series]:
    """Calculate trend using percentage price change."""
    if window is None:
        # Overall trend
        pct_change = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        
        if abs(pct_change) < threshold:
            return 'sideways'
        elif pct_change > 0:
            return 'upward'
        else:
            return 'downward'
    
    else:
        # Rolling trend
        pct_change = prices.pct_change(window) * 100
        
        trend_series = pd.Series(index=prices.index, dtype=object)
        trend_series[pct_change > threshold] = 'upward'
        trend_series[pct_change < -threshold] = 'downward'
        trend_series[(pct_change >= -threshold) & (pct_change <= threshold)] = 'sideways'
        
        return trend_series


def analyze_risk(
    prices: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    window: Optional[int] = None,
    annualized: bool = True,
    trading_days: int = 252
) -> Dict[str, Union[float, str, pd.Series]]:
    """
    Comprehensive risk analysis combining volatility, Sharpe ratio, and trend.
    
    This is a convenience function that computes all three metrics in one call.
    
    Args:
        prices: Price series
        risk_free_rate: Annual risk-free rate. Default: 0.02 (2%)
        window: Rolling window size. If None, calculates overall metrics
        annualized: Whether to annualize metrics. Default: True
        trading_days: Number of trading days per year. Default: 252
    
    Returns:
        Dictionary with keys:
        - 'volatility': Volatility value or Series
        - 'sharpe_ratio': Sharpe ratio value or Series
        - 'trend_direction': Trend direction string or Series
        - 'returns': Returns series (for reference)
    
    Examples:
        >>> prices = pd.Series([100, 102, 101, 105, 103, 108, 106, 110])
        >>> risk_metrics = analyze_risk(prices)
        >>> print(f"Volatility: {risk_metrics['volatility']:.4f}")
        >>> print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
        >>> print(f"Trend: {risk_metrics['trend_direction']}")
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Calculate returns
    returns = calculate_returns(prices)
    
    # Calculate volatility
    volatility = calculate_volatility(
        returns=returns,
        window=window,
        annualized=annualized,
        trading_days=trading_days
    )
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(
        returns=returns,
        risk_free_rate=risk_free_rate,
        annualized=annualized,
        trading_days=trading_days,
        window=window
    )
    
    # Calculate trend direction
    trend_direction = calculate_trend_direction(
        prices=prices,
        window=window
    )
    
    return {
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'trend_direction': trend_direction,
        'returns': returns
    }


# Convenience functions for common use cases

def get_volatility(
    prices: Union[pd.Series, np.ndarray],
    annualized: bool = True
) -> float:
    """
    Quick function to get overall volatility.
    
    Args:
        prices: Price series
        annualized: Whether to annualize. Default: True
    
    Returns:
        Volatility value
    """
    return float(calculate_volatility(prices=prices, annualized=annualized))


def get_sharpe_ratio(
    prices: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    annualized: bool = True
) -> float:
    """
    Quick function to get overall Sharpe ratio.
    
    Args:
        prices: Price series
        risk_free_rate: Annual risk-free rate. Default: 0.02
        annualized: Whether to annualize. Default: True
    
    Returns:
        Sharpe ratio value
    """
    result = calculate_sharpe_ratio(
        prices=prices,
        risk_free_rate=risk_free_rate,
        annualized=annualized
    )
    return float(result) if not isinstance(result, pd.Series) else float(result.iloc[-1])


def get_trend(
    prices: Union[pd.Series, np.ndarray],
    method: str = "linear_regression"
) -> str:
    """
    Quick function to get overall trend direction.
    
    Args:
        prices: Price series
        method: Trend calculation method. Default: 'linear_regression'
    
    Returns:
        Trend direction string ('upward', 'downward', or 'sideways')
    """
    result = calculate_trend_direction(prices=prices, method=method)
    return result if isinstance(result, str) else result.iloc[-1]


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Financial Risk Analysis Engine - Example Usage")
    print("=" * 60)
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    # Generate trending price data
    trend = np.linspace(100, 120, 252)
    noise = np.random.randn(252) * 2
    prices = pd.Series(trend + noise, index=dates)
    
    print("\n1. Volatility Analysis:")
    vol = calculate_volatility(prices, annualized=True)
    print(f"   Annualized Volatility: {vol:.4f} ({vol*100:.2f}%)")
    
    rolling_vol = calculate_volatility(prices, window=20, annualized=True)
    print(f"   Latest Rolling Volatility (20-day): {rolling_vol.iloc[-1]:.4f}")
    
    print("\n2. Sharpe Ratio Analysis:")
    sharpe = calculate_sharpe_ratio(prices, risk_free_rate=0.02, annualized=True)
    print(f"   Sharpe Ratio: {sharpe:.4f}")
    
    rolling_sharpe = calculate_sharpe_ratio(prices, window=20, risk_free_rate=0.02, annualized=True)
    print(f"   Latest Rolling Sharpe (20-day): {rolling_sharpe.iloc[-1]:.4f}")
    
    print("\n3. Trend Direction Analysis:")
    trend = calculate_trend_direction(prices, method="linear_regression")
    print(f"   Overall Trend: {trend}")
    
    rolling_trend = calculate_trend_direction(prices, window=20, method="linear_regression")
    print(f"   Latest Rolling Trend (20-day): {rolling_trend.iloc[-1]}")
    
    print("\n4. Comprehensive Risk Analysis:")
    risk_metrics = analyze_risk(prices, risk_free_rate=0.02)
    print(f"   Volatility: {risk_metrics['volatility']:.4f}")
    print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
    print(f"   Trend Direction: {risk_metrics['trend_direction']}")
    
    print("\n5. Quick Functions:")
    print(f"   Quick Volatility: {get_volatility(prices):.4f}")
    print(f"   Quick Sharpe: {get_sharpe_ratio(prices):.4f}")
    print(f"   Quick Trend: {get_trend(prices)}")
    
    print("\n" + "=" * 60)













