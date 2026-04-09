"""
Feature engineering module for financial data.
Creates technical indicators and statistical features.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger
from config import settings


class FeatureEngineering:
    """Handles feature engineering for financial time series data."""
    
    def __init__(self):
        """Initialize feature engineering."""
        self.logger = logger
    
    def calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    def calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(
        self,
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })
    
    def calculate_bollinger_bands(
        self,
        data: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return pd.DataFrame({
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': (upper_band - lower_band) / sma,
            'bb_position': (data - lower_band) / (upper_band - lower_band)
        })
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return pd.DataFrame({
            'stoch_k': k_percent,
            'stoch_d': d_percent
        })
    
    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Average Directional Index."""
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, window=1)
        
        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate smoothed values
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    def calculate_returns(self, data: pd.Series) -> pd.Series:
        """Calculate returns."""
        return data.pct_change()
    
    def calculate_log_returns(self, data: pd.Series) -> pd.Series:
        """Calculate logarithmic returns."""
        return np.log(data / data.shift(1))
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualized: bool = True
    ) -> pd.Series:
        """Calculate rolling volatility."""
        volatility = returns.rolling(window=window).std()
        
        if annualized:
            volatility = volatility * np.sqrt(252)  # Annualize
        
        return volatility
    
    def create_lag_features(
        self,
        data: pd.Series,
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """Create lagged features."""
        lag_features = pd.DataFrame()
        
        for lag in lags:
            lag_features[f'lag_{lag}'] = data.shift(lag)
        
        return lag_features
    
    def create_rolling_features(
        self,
        data: pd.Series,
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """Create rolling statistical features."""
        rolling_features = pd.DataFrame()
        
        for window in windows:
            rolling_features[f'rolling_mean_{window}'] = data.rolling(window).mean()
            rolling_features[f'rolling_std_{window}'] = data.rolling(window).std()
            rolling_features[f'rolling_min_{window}'] = data.rolling(window).min()
            rolling_features[f'rolling_max_{window}'] = data.rolling(window).max()
        
        return rolling_features
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        include_all: bool = True
    ) -> pd.DataFrame:
        """
        Engineer comprehensive features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            target_column: Column to use for price-based features
            include_all: Whether to include all features
        
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        close = df[target_column]
        high = df.get('high', close)
        low = df.get('low', close)
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Price-based features
        df['returns'] = self.calculate_returns(close)
        df['log_returns'] = self.calculate_log_returns(close)
        df['volatility'] = self.calculate_volatility(df['returns'])
        
        # Moving Averages
        df['sma_20'] = self.calculate_sma(close, 20)
        df['sma_50'] = self.calculate_sma(close, 50)
        df['ema_12'] = self.calculate_ema(close, 12)
        df['ema_26'] = self.calculate_ema(close, 26)
        
        # Technical Indicators
        df['rsi'] = self.calculate_rsi(close)
        
        macd_features = self.calculate_macd(close)
        df = pd.concat([df, macd_features], axis=1)
        
        bb_features = self.calculate_bollinger_bands(close)
        df = pd.concat([df, bb_features], axis=1)
        
        df['atr'] = self.calculate_atr(high, low, close)
        df['obv'] = self.calculate_obv(close, volume)
        
        stoch_features = self.calculate_stochastic(high, low, close)
        df = pd.concat([df, stoch_features], axis=1)
        
        df['adx'] = self.calculate_adx(high, low, close)
        
        # Volume features
        df['volume_sma'] = self.calculate_sma(volume, 20)
        df['volume_ratio'] = volume / df['volume_sma']
        
        # Price position features
        df['high_low_ratio'] = high / low
        df['close_position'] = (close - low) / (high - low)
        
        # Lag features
        if include_all:
            lag_features = self.create_lag_features(close, lags=[1, 2, 3, 5, 10])
            df = pd.concat([df, lag_features], axis=1)
            
            rolling_features = self.create_rolling_features(close)
            df = pd.concat([df, rolling_features], axis=1)
        
        # Drop rows with NaN values created by indicators
        df = df.dropna()
        
        self.logger.info(f"Engineered {len(df.columns)} features from {len(data.columns)} original columns")
        
        return df

    def engineer_directional_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer classification-oriented financial features for directional prediction.

        Required features:
        - returns, log_returns
        - rsi (14)
        - ma_5, ma_10, ma_20
        - ema_12, ema_26
        - macd
        - volatility (rolling std 10)
        - momentum (close - close.shift(5))
        - upper_band, lower_band
        - volume_change

        Notes:
        - This method is leak-safe: all features are computed from current/past values only.
        """
        df = data.copy()
        close = df["close"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        # Core returns
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        # RSI
        df["rsi"] = self.calculate_rsi(close, window=14)

        # Moving averages and EMAs
        df["ma_5"] = self.calculate_sma(close, 5)
        df["ma_10"] = self.calculate_sma(close, 10)
        df["ma_20"] = self.calculate_sma(close, 20)
        df["ema_12"] = self.calculate_ema(close, 12)
        df["ema_26"] = self.calculate_ema(close, 26)

        # MACD
        macd_features = self.calculate_macd(close)
        df["macd"] = macd_features["macd"]

        # Volatility (10-day rolling std of returns)
        df["volatility"] = df["returns"].rolling(10).std()

        # Momentum
        df["momentum"] = close - close.shift(5)

        # Bollinger bands
        bb_features = self.calculate_bollinger_bands(close, window=20, num_std=2.0)
        df["upper_band"] = bb_features["bb_upper"]
        df["lower_band"] = bb_features["bb_lower"]

        # Volume dynamics
        df["volume_change"] = volume.pct_change()

        # Optional ratio-style signals
        df["price_vs_ma_20"] = close / df["ma_20"]
        df["ema_spread"] = df["ema_12"] - df["ema_26"]

        # Drop rows with incomplete rolling windows
        df = df.dropna().copy()
        return df













