"""
Financial Feature Engineering Module for Time-Series Data.

This module provides comprehensive feature engineering for financial time-series data,
designed to be reusable for both ML (tabular) and LSTM (sequential) models.

Features include:
- Returns (simple, log, cumulative)
- Moving Averages (SMA, EMA)
- Volatility measures
- Trend indicators (MACD, ADX, etc.)
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import logging
from warnings import warn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialFeatureEngineer:
    """
    Feature engineering for financial time-series data.
    
    Designed to work with both ML (tabular) and LSTM (sequential) models.
    All features are calculated to preserve temporal order and can be used
    directly for time-series forecasting.
    """
    
    def __init__(self, fill_method: str = "forward"):
        """
        Initialize the feature engineer.
        
        Args:
            fill_method: Method to handle NaN values. Options:
                        'forward' - forward fill (default)
                        'backward' - backward fill
                        'zero' - fill with zeros
                        'drop' - drop NaN rows
        """
        self.fill_method = fill_method
        logger.info(f"FinancialFeatureEngineer initialized with fill_method={fill_method}")
    
    def _fill_na(self, data: pd.Series) -> pd.Series:
        """Fill NaN values based on fill_method."""
        if self.fill_method == "forward":
            return data.ffill().bfill()
        elif self.fill_method == "backward":
            return data.bfill().ffill()
        elif self.fill_method == "zero":
            return data.fillna(0)
        elif self.fill_method == "drop":
            return data.dropna()
        else:
            return data.ffill().bfill()
    
    # ==================== RETURNS ====================
    
    def calculate_returns(
        self,
        prices: pd.Series,
        method: str = "simple",
        periods: int = 1
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: 'simple' or 'log'. Default: 'simple'
            periods: Number of periods to shift. Default: 1
        
        Returns:
            Series of returns
        """
        if method == "simple":
            returns = prices.pct_change(periods=periods)
        elif method == "log":
            returns = np.log(prices / prices.shift(periods))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'")
        
        return self._fill_na(returns)
    
    def calculate_cumulative_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns.
        
        Args:
            prices: Price series
        
        Returns:
            Series of cumulative returns
        """
        returns = self.calculate_returns(prices)
        cumulative = (1 + returns).cumprod() - 1
        return cumulative
    
    def calculate_multiple_returns(
        self,
        prices: pd.Series,
        periods: List[int] = [1, 2, 3, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Calculate returns for multiple periods.
        
        Args:
            prices: Price series
            periods: List of periods to calculate returns for
        
        Returns:
            DataFrame with returns for each period
        """
        returns_df = pd.DataFrame(index=prices.index)
        
        for period in periods:
            returns_df[f'returns_{period}d'] = self.calculate_returns(prices, periods=period)
        
        return returns_df
    
    # ==================== MOVING AVERAGES ====================
    
    def calculate_sma(
        self,
        data: pd.Series,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: Input series
            window: Window size
            min_periods: Minimum number of observations. Default: window
        
        Returns:
            Series with SMA values
        """
        min_periods = min_periods or window
        sma = data.rolling(window=window, min_periods=min_periods).mean()
        return self._fill_na(sma)
    
    def calculate_ema(
        self,
        data: pd.Series,
        span: int,
        adjust: bool = False,
        alpha: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Input series
            span: Span for EMA calculation
            adjust: Whether to use adjust parameter. Default: False
            alpha: Smoothing factor (alternative to span)
        
        Returns:
            Series with EMA values
        """
        if alpha is not None:
            ema = data.ewm(alpha=alpha, adjust=adjust).mean()
        else:
            ema = data.ewm(span=span, adjust=adjust).mean()
        
        return self._fill_na(ema)
    
    def calculate_multiple_sma(
        self,
        data: pd.Series,
        windows: List[int] = [5, 10, 20, 50, 100, 200]
    ) -> pd.DataFrame:
        """
        Calculate multiple SMAs with different windows.
        
        Args:
            data: Input series
            windows: List of window sizes
        
        Returns:
            DataFrame with SMAs for each window
        """
        sma_df = pd.DataFrame(index=data.index)
        
        for window in windows:
            sma_df[f'sma_{window}'] = self.calculate_sma(data, window)
        
        return sma_df
    
    def calculate_multiple_ema(
        self,
        data: pd.Series,
        spans: List[int] = [5, 10, 20, 50, 100]
    ) -> pd.DataFrame:
        """
        Calculate multiple EMAs with different spans.
        
        Args:
            data: Input series
            spans: List of span values
        
        Returns:
            DataFrame with EMAs for each span
        """
        ema_df = pd.DataFrame(index=data.index)
        
        for span in spans:
            ema_df[f'ema_{span}'] = self.calculate_ema(data, span)
        
        return ema_df
    
    def calculate_price_to_ma_ratio(
        self,
        prices: pd.Series,
        ma_window: int,
        ma_type: str = "sma"
    ) -> pd.Series:
        """
        Calculate ratio of price to moving average.
        
        Args:
            prices: Price series
            ma_window: Moving average window
            ma_type: 'sma' or 'ema'. Default: 'sma'
        
        Returns:
            Series with price/MA ratio
        """
        if ma_type == "sma":
            ma = self.calculate_sma(prices, ma_window)
        elif ma_type == "ema":
            ma = self.calculate_ema(prices, ma_window)
        else:
            raise ValueError("ma_type must be 'sma' or 'ema'")
        
        ratio = prices / ma
        return ratio
    
    # ==================== VOLATILITY ====================
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualized: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            returns: Returns series
            window: Rolling window size. Default: 20
            annualized: Whether to annualize volatility. Default: True
            trading_days: Number of trading days per year. Default: 252
        
        Returns:
            Series with volatility values
        """
        volatility = returns.rolling(window=window).std()
        
        if annualized:
            volatility = volatility * np.sqrt(trading_days)
        
        return self._fill_na(volatility)
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualized: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate realized volatility (sum of squared returns).
        
        Args:
            returns: Returns series
            window: Rolling window size
            annualized: Whether to annualize. Default: True
            trading_days: Trading days per year. Default: 252
        
        Returns:
            Series with realized volatility
        """
        realized_vol = np.sqrt(
            (returns ** 2).rolling(window=window).sum()
        )
        
        if annualized:
            realized_vol = realized_vol * np.sqrt(trading_days / window)
        
        return self._fill_na(realized_vol)
    
    def calculate_parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
        annualized: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate Parkinson volatility estimator using high-low prices.
        
        Args:
            high: High price series
            low: Low price series
            window: Rolling window size
            annualized: Whether to annualize. Default: True
            trading_days: Trading days per year. Default: 252
        
        Returns:
            Series with Parkinson volatility
        """
        hl_ratio = np.log(high / low)
        parkinson_vol = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(window=window).mean()
        )
        
        if annualized:
            parkinson_vol = parkinson_vol * np.sqrt(trading_days)
        
        return self._fill_na(parkinson_vol)
    
    def calculate_garman_klass_volatility(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        annualized: bool = True,
        trading_days: int = 252
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator.
        
        Args:
            open_price: Open price series
            high: High price series
            low: Low price series
            close: Close price series
            window: Rolling window size
            annualized: Whether to annualize. Default: True
            trading_days: Trading days per year. Default: 252
        
        Returns:
            Series with Garman-Klass volatility
        """
        hl = np.log(high / low)
        co = np.log(close / open_price)
        
        gk_vol = np.sqrt(
            (0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)).rolling(window=window).mean()
        )
        
        if annualized:
            gk_vol = gk_vol * np.sqrt(trading_days)
        
        return self._fill_na(gk_vol)
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Rolling window size. Default: 14
        
        Returns:
            Series with ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return self._fill_na(atr)
    
    # ==================== TREND INDICATORS ====================
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period. Default: 12
            slow: Slow EMA period. Default: 26
            signal: Signal line EMA period. Default: 9
        
        Returns:
            DataFrame with MACD line, signal line, and histogram
        """
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }, index=prices.index)
    
    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for calculation. Default: 14
        
        Returns:
            Series with ADX values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth TR and DM
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return self._fill_na(adx)
    
    def calculate_aroon(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Aroon indicator.
        
        Args:
            high: High price series
            low: Low price series
            window: Period for calculation. Default: 14
        
        Returns:
            DataFrame with Aroon Up and Aroon Down
        """
        aroon_up = (
            high.rolling(window=window).apply(
                lambda x: (window - x.argmax()) / window * 100
            )
        )
        
        aroon_down = (
            low.rolling(window=window).apply(
                lambda x: (window - x.argmin()) / window * 100
            )
        )
        
        aroon_oscillator = aroon_up - aroon_down
        
        return pd.DataFrame({
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }, index=high.index)
    
    def calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for calculation. Default: 20
        
        Returns:
            Series with CCI values
        """
        tp = (high + low + close) / 3  # Typical Price
        sma_tp = self.calculate_sma(tp, window)
        mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad)
        
        return self._fill_na(cci)
    
    def calculate_trend_strength(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate trend strength as correlation between price and time.
        
        Args:
            prices: Price series
            window: Rolling window size
        
        Returns:
            Series with trend strength (-1 to 1)
        """
        trend_strength = prices.rolling(window=window).apply(
            lambda x: np.corrcoef(range(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
        return self._fill_na(trend_strength)
    
    # ==================== COMPREHENSIVE FEATURE ENGINEERING ====================
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        high_column: Optional[str] = None,
        low_column: Optional[str] = None,
        open_column: Optional[str] = None,
        volume_column: Optional[str] = None,
        include_returns: bool = True,
        include_ma: bool = True,
        include_volatility: bool = True,
        include_trend: bool = True,
        ma_windows: List[int] = [5, 10, 20, 50],
        ema_spans: List[int] = [5, 10, 20, 50],
        return_periods: List[int] = [1, 2, 5, 10],
        volatility_window: int = 20
    ) -> pd.DataFrame:
        """
        Engineer comprehensive features from OHLCV data.
        
        This method creates a feature-rich dataset suitable for both ML and LSTM models.
        Features are added as new columns to preserve temporal order.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Name of close price column. Default: 'close'
            high_column: Name of high price column. If None, uses price_column
            low_column: Name of low price column. If None, uses price_column
            open_column: Name of open price column. If None, uses price_column
            volume_column: Name of volume column. Optional
            include_returns: Whether to include return features. Default: True
            include_ma: Whether to include moving averages. Default: True
            include_volatility: Whether to include volatility features. Default: True
            include_trend: Whether to include trend indicators. Default: True
            ma_windows: List of SMA windows. Default: [5, 10, 20, 50]
            ema_spans: List of EMA spans. Default: [5, 10, 20, 50]
            return_periods: List of return periods. Default: [1, 2, 5, 10]
            volatility_window: Window for volatility calculation. Default: 20
        
        Returns:
            DataFrame with original data and engineered features
        """
        df = data.copy()
        
        # Validate columns
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        prices = df[price_column]
        high = df[high_column] if high_column and high_column in df.columns else prices
        low = df[low_column] if low_column and low_column in df.columns else prices
        open_price = df[open_column] if open_column and open_column in df.columns else prices
        volume = df[volume_column] if volume_column and volume_column in df.columns else None
        
        logger.info(f"Engineering features for {len(df)} records")
        
        # Returns
        if include_returns:
            logger.debug("Calculating returns...")
            returns = self.calculate_returns(prices)
            df['returns'] = returns
            df['log_returns'] = self.calculate_returns(prices, method='log')
            df['cumulative_returns'] = self.calculate_cumulative_returns(prices)
            
            # Multiple period returns
            for period in return_periods:
                df[f'returns_{period}d'] = self.calculate_returns(prices, periods=period)
        
        # Moving Averages
        if include_ma:
            logger.debug("Calculating moving averages...")
            # SMAs
            for window in ma_windows:
                df[f'sma_{window}'] = self.calculate_sma(prices, window)
                df[f'price_sma_{window}_ratio'] = self.calculate_price_to_ma_ratio(
                    prices, window, ma_type='sma'
                )
            
            # EMAs
            for span in ema_spans:
                df[f'ema_{span}'] = self.calculate_ema(prices, span)
                df[f'price_ema_{span}_ratio'] = self.calculate_price_to_ma_ratio(
                    prices, span, ma_type='ema'
                )
            
            # MA crossovers
            if len(ma_windows) >= 2:
                df['sma_cross'] = (
                    self.calculate_sma(prices, ma_windows[0]) - 
                    self.calculate_sma(prices, ma_windows[1])
                )
        
        # Volatility
        if include_volatility:
            logger.debug("Calculating volatility...")
            returns = df.get('returns', self.calculate_returns(prices))
            
            df['volatility'] = self.calculate_volatility(returns, window=volatility_window)
            df['realized_volatility'] = self.calculate_realized_volatility(
                returns, window=volatility_window
            )
            
            if high_column and low_column:
                df['parkinson_volatility'] = self.calculate_parkinson_volatility(
                    high, low, window=volatility_window
                )
            
            if open_column and high_column and low_column:
                df['garman_klass_volatility'] = self.calculate_garman_klass_volatility(
                    open_price, high, low, prices, window=volatility_window
                )
            
            if high_column and low_column:
                df['atr'] = self.calculate_atr(high, low, prices)
        
        # Trend Indicators
        if include_trend:
            logger.debug("Calculating trend indicators...")
            
            # MACD
            macd_features = self.calculate_macd(prices)
            df = pd.concat([df, macd_features], axis=1)
            
            # ADX
            if high_column and low_column:
                df['adx'] = self.calculate_adx(high, low, prices)
                
                # Aroon
                aroon_features = self.calculate_aroon(high, low)
                df = pd.concat([df, aroon_features], axis=1)
                
                # CCI
                if open_column:
                    df['cci'] = self.calculate_cci(high, low, prices)
            
            # Trend strength
            df['trend_strength'] = self.calculate_trend_strength(prices)
        
        # Drop rows with NaN (created by rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        
        return df
    
    def prepare_for_ml(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        drop_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML models (tabular format).
        
        Args:
            data: DataFrame with features
            target_column: Name of target column
            drop_columns: List of columns to drop (e.g., date, ticker)
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = data.copy()
        
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Drop any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        return X, y
    
    def prepare_for_lstm(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for LSTM models (sequential format).
        
        Args:
            data: DataFrame with features
            sequence_length: Length of input sequences
            target_column: Name of target column
            feature_columns: List of feature columns to use. If None, uses all numeric columns
        
        Returns:
            Tuple of (X sequences, y targets) as numpy arrays
        """
        df = data.copy()
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        # Extract features and target
        X_data = df[feature_columns].values
        y_data = df[target_column].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_data)):
            X_sequences.append(X_data[i - sequence_length:i])
            y_sequences.append(y_data[i])
        
        return np.array(X_sequences), np.array(y_sequences)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Financial Feature Engineering - Example Usage")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    prices = 100 + np.cumsum(np.random.randn(252) * 0.5)
    high = prices + np.abs(np.random.randn(252) * 0.3)
    low = prices - np.abs(np.random.randn(252) * 0.3)
    open_price = prices + np.random.randn(252) * 0.2
    volume = np.random.randint(1000000, 10000000, 252)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    # Initialize feature engineer
    engineer = FinancialFeatureEngineer()
    
    # Engineer features
    print("\n1. Engineering comprehensive features...")
    features_df = engineer.engineer_features(
        sample_data,
        price_column='close',
        high_column='high',
        low_column='low',
        open_column='open',
        volume_column='volume'
    )
    
    print(f"   Original columns: {len(sample_data.columns)}")
    print(f"   Features columns: {len(features_df.columns)}")
    print(f"   Records: {len(features_df)}")
    
    # Prepare for ML
    print("\n2. Preparing for ML model...")
    X_ml, y_ml = engineer.prepare_for_ml(features_df, target_column='close')
    print(f"   Features shape: {X_ml.shape}")
    print(f"   Target shape: {y_ml.shape}")
    
    # Prepare for LSTM
    print("\n3. Preparing for LSTM model...")
    X_lstm, y_lstm = engineer.prepare_for_lstm(
        features_df,
        sequence_length=20,
        target_column='close'
    )
    print(f"   Sequences shape: {X_lstm.shape}")
    print(f"   Targets shape: {y_lstm.shape}")
    
    # Display feature summary
    print("\n4. Feature Summary:")
    feature_cols = [col for col in features_df.columns if col not in sample_data.columns]
    print(f"   Total engineered features: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:10]}")
    
    print("\n" + "=" * 60)













