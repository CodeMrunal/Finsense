"""
Financial risk metrics calculation module.
Computes VaR, CVaR, Sharpe Ratio, and other risk measures.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional
from loguru import logger
from config import settings


class RiskMetrics:
    """Calculate financial risk metrics."""
    
    def __init__(self, risk_free_rate: float = None):
        """
        Initialize risk metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default from settings)
        """
        self.risk_free_rate = risk_free_rate or settings.RISK_FREE_RATE
        self.logger = logger
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from prices."""
        return prices.pct_change().dropna()
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns."""
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        annualized: bool = True
    ) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Series of returns
            annualized: Whether to annualize volatility
        
        Returns:
            Volatility value
        """
        volatility = returns.std()
        
        if annualized:
            # Annualize assuming 252 trading days
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = None,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Method to calculate VaR ('historical', 'parametric', 'monte_carlo')
        
        Returns:
            VaR value (negative number representing potential loss)
        """
        confidence_level = confidence_level or settings.CONFIDENCE_LEVEL
        
        if method == "historical":
            # Historical simulation method
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == "parametric":
            # Parametric method (assumes normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean_return + z_score * std_return
        
        elif method == "monte_carlo":
            # Monte Carlo simulation
            mean_return = returns.mean()
            std_return = returns.std()
            simulations = np.random.normal(mean_return, std_return, 10000)
            var = np.percentile(simulations, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = None
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        confidence_level = confidence_level or settings.CONFIDENCE_LEVEL
        var = self.calculate_var(returns, confidence_level, method="historical")
        
        # Average of returns below VaR threshold
        cvar = returns[returns <= var].mean()
        
        return cvar
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None,
        annualized: bool = True
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            annualized: Whether to annualize the ratio
        
        Returns:
            Sharpe Ratio
        """
        risk_free_rate = risk_free_rate or self.risk_free_rate
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = excess_returns.mean() / excess_returns.std()
        
        if annualized:
            sharpe = sharpe * np.sqrt(252)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None,
        annualized: bool = True
    ) -> float:
        """
        Calculate Sortino Ratio (downside deviation only).
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            annualized: Whether to annualize the ratio
        
        Returns:
            Sortino Ratio
        """
        risk_free_rate = risk_free_rate or self.risk_free_rate
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return np.inf
        
        sortino = excess_returns.mean() / downside_std
        
        if annualized:
            sortino = sortino * np.sqrt(252)
        
        return sortino
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Dict:
        """
        Calculate Maximum Drawdown.
        
        Args:
            prices: Series of prices
        
        Returns:
            Dictionary with max drawdown and related metrics
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        
        # Calculate drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_duration = drawdown_periods.sum()
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'drawdown_duration': drawdown_duration,
            'current_drawdown': drawdown.iloc[-1]
        }
    
    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate Beta (sensitivity to market movements).
        
        Args:
            asset_returns: Returns of the asset
            market_returns: Returns of the market (e.g., S&P 500)
        
        Returns:
            Beta value
        """
        # Align indices
        aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
        asset = aligned.iloc[:, 0]
        market = aligned.iloc[:, 1]
        
        covariance = np.cov(asset, market)[0, 1]
        market_variance = np.var(market)
        
        beta = covariance / market_variance
        
        return beta
    
    def calculate_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Tracking Error.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
        
        Returns:
            Tracking error (annualized)
        """
        # Align indices
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]
        
        active_returns = portfolio - benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
        
        return tracking_error
    
    def calculate_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
        
        Returns:
            Information ratio (annualized)
        """
        # Align indices
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]
        
        active_returns = portfolio - benchmark
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return np.inf
        
        information_ratio = (active_returns.mean() / tracking_error) * np.sqrt(252)
        
        return information_ratio
    
    def calculate_all_metrics(
        self,
        prices: pd.Series,
        market_prices: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate all risk metrics.
        
        Args:
            prices: Series of prices
            market_prices: Optional market prices for beta calculation
        
        Returns:
            Dictionary with all risk metrics
        """
        returns = self.calculate_returns(prices)
        
        metrics = {
            'volatility': self.calculate_volatility(returns),
            'var_95': self.calculate_var(returns, confidence_level=0.95),
            'var_99': self.calculate_var(returns, confidence_level=0.99),
            'cvar_95': self.calculate_cvar(returns, confidence_level=0.95),
            'cvar_99': self.calculate_cvar(returns, confidence_level=0.99),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(prices),
            'mean_return': returns.mean() * 252,  # Annualized
            'total_return': (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        }
        
        if market_prices is not None:
            market_returns = self.calculate_returns(market_prices)
            metrics['beta'] = self.calculate_beta(returns, market_returns)
            metrics['tracking_error'] = self.calculate_tracking_error(returns, market_returns)
            metrics['information_ratio'] = self.calculate_information_ratio(returns, market_returns)
        
        return metrics













