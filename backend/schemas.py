"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    model_type: Optional[str] = Field("xgboost", description="Model type: 'xgboost', 'random_forest', or 'lstm'")
    forecast_days: Optional[int] = Field(1, ge=1, le=30, description="Number of days to forecast")
    use_lstm: Optional[bool] = Field(False, description="Whether to use LSTM model")


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    symbol: str
    model_type: str
    forecast_days: int
    forecasts: List[float] = Field(..., description="Predicted prices")
    forecast_dates: List[str] = Field(..., description="Dates for forecasts")
    confidence_interval: Optional[Dict[str, List[float]]] = Field(None, description="Confidence intervals")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RiskRequest(BaseModel):
    """Request model for risk analysis."""
    symbol: str = Field(..., description="Stock ticker symbol")
    benchmark_symbol: Optional[str] = Field(None, description="Benchmark symbol (e.g., ^GSPC for S&P 500)")
    period: Optional[str] = Field("1y", description="Analysis period: '3mo', '6mo', '1y', '2y'")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class RiskResponse(BaseModel):
    """Response model for risk analysis."""
    symbol: str
    benchmark: Optional[str]
    period: str
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    var_95: Optional[float] = Field(None, description="Value at Risk (95%)")
    cvar_95: Optional[float] = Field(None, description="Conditional VaR (95%)")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    beta: Optional[float] = Field(None, description="Beta (if benchmark provided)")
    trend_direction: Optional[str] = Field(None, description="Trend direction: 'upward', 'downward', 'sideways'")
    metrics: Dict[str, Any] = Field(..., description="Additional risk metrics")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExplainRequest(BaseModel):
    """Request model for LLM explanation."""
    question: str = Field(..., description="Question about financial data or indicators")
    symbol: Optional[str] = Field(None, description="Stock ticker symbol for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional financial context")
    include_sources: Optional[bool] = Field(True, description="Whether to include source documents")


class ExplainResponse(BaseModel):
    """Response model for LLM explanation."""
    answer: str = Field(..., description="LLM-generated explanation")
    question: str
    symbol: Optional[str]
    sources: Optional[List[Dict[str, str]]] = Field(None, description="Source documents")
    tokens_used: Optional[int] = Field(None, description="Tokens used for generation")
    confidence: Optional[str] = Field(None, description="Confidence level")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())













