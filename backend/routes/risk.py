"""
Risk analysis endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from ..schemas import RiskRequest, RiskResponse, ErrorResponse
from ..dependencies import services

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("", response_model=RiskResponse)
async def analyze_risk_metrics(request: RiskRequest):
    """
    Analyze financial risk metrics for a symbol.
    
    Args:
        request: Risk analysis request with symbol and parameters
    
    Returns:
        RiskResponse with comprehensive risk metrics
    """
    try:
        logger.info(f"Risk analysis request for {request.symbol}")
        
        # Get data downloader
        downloader = services.get_data_downloader()
        if not downloader:
            raise HTTPException(status_code=503, detail="Data downloader not available")
        
        # Download data
        data = downloader.download_data(
            ticker=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            period=request.period
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        prices = data['close']
        
        # Download benchmark data if provided
        benchmark_prices = None
        if request.benchmark_symbol:
            try:
                benchmark_data = downloader.download_data(
                    ticker=request.benchmark_symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    period=request.period
                )
                benchmark_prices = benchmark_data['close']
            except Exception as e:
                logger.warning(f"Could not fetch benchmark data: {str(e)}")
        
        # Calculate risk metrics using risk_analysis module
        try:
            from risk_analysis.risk_engine import analyze_risk
            
            risk_metrics = analyze_risk(
                prices=prices,
                risk_free_rate=0.02,
                annualized=True
            )
            
            # Extract metrics
            volatility = float(risk_metrics['volatility'])
            sharpe_ratio = float(risk_metrics['sharpe_ratio']) if not isinstance(risk_metrics['sharpe_ratio'], str) else None
            trend_direction = str(risk_metrics['trend_direction'])
            
            # Additional metrics
            additional_metrics = {
                "mean_return": float(prices.pct_change().mean() * 252),
                "total_return": float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
                "price_range": {
                    "min": float(prices.min()),
                    "max": float(prices.max()),
                    "current": float(prices.iloc[-1])
                }
            }
            
            # Calculate beta if benchmark provided
            beta = None
            if benchmark_prices is not None:
                try:
                    import pandas as pd
                    from risk_analysis.risk_engine import calculate_returns
                    asset_returns = calculate_returns(prices)
                    benchmark_returns = calculate_returns(benchmark_prices)
                    
                    # Align returns
                    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
                    if len(aligned) > 0:
                        covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                        market_variance = aligned.iloc[:, 1].var()
                        beta = float(covariance / market_variance) if market_variance > 0 else None
                except Exception as e:
                    logger.warning(f"Could not calculate beta: {str(e)}")
            
            return RiskResponse(
                symbol=request.symbol.upper(),
                benchmark=request.benchmark_symbol.upper() if request.benchmark_symbol else None,
                period=request.period,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio or 0.0,
                trend_direction=trend_direction,
                beta=beta,
                metrics=additional_metrics
            )
            
        except ImportError:
            # Fallback to basic calculations
            import pandas as pd
            import numpy as np
            
            returns = prices.pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))
            sharpe_ratio = float((returns.mean() * 252 - 0.02) / volatility) if volatility > 0 else 0.0
            
            # Calculate trend direction
            trend_direction = "upward" if prices.iloc[-1] > prices.iloc[0] else "downward"
            
            return RiskResponse(
                symbol=request.symbol.upper(),
                benchmark=request.benchmark_symbol,
                period=request.period,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                trend_direction=trend_direction,
                metrics={}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")


@router.get("/{symbol}")
async def get_quick_risk(symbol: str, period: str = "1y"):
    """
    Quick risk analysis endpoint.
    
    Args:
        symbol: Stock ticker symbol
        period: Analysis period
    
    Returns:
        Basic risk metrics
    """
    request = RiskRequest(symbol=symbol, period=period)
    return await analyze_risk_metrics(request)

