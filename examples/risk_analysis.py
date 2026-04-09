"""
Example script: Comprehensive risk analysis.
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.risk_service import RiskService
from src.data.data_ingestion import DataIngestion
import pandas as pd


def main():
    """Main risk analysis example."""
    symbol = "AAPL"
    benchmark = "^GSPC"  # S&P 500
    
    print(f"\n{'='*60}")
    print(f"Risk Analysis Example: {symbol} vs {benchmark}")
    print(f"{'='*60}\n")
    
    # Initialize services
    risk_service = RiskService()
    data_ingestion = DataIngestion()
    
    # Perform risk analysis
    print("Analyzing risk metrics...")
    risk_analysis = risk_service.analyze_risk(
        symbol=symbol,
        benchmark_symbol=benchmark,
        period="1y"
    )
    
    metrics = risk_analysis['metrics']
    
    # Display results
    print("\n" + "="*60)
    print("RISK METRICS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Return Metrics:")
    print(f"   Mean Return (Annualized): {metrics['mean_return']:.2%}")
    print(f"   Total Return: {metrics['total_return']:.2f}%")
    
    print(f"\n📈 Volatility:")
    print(f"   Volatility (Annualized): {metrics['volatility']:.2%}")
    
    print(f"\n⚠️  Value at Risk:")
    print(f"   VaR (95%): {metrics['var_95']:.2%}")
    print(f"   VaR (99%): {metrics['var_99']:.2%}")
    print(f"   CVaR (95%): {metrics['cvar_95']:.2%}")
    print(f"   CVaR (99%): {metrics['cvar_99']:.2%}")
    
    print(f"\n📉 Risk-Adjusted Returns:")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    
    print(f"\n📊 Drawdown Analysis:")
    drawdown = metrics['max_drawdown']
    print(f"   Maximum Drawdown: {drawdown['max_drawdown']:.2%}")
    print(f"   Drawdown Date: {drawdown['max_drawdown_date']}")
    print(f"   Current Drawdown: {drawdown['current_drawdown']:.2%}")
    print(f"   Drawdown Duration: {drawdown['drawdown_duration']} periods")
    
    if 'beta' in metrics:
        print(f"\n🔗 Market Comparison:")
        print(f"   Beta: {metrics['beta']:.2f}")
        print(f"   Tracking Error: {metrics['tracking_error']:.2%}")
        print(f"   Information Ratio: {metrics['information_ratio']:.2f}")
    
    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()













