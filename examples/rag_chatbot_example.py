"""
Example usage of the Financial RAG Chatbot.
"""
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import FinancialRAGChatbot
from risk_analysis import analyze_risk
from data_ingestion import HistoricalDataDownloader
import pandas as pd


def main():
    """Example usage of the Financial RAG Chatbot."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Please set it to use the chatbot.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return
    
    print("=" * 60)
    print("Financial RAG Chatbot - Example")
    print("=" * 60)
    
    # Initialize chatbot
    print("\n1. Initializing chatbot...")
    chatbot = FinancialRAGChatbot(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Initialize with default knowledge
    print("\n2. Loading knowledge base...")
    chatbot.initialize_with_default_knowledge()
    
    # Download and analyze real data
    print("\n3. Fetching and analyzing AAPL data...")
    try:
        downloader = HistoricalDataDownloader()
        data = downloader.download_data("AAPL", period="1y")
        
        # Calculate risk metrics
        prices = data['close']
        risk_metrics = analyze_risk(prices, risk_free_rate=0.02)
        
        # Add financial context
        chatbot.add_financial_context(
            ticker="AAPL",
            indicators={
                "volatility": float(risk_metrics['volatility']),
                "sharpe_ratio": float(risk_metrics['sharpe_ratio']),
                "trend_direction": str(risk_metrics['trend_direction']),
                "current_price": float(prices.iloc[-1]),
                "price_change_1y": float((prices.iloc[-1] / prices.iloc[0] - 1) * 100)
            },
            prices=prices
        )
        
        print(f"   Volatility: {risk_metrics['volatility']:.4f}")
        print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
        print(f"   Trend: {risk_metrics['trend_direction']}")
        
    except Exception as e:
        print(f"   Error fetching data: {str(e)}")
        # Use dummy data
        chatbot.add_financial_context(
            ticker="AAPL",
            indicators={
                "volatility": 0.25,
                "sharpe_ratio": 1.5,
                "trend_direction": "upward",
                "current_price": 150.0
            }
        )
    
    # Example queries
    print("\n4. Example Queries:")
    print("-" * 60)
    
    questions = [
        "What does a Sharpe ratio of 1.5 indicate about AAPL's risk-adjusted returns?",
        "Explain what volatility means in the context of stock investing.",
        "How should I interpret the trend direction for AAPL?",
        "What is the relationship between volatility and investment risk?",
        "Based on the current indicators, what insights can you provide about AAPL?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 60)
        
        try:
            result = chatbot.query(question, ticker="AAPL", k=3)
            print(f"Answer: {result['answer']}")
            print(f"\nSources: {result['retrieved_documents_count']} documents")
            print(f"Tokens used: {result['tokens_used']}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()













