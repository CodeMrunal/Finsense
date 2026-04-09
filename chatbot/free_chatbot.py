"""
Free Chatbot Alternative - Works Without OpenAI API.

This version uses pre-defined responses and pattern matching instead of LLM,
making it completely free to use.
"""
import re
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class FreeFinancialChatbot:
    """
    Free chatbot that works without API costs.
    Uses pattern matching and pre-defined knowledge base.
    """
    
    def __init__(self):
        """Initialize free chatbot."""
        self.knowledge_base = self._load_knowledge_base()
        logger.info("FreeFinancialChatbot initialized (no API required)")
    
    def _load_knowledge_base(self) -> Dict[str, str]:
        """Load financial knowledge base."""
        return {
            "volatility": """
            Volatility measures how much a stock's price fluctuates over time.
            - High volatility = Large price swings (risky)
            - Low volatility = Stable prices (safer)
            - Measured as standard deviation of returns
            - Annualized volatility multiplies daily volatility by sqrt(252)
            """,
            
            "sharpe ratio": """
            Sharpe Ratio measures risk-adjusted returns.
            Formula: (Return - Risk-Free Rate) / Volatility
            - Sharpe > 1 = Good risk-adjusted returns
            - Sharpe > 2 = Excellent
            - Sharpe < 1 = Poor risk-adjusted returns
            Higher is better - shows you're getting good returns for the risk taken.
            """,
            
            "trend": """
            Trend direction shows if prices are moving up, down, or sideways.
            - Upward trend: Prices generally increasing over time
            - Downward trend: Prices generally decreasing
            - Sideways: Prices moving in a range
            Trends help identify buying/selling opportunities.
            """,
            
            "risk": """
            Financial risk refers to potential losses in investments.
            Types:
            - Market risk: Overall market movements
            - Volatility risk: Price fluctuations
            - Credit risk: Default risk
            Risk management involves diversification and position sizing.
            """,
            
            "returns": """
            Returns measure investment performance.
            - Simple returns: (Current Price - Previous Price) / Previous Price
            - Log returns: ln(Current / Previous)
            - Annualized returns: Daily returns × 252 trading days
            Positive returns = profit, Negative = loss
            """,
            
            "moving average": """
            Moving averages smooth out price data to show trends.
            - SMA (Simple): Average of last N prices
            - EMA (Exponential): Gives more weight to recent prices
            Common periods: 20, 50, 200 days
            Price above MA = uptrend, below = downtrend
            """,
            
            "rsi": """
            RSI (Relative Strength Index) measures momentum.
            Range: 0-100
            - RSI > 70: Overbought (might fall)
            - RSI < 30: Oversold (might rise)
            - RSI 30-70: Normal range
            Helps identify buy/sell signals.
            """,
            
            "macd": """
            MACD (Moving Average Convergence Divergence) shows trend changes.
            Components:
            - MACD line: Fast EMA - Slow EMA
            - Signal line: EMA of MACD line
            - Histogram: MACD - Signal
            MACD above signal = bullish, below = bearish
            """,
            
            "beta": """
            Beta measures stock sensitivity to market movements.
            - Beta = 1: Moves with market
            - Beta > 1: More volatile than market (aggressive)
            - Beta < 1: Less volatile (defensive)
            - Beta < 0: Moves opposite to market (rare)
            """,
            
            "var": """
            VaR (Value at Risk) estimates potential losses.
            - VaR 95%: 95% chance losses won't exceed this amount
            - VaR 99%: 99% chance losses won't exceed this amount
            Example: VaR 95% = -2% means 95% chance losses < 2%
            """,
            
            "drawdown": """
            Drawdown measures peak-to-trough decline.
            - Maximum drawdown: Largest decline from peak
            - Current drawdown: Current decline from recent peak
            Lower drawdown = better risk management
            """,
            
            "portfolio": """
            Portfolio management involves:
            - Diversification: Spreading risk across assets
            - Asset allocation: Mix of stocks, bonds, etc.
            - Rebalancing: Adjusting holdings periodically
            Goal: Maximize returns while managing risk
            """,
            
            "investment": """
            Investment strategies:
            - Buy and hold: Long-term approach
            - Dollar-cost averaging: Regular investments
            - Value investing: Buying undervalued stocks
            - Growth investing: Focusing on high-growth companies
            Choose based on your risk tolerance and goals.
            """
        }
    
    def _find_keywords(self, question: str) -> List[str]:
        """Extract keywords from question."""
        question_lower = question.lower()
        keywords = []
        
        for key in self.knowledge_base.keys():
            if key in question_lower:
                keywords.append(key)
        
        # Also check for variations
        variations = {
            "volatile": "volatility",
            "sharpe": "sharpe ratio",
            "trending": "trend",
            "risky": "risk",
            "return": "returns",
            "ma": "moving average",
            "moving average": "moving average",
            "relative strength": "rsi",
            "momentum": "rsi",
            "convergence": "macd",
            "sensitivity": "beta",
            "value at risk": "var",
            "loss": "var",
            "decline": "drawdown",
            "diversification": "portfolio",
            "strategy": "investment"
        }
        
        for variation, key in variations.items():
            if variation in question_lower and key not in keywords:
                keywords.append(key)
        
        return keywords
    
    def _generate_response(self, question: str, keywords: List[str], context: Optional[Dict] = None) -> str:
        """Generate response based on keywords and context."""
        response_parts = []
        
        # Add relevant knowledge
        for keyword in keywords[:3]:  # Limit to top 3 topics
            if keyword in self.knowledge_base:
                response_parts.append(self.knowledge_base[keyword].strip())
        
        # Add context-specific information
        if context:
            for ticker, indicators in context.items():
                context_info = f"\n\nFor {ticker}:"
                if 'volatility' in indicators and indicators['volatility']:
                    context_info += f"\n- Volatility: {indicators['volatility']:.2%}"
                if 'sharpe_ratio' in indicators and indicators['sharpe_ratio']:
                    context_info += f"\n- Sharpe Ratio: {indicators['sharpe_ratio']:.2f}"
                if 'trend_direction' in indicators:
                    context_info += f"\n- Trend: {indicators['trend_direction']}"
                response_parts.append(context_info)
        
        # If no keywords found, provide general answer
        if not response_parts:
            response_parts.append("""
            I can help explain financial concepts like:
            - Volatility and risk metrics
            - Sharpe Ratio and risk-adjusted returns
            - Trend analysis
            - Technical indicators (RSI, MACD, etc.)
            - Portfolio management
            
            Please ask a specific question about these topics!
            """)
        
        return "\n\n".join(response_parts)
    
    def chat(self, question: str, ticker: Optional[str] = None, context: Optional[Dict] = None) -> str:
        """
        Answer financial questions using free pattern matching.
        
        Args:
            question: User question
            ticker: Optional ticker symbol
            context: Optional financial context/indicators
        
        Returns:
            Answer string
        """
        keywords = self._find_keywords(question)
        response = self._generate_response(question, keywords, context)
        
        logger.info(f"Answered question with keywords: {keywords}")
        return response
    
    def query(
        self,
        question: str,
        ticker: Optional[str] = None,
        k: int = 4,
        return_source_documents: bool = True
    ) -> Dict[str, any]:
        """
        Query interface matching RAG chatbot API.
        
        Args:
            question: User question
            ticker: Optional ticker symbol
            k: Number of sources (for compatibility)
            return_source_documents: Whether to return sources
        
        Returns:
            Dictionary matching RAG chatbot response format
        """
        # Extract context if ticker provided
        context = None
        if ticker:
            context = {ticker.upper(): {}}
        
        answer = self.chat(question, ticker=ticker, context=context)
        
        result = {
            "answer": answer,
            "question": question,
            "ticker": ticker,
            "tokens_used": 0,  # Free version
            "cost": 0.0,  # Free version
            "retrieved_documents_count": len(self._find_keywords(question))
        }
        
        if return_source_documents:
            keywords = self._find_keywords(question)
            result["source_documents"] = [
                {
                    "content": self.knowledge_base.get(keyword, "")[:200],
                    "source": f"knowledge_base_{keyword}",
                    "type": "financial_knowledge"
                }
                for keyword in keywords[:k]
            ]
        
        return result


# Compatibility alias
FinancialRAGChatbot = FreeFinancialChatbot  # For easy switching

