"""
Personal Finance Advisor Chatbot with Price Prediction.

This enhanced chatbot acts as a personal finance advisor, providing:
- Creative and personalized financial advice
- Price predictions using ML/LSTM models
- Investment recommendations
- Risk analysis insights
- Conversational and helpful responses
"""
import re
import sys
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.services.forecasting_service import ForecastingService
    from src.services.risk_service import RiskService
    from src.data.data_ingestion import DataIngestion
    from data_ingestion.live_data import LiveDataFetcher
    FORECASTING_AVAILABLE = True
except ImportError as e:
    FORECASTING_AVAILABLE = False
    logging.warning(f"Forecasting services not available: {e}")

logger = logging.getLogger(__name__)


class PersonalFinanceAdvisor:
    """
    Enhanced personal finance advisor chatbot with price prediction capabilities.
    
    Features:
    - Creative and personalized financial advice
    - Price predictions using ML/LSTM models
    - Investment recommendations based on risk analysis
    - Conversational and helpful responses
    """
    
    def __init__(self):
        """Initialize the personal finance advisor."""
        self.knowledge_base = self._load_knowledge_base()
        self.advice_templates = self._load_advice_templates()
        
        # Initialize services if available
        if FORECASTING_AVAILABLE:
            try:
                self.forecasting_service = ForecastingService()
                self.risk_service = RiskService()
                self.data_ingestion = DataIngestion()
                self.live_fetcher = LiveDataFetcher()
                self.services_available = True
            except Exception as e:
                logger.warning(f"Some services not available: {e}")
                self.services_available = False
        else:
            self.services_available = False
        
        logger.info("PersonalFinanceAdvisor initialized")
    
    def _load_knowledge_base(self) -> Dict[str, str]:
        """Load financial knowledge base with enhanced explanations."""
        return {
            "volatility": """
            Volatility is like the heartbeat of a stock - it shows how much the price moves up and down.
            
            Think of it this way:
            - High volatility (30%+) = A roller coaster ride - exciting but risky! Great for traders who can handle stress.
            - Medium volatility (15-30%) = A bumpy road - some ups and downs, manageable for most investors.
            - Low volatility (<15%) = A smooth highway - steady and predictable, perfect for conservative investors.
            
            💡 Pro Tip: If you're new to investing, start with low-volatility stocks. As you gain experience, you can explore higher volatility opportunities for potentially better returns.
            """,
            
            "sharpe ratio": """
            The Sharpe Ratio is your "risk-adjusted return scorecard" - it tells you if you're getting good returns for the risk you're taking.
            
            Here's the simple breakdown:
            - Sharpe > 2.0 = ⭐⭐⭐ Excellent! You're getting great returns with controlled risk.
            - Sharpe 1.0-2.0 = ⭐⭐ Good! Solid risk-adjusted performance.
            - Sharpe 0.5-1.0 = ⭐ Okay, but could be better. Consider reviewing your strategy.
            - Sharpe < 0.5 = ⚠️ Needs improvement. You might be taking too much risk for the returns.
            
            Formula: (Your Return - Risk-Free Rate) / Volatility
            
            💡 Think of it like a restaurant rating: A 5-star restaurant (high Sharpe) gives you great food (returns) with good service (low risk). A 2-star place might have good food but terrible service (high risk).
            """,
            
            "trend": """
            Trends are like the direction of a river - they show where prices are flowing.
            
            Three types of trends:
            1. 📈 Uptrend (Bullish): Prices climbing higher - like climbing a mountain. Great for buying opportunities!
            2. 📉 Downtrend (Bearish): Prices falling - like going downhill. Time to be cautious or consider selling.
            3. ➡️ Sideways (Range-bound): Prices moving in a box - like a boat in calm waters. Good for range trading.
            
            💡 Golden Rule: "The trend is your friend until it ends." Following trends can be profitable, but always have an exit strategy!
            """,
            
            "risk": """
            Risk in investing is like weather - you can't control it, but you can prepare for it!
            
            Types of risk:
            - Market Risk: The whole market goes down (like a storm affecting everyone)
            - Volatility Risk: Prices swing wildly (like sudden weather changes)
            - Company Risk: A specific company has problems (like one area getting hit harder)
            
            💡 Risk Management Strategies:
            1. Diversification: Don't put all eggs in one basket - spread investments across different stocks/sectors
            2. Position Sizing: Only invest what you can afford to lose
            3. Stop-Loss Orders: Set automatic sell points to limit losses
            4. Time Horizon: Longer time horizons can reduce risk through compounding
            
            Remember: No risk, no reward - but smart risk management is key to long-term success!
            """,
            
            "investment": """
            Investment strategies are like different paths up a mountain - they all lead to the top, but some are faster, some are safer.
            
            Popular Strategies:
            
            1. 🏛️ Buy and Hold (Long-term):
               - Best for: Patient investors, retirement planning
               - Time: 5+ years
               - Risk: Lower (time smooths out volatility)
               - Example: Buying S&P 500 index and holding for decades
            
            2. 💰 Dollar-Cost Averaging:
               - Best for: Regular investors, reducing timing risk
               - How: Invest fixed amount monthly regardless of price
               - Benefit: Buy more when prices are low, less when high
            
            3. 🔍 Value Investing:
               - Best for: Patient investors who like research
               - Strategy: Buy undervalued stocks (like finding a sale)
               - Famous: Warren Buffett's approach
            
            4. 🚀 Growth Investing:
               - Best for: Investors seeking high growth
               - Strategy: Buy companies with high growth potential
               - Risk: Higher (growth stocks can be volatile)
            
            💡 My Recommendation: Start with a mix! Use dollar-cost averaging to buy a diversified portfolio, then hold for the long term. As you learn, you can explore value or growth strategies.
            """,
            
            "portfolio": """
            A portfolio is like a well-balanced meal - you need different ingredients (stocks) for nutrition (returns) and health (risk management).
            
            Portfolio Building 101:
            
            🥗 The Balanced Plate:
            - 60% Stocks (for growth)
            - 30% Bonds (for stability)
            - 10% Cash/Alternatives (for emergencies/opportunities)
            
            📊 Diversification Rules:
            1. Sector Diversification: Don't put everything in tech - spread across healthcare, finance, consumer goods, etc.
            2. Company Size: Mix large-cap (stable), mid-cap (growth), small-cap (high risk/reward)
            3. Geographic: Consider international stocks for global exposure
            4. Asset Classes: Stocks, bonds, REITs, commodities
            
            💡 The 80/20 Rule: Often, 20% of your investments generate 80% of returns. Focus on quality over quantity!
            
            Remember: A good portfolio is like a good team - each member plays a different role, but together they win!
            """,
            
            "rsi": """
            RSI (Relative Strength Index) is like a speedometer for stock momentum - it tells you if a stock is "overheating" or "running out of gas."
            
            The RSI Scale (0-100):
            - RSI > 70: 🔴 Overbought - Stock might be too hot, could cool down soon (potential sell signal)
            - RSI 50-70: 🟡 Strong momentum - Stock is doing well, but watch for overbought conditions
            - RSI 30-50: 🟢 Normal range - Healthy momentum
            - RSI < 30: 🟢 Oversold - Stock might be oversold, could bounce back (potential buy signal)
            
            💡 Think of RSI like a car's RPM gauge:
            - Too high (RSI > 70) = Engine overheating, slow down!
            - Too low (RSI < 30) = Engine idling, might need a push to start
            
            Pro Tip: RSI works best when combined with other indicators. Don't rely on it alone!
            """,
            
            "macd": """
            MACD (Moving Average Convergence Divergence) is like a traffic light for stock trends - it helps you know when to go, slow down, or stop.
            
            MACD Components:
            1. MACD Line: The difference between fast and slow moving averages
            2. Signal Line: The average of the MACD line
            3. Histogram: The difference between MACD and Signal lines
            
            Signals:
            - 🟢 MACD crosses above Signal = Bullish (green light - consider buying)
            - 🔴 MACD crosses below Signal = Bearish (red light - consider selling)
            - 📊 Histogram growing = Momentum increasing
            - 📉 Histogram shrinking = Momentum decreasing
            
            💡 Real-World Analogy: MACD is like watching two runners:
            - When the fast runner (MACD) passes the slow runner (Signal), momentum is building!
            - When the fast runner falls behind, momentum is weakening.
            
            Best Use: MACD is great for identifying trend changes and momentum shifts.
            """,
            
            "beta": """
            Beta measures how "jumpy" a stock is compared to the overall market - it's like comparing a sports car to a family sedan.
            
            Beta Values Explained:
            - Beta = 1.0: Moves exactly with the market (like an average car)
            - Beta > 1.0: More volatile than market (like a sports car - faster but riskier)
            - Beta < 1.0: Less volatile than market (like a sedan - slower but safer)
            - Beta < 0: Moves opposite to market (rare, like a reverse gear)
            
            Examples:
            - Tech stocks often have Beta > 1.5 (volatile, high growth potential)
            - Utility stocks often have Beta < 0.8 (stable, defensive)
            
            💡 Investment Strategy:
            - High Beta (>1.5): For aggressive investors who can handle volatility
            - Low Beta (<0.8): For conservative investors seeking stability
            - Beta around 1.0: Balanced approach, moves with market
            
            Remember: High beta = higher risk but potentially higher returns. Low beta = lower risk but potentially lower returns.
            """
        }
    
    def _load_advice_templates(self) -> Dict[str, List[str]]:
        """Load creative advice templates for personalized responses."""
        return {
            "greeting": [
                "Hello! I'm your personal finance advisor. Ready to make smart financial decisions? Let's dive in! 💰",
                "Hi there! I'm here to help you navigate the financial markets. What would you like to know? 📈",
                "Welcome! Think of me as your financial co-pilot. Where would you like to go today? 🚀"
            ],
            "price_prediction_intro": [
                "Great question! Let me analyze {ticker} and give you a prediction based on our AI models. 🔮",
                "I'll use our advanced ML models to predict {ticker}'s future price. Give me a moment... ⚡",
                "Predicting prices is like forecasting weather - we use patterns and data. Let me check {ticker} for you! 🌤️"
            ],
            "investment_recommendation": [
                "Based on my analysis, here's what I think: 💡",
                "Here's my take on this investment opportunity: 🎯",
                "Let me share some insights that might help: 📊"
            ],
            "risk_warning": [
                "⚠️ Important: Remember that all investments carry risk. Past performance doesn't guarantee future results.",
                "💡 Pro Tip: Always do your own research and never invest more than you can afford to lose.",
                "📌 Disclaimer: These are predictions, not guarantees. Markets can be unpredictable!"
            ],
            "encouragement": [
                "You're asking the right questions! That's the first step to financial success. 🌟",
                "Smart thinking! Understanding these concepts will make you a better investor. 🎓",
                "Great question! Knowledge is power in investing. Keep learning! 💪"
            ]
        }
    
    def _extract_ticker(self, question: str) -> Optional[str]:
        """Extract ticker symbol from question."""
        # Common patterns: "AAPL", "$AAPL", "Apple stock", "predict AAPL"
        patterns = [
            r'\$?([A-Z]{1,5})\b',  # Ticker symbols (1-5 uppercase letters)
            r'(?:stock|ticker|symbol)\s+([A-Z]{1,5})',  # "stock AAPL"
            r'([A-Z]{1,5})\s+(?:stock|price|prediction)',  # "AAPL stock"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.upper())
            if match:
                ticker = match.group(1)
                # Validate it's likely a ticker (not common words)
                if len(ticker) <= 5 and ticker.isalpha():
                    return ticker
        
        # Check for common company names
        company_map = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'amazon': 'AMZN',
            'tesla': 'TSLA', 'meta': 'META', 'nvidia': 'NVDA', 'netflix': 'NFLX',
            'disney': 'DIS', 'coca': 'KO', 'pepsi': 'PEP', 'walmart': 'WMT',
            'jpmorgan': 'JPM', 'bank': 'BAC', 'goldman': 'GS', 'visa': 'V',
            'mastercard': 'MA', 'paypal': 'PYPL', 'uber': 'UBER', 'airbnb': 'ABNB'
        }
        
        question_lower = question.lower()
        for company, ticker in company_map.items():
            if company in question_lower:
                return ticker
        
        return None
    
    def _extract_forecast_days(self, question: str) -> int:
        """Extract number of days to forecast from question."""
        # Patterns: "next 7 days", "7 days", "next week", "next month"
        patterns = [
            (r'next\s+(\d+)\s+days?', 1),
            (r'(\d+)\s+days?', 1),
            (r'next\s+week', 7),
            (r'next\s+month', 30),
            (r'(\d+)\s+weeks?', lambda m: int(m.group(1)) * 7),
            (r'(\d+)\s+months?', lambda m: int(m.group(1)) * 30),
        ]
        
        for pattern, default in patterns:
            match = re.search(pattern, question.lower())
            if match:
                if callable(default):
                    return default(match)
                elif isinstance(default, int):
                    if match.groups():
                        return int(match.group(1))
                    return default
        
        return 7  # Default to 7 days
    
    def _detect_prediction_intent(self, question: str) -> bool:
        """Detect if user wants a price prediction."""
        prediction_keywords = [
            'predict', 'forecast', 'future price', 'will go', 'will be',
            'price prediction', 'where will', 'what will', 'next price',
            'tomorrow', 'next week', 'next month', 'future value'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in prediction_keywords)
    
    def _detect_stock_comparison_intent(self, question: str) -> bool:
        """Detect if user wants to compare stocks or find best stock."""
        comparison_keywords = [
            'which stock', 'best stock', 'better', 'compare', 'recommend',
            'should i invest', 'which is better', 'top stock', 'best investment',
            'good stock', 'safe stock', 'best to buy'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in comparison_keywords)
    
    def _detect_safety_intent(self, question: str) -> bool:
        """Detect if user is asking about investment safety."""
        safety_keywords = [
            'safe', 'safety', 'risky', 'risk', 'dangerous', 'secure',
            'is it safe', 'safe to invest', 'too risky', 'low risk'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in safety_keywords)
    
    def _detect_position_sizing_intent(self, question: str) -> bool:
        """Detect if user is asking about how much to invest."""
        sizing_keywords = [
            'how much', 'how many', 'amount to invest', 'position size',
            'allocation', 'how much money', 'investment amount', 'dollar amount',
            'percentage', 'portion', 'share allocation'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in sizing_keywords)
    
    def _extract_multiple_tickers(self, question: str) -> List[str]:
        """Extract multiple ticker symbols from question."""
        # Extract all potential tickers
        pattern = r'\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, question.upper())
        
        # Filter out common words that might match
        common_words = {'THE', 'AND', 'OR', 'FOR', 'ARE', 'IS', 'IT', 'TO', 'IN', 'ON', 'AT'}
        tickers = [t for t in matches if t not in common_words and len(t) <= 5]
        
        return list(set(tickers))  # Remove duplicates
    
    def _compare_stocks(self, tickers: List[str]) -> Optional[Dict]:
        """Compare multiple stocks and provide recommendations."""
        if not self.services_available or len(tickers) < 2:
            return None
        
        comparison_results = []
        
        for ticker in tickers[:5]:  # Limit to 5 stocks
            try:
                # Get risk analysis
                risk_analysis = self.risk_service.analyze_risk(ticker, period="1y")
                metrics = risk_analysis.get('metrics', {})
                
                # Get price prediction
                try:
                    prediction = self.forecasting_service.predict(
                        symbol=ticker,
                        model_type="xgboost",
                        forecast_days=7,
                        use_lstm=False
                    )
                    avg_forecast = np.mean(prediction.get('forecasts', [0]))
                    current_price = None
                    try:
                        live_data = self.live_fetcher.get_live_price(ticker)
                        current_price = live_data.get('price')
                    except:
                        pass
                    
                    if current_price:
                        expected_return = ((avg_forecast - current_price) / current_price) * 100
                    else:
                        expected_return = 0
                except:
                    expected_return = 0
                
                # Calculate investment score
                volatility = metrics.get('volatility', 0) * 100
                sharpe = metrics.get('sharpe_ratio', 0)
                
                # Score calculation (higher is better)
                score = 0
                if sharpe > 2:
                    score += 30
                elif sharpe > 1:
                    score += 20
                elif sharpe > 0.5:
                    score += 10
                
                if volatility < 15:
                    score += 25  # Low risk bonus
                elif volatility < 30:
                    score += 15
                
                if expected_return > 5:
                    score += 25
                elif expected_return > 0:
                    score += 15
                elif expected_return > -5:
                    score += 5
                
                comparison_results.append({
                    'ticker': ticker,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'expected_return': expected_return,
                    'score': score,
                    'metrics': metrics
                })
            except Exception as e:
                logger.warning(f"Could not analyze {ticker}: {e}")
                continue
        
        if not comparison_results:
            return None
        
        # Sort by score (highest first)
        comparison_results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'stocks': comparison_results,
            'best_stock': comparison_results[0] if comparison_results else None
        }
    
    def _assess_investment_safety(self, ticker: str) -> Optional[Dict]:
        """Assess if an investment is safe."""
        if not self.services_available:
            return None
        
        try:
            # Get risk analysis
            risk_analysis = self.risk_service.analyze_risk(ticker, period="1y")
            metrics = risk_analysis.get('metrics', {})
            
            volatility = metrics.get('volatility', 0) * 100
            sharpe = metrics.get('sharpe_ratio', 0)
            var_95 = metrics.get('var_95', 0) * 100
            max_drawdown = metrics.get('max_drawdown', {}).get('max_drawdown', 0) * 100 if isinstance(metrics.get('max_drawdown'), dict) else 0
            
            # Safety score (0-100, higher is safer)
            safety_score = 50  # Start neutral
            
            # Adjust based on volatility
            if volatility < 15:
                safety_score += 25
            elif volatility < 25:
                safety_score += 10
            elif volatility > 40:
                safety_score -= 30
            
            # Adjust based on Sharpe ratio
            if sharpe > 1.5:
                safety_score += 20
            elif sharpe > 1:
                safety_score += 10
            elif sharpe < 0.5:
                safety_score -= 20
            
            # Adjust based on VaR
            if abs(var_95) < 5:
                safety_score += 10
            elif abs(var_95) > 15:
                safety_score -= 15
            
            # Adjust based on max drawdown
            if abs(max_drawdown) < 20:
                safety_score += 10
            elif abs(max_drawdown) > 40:
                safety_score -= 15
            
            # Clamp score
            safety_score = max(0, min(100, safety_score))
            
            # Determine safety level
            if safety_score >= 75:
                safety_level = "Very Safe"
                emoji = "✅"
            elif safety_score >= 60:
                safety_level = "Safe"
                emoji = "✅"
            elif safety_score >= 45:
                safety_level = "Moderate Risk"
                emoji = "⚠️"
            elif safety_score >= 30:
                safety_level = "Risky"
                emoji = "⚠️"
            else:
                safety_level = "Very Risky"
                emoji = "🔴"
            
            return {
                'ticker': ticker,
                'safety_score': safety_score,
                'safety_level': safety_level,
                'emoji': emoji,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Safety assessment error: {e}")
            return None
    
    def _get_position_sizing_advice(self, ticker: Optional[str] = None, portfolio_value: Optional[float] = None) -> str:
        """Provide position sizing recommendations."""
        import random
        
        advice_parts = []
        
        # General position sizing rules
        advice_parts.append("💡 **Position Sizing Guidelines:**\n")
        
        advice_parts.append("**1. The 5% Rule (Conservative):**")
        advice_parts.append("   • Invest no more than 5% of your portfolio in a single stock")
        advice_parts.append("   • This limits risk if one stock performs poorly")
        advice_parts.append("   • Example: If you have $10,000, invest max $500 per stock\n")
        
        advice_parts.append("**2. The 10% Rule (Moderate):**")
        advice_parts.append("   • For experienced investors, up to 10% per stock is acceptable")
        advice_parts.append("   • Still maintain diversification across 10+ stocks")
        advice_parts.append("   • Example: $10,000 portfolio = max $1,000 per stock\n")
        
        advice_parts.append("**3. Risk-Based Sizing:**")
        advice_parts.append("   • High volatility stocks (30%+): Reduce position size by 50%")
        advice_parts.append("   • Low volatility stocks (<15%): Can use full position size")
        advice_parts.append("   • Example: If normal size is $1,000, risky stock = $500\n")
        
        if ticker and self.services_available:
            try:
                safety = self._assess_investment_safety(ticker)
                if safety:
                    volatility = safety['volatility']
                    safety_level = safety['safety_level']
                    
                    advice_parts.append(f"\n**4. Specific Advice for {ticker}:**")
                    
                    if volatility > 30:
                        advice_parts.append(f"   • {ticker} has HIGH volatility ({volatility:.1f}%)")
                        advice_parts.append(f"   • Recommended position size: 2-3% of portfolio")
                        advice_parts.append(f"   • Consider smaller positions due to high risk")
                    elif volatility > 20:
                        advice_parts.append(f"   • {ticker} has MODERATE volatility ({volatility:.1f}%)")
                        advice_parts.append(f"   • Recommended position size: 3-5% of portfolio")
                        advice_parts.append(f"   • Standard position sizing applies")
                    else:
                        advice_parts.append(f"   • {ticker} has LOW volatility ({volatility:.1f}%)")
                        advice_parts.append(f"   • Recommended position size: 5-7% of portfolio")
                        advice_parts.append(f"   • Can use larger positions due to lower risk")
                    
                    advice_parts.append(f"   • Safety Level: {safety['emoji']} {safety_level}")
            except:
                pass
        
        advice_parts.append("\n**5. Portfolio Allocation Strategy:**")
        advice_parts.append("   • 60% in stocks (diversified)")
        advice_parts.append("   • 30% in bonds/stable assets")
        advice_parts.append("   • 10% in cash (for opportunities/emergencies)")
        advice_parts.append("   • Within stocks: Spread across different sectors\n")
        
        advice_parts.append("**6. Dollar-Cost Averaging:**")
        advice_parts.append("   • Don't invest everything at once")
        advice_parts.append("   • Invest fixed amounts regularly (weekly/monthly)")
        advice_parts.append("   • Reduces timing risk and emotional decisions\n")
        
        advice_parts.append("**⚠️ Important Reminders:**")
        advice_parts.append("   • Never invest more than you can afford to lose")
        advice_parts.append("   • Start small and increase as you learn")
        advice_parts.append("   • Diversification is your best friend")
        advice_parts.append("   • Review and rebalance quarterly")
        
        return "\n".join(advice_parts)
    
    def _get_price_prediction(self, ticker: str, forecast_days: int = 7, model_type: str = "xgboost") -> Optional[Dict]:
        """Get price prediction using forecasting service."""
        if not self.services_available:
            return None
        
        try:
            # Try to get prediction
            prediction = self.forecasting_service.predict(
                symbol=ticker,
                model_type=model_type,
                forecast_days=forecast_days,
                use_lstm=False
            )
            
            # Get current price for context
            try:
                live_data = self.live_fetcher.get_live_price(ticker)
                current_price = live_data.get('price', None)
            except:
                # Fallback: try to get from historical data
                try:
                    data = self.data_ingestion.fetch_historical_data(ticker, period="1mo")
                    current_price = data['close'].iloc[-1] if not data.empty else None
                except:
                    current_price = None
            
            # Get risk metrics for context
            try:
                risk_analysis = self.risk_service.analyze_risk(ticker, period="1y")
            except:
                risk_analysis = None
            
            return {
                'forecasts': prediction.get('forecasts', []),
                'current_price': current_price,
                'ticker': ticker,
                'forecast_days': forecast_days,
                'model_type': prediction.get('model_type', model_type),
                'risk_metrics': risk_analysis
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def _format_prediction_response(self, prediction_data: Dict, question: str) -> str:
        """Format price prediction into creative, helpful response."""
        ticker = prediction_data['ticker']
        forecasts = prediction_data['forecasts']
        current_price = prediction_data.get('current_price')
        forecast_days = prediction_data['forecast_days']
        risk_metrics = prediction_data.get('risk_metrics', {})
        
        # Select creative intro
        import random
        intro = random.choice(self.advice_templates['price_prediction_intro']).format(ticker=ticker)
        
        response_parts = [intro]
        
        # Current price context
        if current_price:
            response_parts.append(f"\n📊 Current Price: ${current_price:.2f}")
        
        # Predictions
        if forecasts:
            avg_forecast = np.mean(forecasts)
            max_forecast = np.max(forecasts)
            min_forecast = np.min(forecasts)
            change_pct = ((avg_forecast - current_price) / current_price * 100) if current_price else 0
            
            response_parts.append(f"\n🔮 Price Predictions for Next {forecast_days} Days:")
            response_parts.append(f"   • Average Forecast: ${avg_forecast:.2f}")
            response_parts.append(f"   • Expected Range: ${min_forecast:.2f} - ${max_forecast:.2f}")
            
            if current_price:
                if change_pct > 5:
                    response_parts.append(f"   • Expected Change: +{change_pct:.1f}% 📈 (Bullish!)")
                elif change_pct < -5:
                    response_parts.append(f"   • Expected Change: {change_pct:.1f}% 📉 (Bearish)")
                else:
                    response_parts.append(f"   • Expected Change: {change_pct:+.1f}% ➡️ (Neutral)")
            
            # Add risk context
            if risk_metrics:
                volatility = risk_metrics.get('volatility', 0)
                sharpe = risk_metrics.get('sharpe_ratio', 0)
                
                response_parts.append(f"\n📊 Risk Analysis:")
                if volatility:
                    vol_pct = volatility * 100
                    if vol_pct > 30:
                        response_parts.append(f"   • Volatility: {vol_pct:.1f}% ⚠️ (High - expect price swings)")
                    elif vol_pct > 15:
                        response_parts.append(f"   • Volatility: {vol_pct:.1f}% ⚡ (Moderate)")
                    else:
                        response_parts.append(f"   • Volatility: {vol_pct:.1f}% ✅ (Low - relatively stable)")
                
                if sharpe:
                    if sharpe > 2:
                        response_parts.append(f"   • Sharpe Ratio: {sharpe:.2f} ⭐⭐⭐ (Excellent risk-adjusted returns!)")
                    elif sharpe > 1:
                        response_parts.append(f"   • Sharpe Ratio: {sharpe:.2f} ⭐⭐ (Good risk-adjusted returns)")
                    else:
                        response_parts.append(f"   • Sharpe Ratio: {sharpe:.2f} ⚠️ (Could be better)")
            
            # Investment advice based on prediction
            response_parts.append(f"\n💡 My Recommendation:")
            if change_pct > 5:
                response_parts.append(f"   The models suggest {ticker} has upward potential! However, remember:")
                response_parts.append(f"   • Consider dollar-cost averaging (don't invest everything at once)")
                response_parts.append(f"   • Set stop-loss orders to protect your investment")
                response_parts.append(f"   • High volatility means prices can swing - be prepared!")
            elif change_pct < -5:
                response_parts.append(f"   The models suggest caution with {ticker}. Consider:")
                response_parts.append(f"   • Waiting for a better entry point")
                response_parts.append(f"   • If you own it, consider setting stop-loss orders")
                response_parts.append(f"   • This might be a buying opportunity if you believe in long-term value")
            else:
                response_parts.append(f"   {ticker} appears to be in a neutral trend. Consider:")
                response_parts.append(f"   • This might be good for range trading (buy low, sell high)")
                response_parts.append(f"   • Long-term investors can use this as an accumulation phase")
                response_parts.append(f"   • Watch for breakout signals above resistance or below support")
        
        # Add disclaimer
        response_parts.append(f"\n{random.choice(self.advice_templates['risk_warning'])}")
        
        return "\n".join(response_parts)
    
    def _format_stock_comparison_response(self, comparison: Dict) -> str:
        """Format stock comparison into creative response."""
        import random
        
        stocks = comparison['stocks']
        best = comparison['best_stock']
        
        response_parts = []
        response_parts.append("📊 **Stock Comparison Analysis**\n")
        response_parts.append(f"I've analyzed {len(stocks)} stocks for you. Here's my comparison:\n")
        
        # Show all stocks ranked
        response_parts.append("**Ranking (Best to Worst):**\n")
        for i, stock in enumerate(stocks, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            response_parts.append(f"{medal} **{stock['ticker']}** (Score: {stock['score']}/100)")
            response_parts.append(f"   • Volatility: {stock['volatility']:.1f}%")
            response_parts.append(f"   • Sharpe Ratio: {stock['sharpe_ratio']:.2f}")
            response_parts.append(f"   • Expected Return: {stock['expected_return']:+.1f}%")
            response_parts.append("")
        
        # Best stock recommendation
        if best:
            response_parts.append(f"**🏆 My Top Recommendation: {best['ticker']}**\n")
            response_parts.append(f"**Why {best['ticker']} is the best choice:**")
            
            if best['sharpe_ratio'] > 1.5:
                response_parts.append(f"   ✅ Excellent risk-adjusted returns (Sharpe: {best['sharpe_ratio']:.2f})")
            if best['volatility'] < 20:
                response_parts.append(f"   ✅ Low volatility ({best['volatility']:.1f}%) - more stable")
            if best['expected_return'] > 0:
                response_parts.append(f"   ✅ Positive expected returns ({best['expected_return']:+.1f}%)")
            
            response_parts.append(f"\n**Investment Strategy for {best['ticker']}:**")
            if best['volatility'] > 30:
                response_parts.append("   • Use smaller position size (2-3% of portfolio) due to high volatility")
            elif best['volatility'] > 20:
                response_parts.append("   • Standard position size (3-5% of portfolio)")
            else:
                response_parts.append("   • Can use larger position size (5-7% of portfolio) due to lower risk")
            
            response_parts.append("   • Consider dollar-cost averaging (invest gradually)")
            response_parts.append("   • Set stop-loss orders to protect your investment")
        
        response_parts.append(f"\n{random.choice(self.advice_templates['risk_warning'])}")
        
        return "\n".join(response_parts)
    
    def _format_single_stock_recommendation(self, ticker: str) -> str:
        """Format single stock recommendation."""
        import random
        
        if not self.services_available:
            return f"I'd love to recommend {ticker}, but the analysis services aren't available right now."
        
        try:
            # Get comprehensive analysis
            risk_analysis = self.risk_service.analyze_risk(ticker, period="1y")
            metrics = risk_analysis.get('metrics', {})
            
            prediction = self._get_price_prediction(ticker, 7)
            safety = self._assess_investment_safety(ticker)
            
            response_parts = []
            response_parts.append(f"📊 **Investment Recommendation for {ticker}**\n")
            
            # Safety assessment
            if safety:
                response_parts.append(f"**Safety Level:** {safety['emoji']} {safety['safety_level']} ({safety['safety_score']}/100)\n")
            
            # Risk metrics
            volatility = metrics.get('volatility', 0) * 100
            sharpe = metrics.get('sharpe_ratio', 0)
            
            response_parts.append("**Risk Analysis:**")
            response_parts.append(f"   • Volatility: {volatility:.1f}%")
            response_parts.append(f"   • Sharpe Ratio: {sharpe:.2f}")
            
            # Prediction
            if prediction and prediction.get('forecasts'):
                avg_forecast = np.mean(prediction['forecasts'])
                current_price = prediction.get('current_price')
                if current_price:
                    change_pct = ((avg_forecast - current_price) / current_price) * 100
                    response_parts.append(f"   • Expected Return (7 days): {change_pct:+.1f}%")
            
            # Recommendation
            response_parts.append("\n**💡 My Recommendation:**")

            recommendation = "HOLD"
            if safety and safety['safety_score'] >= 60:
                recommendation = "BUY"
                response_parts.append(f"   ✅ {ticker} appears to be a relatively safe investment")
                response_parts.append(f"   • Good for conservative to moderate risk investors")
                response_parts.append(f"   • Consider position size: 5-7% of portfolio")
            elif safety and safety['safety_score'] >= 45:
                recommendation = "HOLD"
                response_parts.append(f"   ⚠️ {ticker} has moderate risk")
                response_parts.append(f"   • Suitable for moderate risk investors")
                response_parts.append(f"   • Consider position size: 3-5% of portfolio")
            else:
                recommendation = "SELL"
                response_parts.append(f"   🔴 {ticker} is a high-risk investment")
                response_parts.append(f"   • Only for aggressive investors who can handle volatility")
                response_parts.append(f"   • Consider position size: 2-3% of portfolio")

            confidence = safety['safety_score'] / 10 if safety else 5.0
            response_parts.append(f"\n**Decision:** `{recommendation}`")
            response_parts.append(f"**Confidence:** `{confidence:.1f}/10`")
            
            response_parts.append(f"   • Use dollar-cost averaging")
            response_parts.append(f"   • Set stop-loss orders")
            response_parts.append(f"   • Monitor regularly")
            
            response_parts.append(f"\n{random.choice(self.advice_templates['risk_warning'])}")
            
            return "\n".join(response_parts)
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return f"I tried to analyze {ticker}, but encountered an issue. Make sure the ticker is valid and models are trained."
    
    def _format_safety_response(self, safety: Dict) -> str:
        """Format safety assessment into creative response."""
        import random
        
        ticker = safety['ticker']
        safety_level = safety['safety_level']
        safety_score = safety['safety_score']
        emoji = safety['emoji']
        volatility = safety['volatility']
        sharpe = safety['sharpe_ratio']
        var_95 = safety['var_95']
        
        response_parts = []
        response_parts.append(f"🛡️ **Investment Safety Assessment for {ticker}**\n")
        response_parts.append(f"**Overall Safety:** {emoji} **{safety_level}** (Score: {safety_score}/100)\n")
        
        response_parts.append("**Detailed Analysis:**")
        response_parts.append(f"   • Volatility: {volatility:.1f}%")
        if volatility < 15:
            response_parts.append("      ✅ Low volatility - relatively stable")
        elif volatility < 30:
            response_parts.append("      ⚡ Moderate volatility - expect some swings")
        else:
            response_parts.append("      ⚠️ High volatility - significant price swings expected")
        
        response_parts.append(f"   • Sharpe Ratio: {sharpe:.2f}")
        if sharpe > 2:
            response_parts.append("      ⭐⭐⭐ Excellent risk-adjusted returns")
        elif sharpe > 1:
            response_parts.append("      ⭐⭐ Good risk-adjusted returns")
        else:
            response_parts.append("      ⚠️ Could be better - high risk for returns")
        
        response_parts.append(f"   • VaR (95%): {var_95:.1f}%")
        response_parts.append("      (95% chance losses won't exceed this amount)")
        
        # Safety verdict
        response_parts.append(f"\n**💡 Verdict:**")
        if safety_score >= 75:
            response_parts.append(f"   ✅ **YES, {ticker} is relatively safe to invest in**")
            response_parts.append(f"   • Suitable for conservative investors")
            response_parts.append(f"   • Low to moderate risk profile")
            response_parts.append(f"   • Good for long-term investment")
        elif safety_score >= 60:
            response_parts.append(f"   ✅ **{ticker} is reasonably safe**")
            response_parts.append(f"   • Suitable for moderate risk investors")
            response_parts.append(f"   • Some volatility expected")
            response_parts.append(f"   • Diversify your portfolio")
        elif safety_score >= 45:
            response_parts.append(f"   ⚠️ **{ticker} has moderate risk**")
            response_parts.append(f"   • Not for conservative investors")
            response_parts.append(f"   • Higher volatility expected")
            response_parts.append(f"   • Use smaller position sizes")
        elif safety_score >= 30:
            response_parts.append(f"   ⚠️ **{ticker} is risky**")
            response_parts.append(f"   • Only for aggressive investors")
            response_parts.append(f"   • High volatility and risk")
            response_parts.append(f"   • Use very small position sizes (2-3%)")
        else:
            response_parts.append(f"   🔴 **{ticker} is very risky**")
            response_parts.append(f"   • Not recommended for most investors")
            response_parts.append(f"   • Very high volatility")
            response_parts.append(f"   • Only invest what you can afford to lose")
        
        response_parts.append(f"\n**📋 Risk Management Tips:**")
        response_parts.append(f"   • Position size: {max(2, min(7, int(safety_score/10)))}% of portfolio")
        response_parts.append(f"   • Use stop-loss orders")
        response_parts.append(f"   • Don't invest more than you can afford to lose")
        response_parts.append(f"   • Monitor regularly")
        
        response_parts.append(f"\n{random.choice(self.advice_templates['risk_warning'])}")
        
        return "\n".join(response_parts)
    
    def _generate_creative_response(self, question: str, keywords: List[str], context: Optional[Dict] = None) -> str:
        """Generate creative, personalized financial advice."""
        response_parts = []
        
        # Add relevant knowledge with creative formatting
        for keyword in keywords[:3]:  # Top 3 topics
            if keyword in self.knowledge_base:
                response_parts.append(self.knowledge_base[keyword].strip())
        
        # Add context-specific insights
        if context:
            for ticker, indicators in context.items():
                context_info = f"\n\n📊 Personalized Insights for {ticker}:"
                if 'volatility' in indicators and indicators['volatility']:
                    vol = indicators['volatility']
                    vol_pct = vol * 100
                    if vol_pct > 30:
                        context_info += f"\n   • Volatility: {vol_pct:.1f}% - This is a roller coaster! 🎢"
                    elif vol_pct > 15:
                        context_info += f"\n   • Volatility: {vol_pct:.1f}% - Moderate swings expected ⚡"
                    else:
                        context_info += f"\n   • Volatility: {vol_pct:.1f}% - Relatively stable ✅"
                
                if 'sharpe_ratio' in indicators and indicators['sharpe_ratio']:
                    sharpe = indicators['sharpe_ratio']
                    if sharpe > 2:
                        context_info += f"\n   • Sharpe Ratio: {sharpe:.2f} - Excellent! You're getting great returns for the risk ⭐⭐⭐"
                    elif sharpe > 1:
                        context_info += f"\n   • Sharpe Ratio: {sharpe:.2f} - Good risk-adjusted performance ⭐⭐"
                    else:
                        context_info += f"\n   • Sharpe Ratio: {sharpe:.2f} - Room for improvement ⚠️"
                
                if 'trend_direction' in indicators:
                    trend = indicators['trend_direction']
                    if trend.lower() == 'upward':
                        context_info += f"\n   • Trend: 📈 Upward - Momentum is positive!"
                    elif trend.lower() == 'downward':
                        context_info += f"\n   • Trend: 📉 Downward - Be cautious"
                    else:
                        context_info += f"\n   • Trend: ➡️ Sideways - Range-bound movement"
                
                response_parts.append(context_info)
        
        # If no keywords found, provide helpful general response
        if not response_parts:
            response_parts.append("""
            👋 Hi! I'm your personal finance advisor. I can help you with:
            
            💰 Price Predictions: Ask me "What will AAPL be next week?" or "Predict TSLA price"
            📊 Financial Concepts: Ask about volatility, Sharpe ratio, RSI, MACD, Beta, etc.
            💡 Investment Advice: Get personalized recommendations based on risk analysis
            📈 Market Analysis: Understand trends, risk metrics, and portfolio strategies
            🏆 Stock Comparison: Compare stocks to find the best investment
            🛡️ Safety Assessment: Check if an investment is safe
            💵 Position Sizing: Learn how much to invest
            
            Try asking:
            - "What is volatility?"
            - "Predict AAPL price for next 7 days"
            - "Which is better: AAPL or TSLA?"
            - "Is AAPL safe to invest?"
            - "How much should I invest in MSFT?"
            - "What's the best stock to invest in?"
            - "Explain Sharpe ratio"
            - "What's a good investment strategy?"
            """)
        
        return "\n\n".join(response_parts)
    
    def chat(
        self,
        question: str,
        ticker: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Chat with the personal finance advisor.
        
        Args:
            question: User question
            ticker: Optional ticker symbol
            context: Optional financial context/indicators
        
        Returns:
            Creative, personalized financial advice
        """
        question_lower = question.lower()
        
        # Check for greeting
        if any(word in question_lower for word in ['hi', 'hello', 'hey', 'greetings']):
            import random
            return random.choice(self.advice_templates['greeting'])
        
        # Check for position sizing question
        if self._detect_position_sizing_intent(question):
            extracted_ticker = self._extract_ticker(question) if not ticker else ticker
            return self._get_position_sizing_advice(ticker=extracted_ticker)
        
        # Check for stock comparison question
        if self._detect_stock_comparison_intent(question):
            tickers = self._extract_multiple_tickers(question)
            if not ticker and not tickers:
                ticker = self._extract_ticker(question)
                if ticker:
                    tickers = [ticker]
            
            if len(tickers) >= 2:
                # Compare multiple stocks
                comparison = self._compare_stocks(tickers)
                if comparison:
                    return self._format_stock_comparison_response(comparison)
                else:
                    return f"I tried to compare {', '.join(tickers)}, but encountered an issue. Make sure the tickers are valid and models are trained."
            elif len(tickers) == 1:
                # Single stock recommendation
                return self._format_single_stock_recommendation(tickers[0])
            else:
                return "I'd be happy to help you find the best stock to invest in! Please mention ticker symbols (like AAPL, TSLA, MSFT) in your question. For example: 'Which is better: AAPL or TSLA?' or 'What's the best stock to invest in?'"
        
        # Check for safety assessment question
        if self._detect_safety_intent(question):
            extracted_ticker = self._extract_ticker(question) if not ticker else ticker
            if extracted_ticker:
                safety = self._assess_investment_safety(extracted_ticker)
                if safety:
                    return self._format_safety_response(safety)
                else:
                    return f"I tried to assess {extracted_ticker}'s safety, but encountered an issue. Make sure the ticker is valid and models are trained."
            else:
                return "I'd be happy to assess investment safety! Please mention a ticker symbol (like AAPL, TSLA, MSFT) in your question. For example: 'Is AAPL safe to invest?' or 'Is it safe to invest in TSLA?'"
        
        # Check for price prediction request
        if self._detect_prediction_intent(question):
            # Extract ticker if not provided
            if not ticker:
                ticker = self._extract_ticker(question)
            
            if ticker:
                forecast_days = self._extract_forecast_days(question)
                
                if self.services_available:
                    prediction_data = self._get_price_prediction(ticker, forecast_days)
                    if prediction_data:
                        return self._format_prediction_response(prediction_data, question)
                    else:
                        return f"I tried to predict {ticker}'s price, but encountered an issue. Make sure the ticker is valid and models are trained. You can train models using the 'Model Training' page in the dashboard."
                else:
                    return f"I'd love to predict {ticker}'s price, but the prediction services aren't available right now. Please ensure all dependencies are installed and services are running."
            else:
                return "I'd be happy to predict a stock price! Please mention the ticker symbol (like AAPL, TSLA, MSFT) in your question. For example: 'Predict AAPL price for next 7 days' or 'What will TSLA be next week?'"
        
        # Handle general financial questions
        keywords = self._find_keywords(question)
        response = self._generate_creative_response(question, keywords, context)
        
        logger.info(f"Answered question with keywords: {keywords}")
        return response
    
    def _find_keywords(self, question: str) -> List[str]:
        """Extract keywords from question."""
        question_lower = question.lower()
        keywords = []
        
        for key in self.knowledge_base.keys():
            if key in question_lower:
                keywords.append(key)
        
        # Check for variations
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
            "strategy": "investment",
            "invest": "investment",
            "should i buy": "investment",
            "should i invest": "investment"
        }
        
        for variation, key in variations.items():
            if variation in question_lower and key not in keywords:
                keywords.append(key)
        
        return keywords
    
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
            Dictionary matching chatbot response format
        """
        answer = self.chat(question, ticker=ticker, context=None)
        
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
                    # Provide full content so the UI can render complete sources
                    "content": self.knowledge_base.get(keyword, ""),
                    "source": f"knowledge_base_{keyword}",
                    "type": "financial_knowledge"
                }
                for keyword in keywords[:k]
            ]
        
        return result


# Compatibility alias
FinancialRAGChatbot = PersonalFinanceAdvisor  # For easy switching


