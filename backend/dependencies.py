"""
Shared dependencies for FastAPI routes.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from typing import Optional

# Import services
try:
    from models.ml.train_ml_model import MLTrainingPipeline
    from models.lstm.train_lstm_model import LSTMTrainingPipeline
    from risk_analysis.risk_engine import analyze_risk, calculate_volatility, calculate_sharpe_ratio, calculate_trend_direction
    from data_ingestion.historical_data import HistoricalDataDownloader
    from data_ingestion.live_data import LiveDataFetcher
    from chatbot.financial_rag import FinancialRAGChatbot
    from feature_engineering.financial_features import FinancialFeatureEngineer
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    import logging
    logging.warning(f"Some services not available: {str(e)}")


class ServiceContainer:
    """Container for shared services."""
    
    def __init__(self):
        """Initialize services."""
        self.ml_pipeline = None
        self.lstm_pipeline = None
        self.data_downloader = None
        self.live_fetcher = None
        self.rag_chatbot = None
        self.feature_engineer = None
        
        if SERVICES_AVAILABLE:
            self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all services."""
        try:
            self.ml_pipeline = MLTrainingPipeline(model_dir="models/ml")
            self.lstm_pipeline = LSTMTrainingPipeline(model_dir="models/lstm")
            self.data_downloader = HistoricalDataDownloader()
            self.live_fetcher = LiveDataFetcher()
            self.feature_engineer = FinancialFeatureEngineer()
            
            # Initialize RAG chatbot if Groq/OpenAI API key is available
            groq_api_key = os.getenv("GROQ_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            llm_provider = os.getenv("LLM_PROVIDER")
            if groq_api_key or openai_api_key:
                try:
                    self.rag_chatbot = FinancialRAGChatbot(
                        openai_api_key=openai_api_key,
                        groq_api_key=groq_api_key,
                        provider=llm_provider
                    )
                    self.rag_chatbot.initialize_with_default_knowledge()
                except Exception as e:
                    import logging
                    logging.warning(f"RAG chatbot not available: {str(e)}")
        except Exception as e:
            import logging
            logging.error(f"Error initializing services: {str(e)}")
    
    def get_ml_pipeline(self) -> Optional[MLTrainingPipeline]:
        """Get ML training pipeline."""
        return self.ml_pipeline
    
    def get_lstm_pipeline(self) -> Optional[LSTMTrainingPipeline]:
        """Get LSTM training pipeline."""
        return self.lstm_pipeline
    
    def get_data_downloader(self) -> Optional[HistoricalDataDownloader]:
        """Get data downloader."""
        return self.data_downloader
    
    def get_live_fetcher(self) -> Optional[LiveDataFetcher]:
        """Get live data fetcher."""
        return self.live_fetcher
    
    def get_rag_chatbot(self) -> Optional[FinancialRAGChatbot]:
        """Get RAG chatbot."""
        return self.rag_chatbot
    
    def get_feature_engineer(self) -> Optional[FinancialFeatureEngineer]:
        """Get feature engineer."""
        return self.feature_engineer


# Global service container
services = ServiceContainer()













