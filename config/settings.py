"""
Configuration settings for FinSense Financial Intelligence System.
"""

from typing import Optional, List
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # 🔑 API Keys
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(None, env="GROQ_API_KEY")
    LLM_PROVIDER: str = Field("groq", env="LLM_PROVIDER")
    GROQ_MODEL: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    OPENAI_MODEL: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")

    # 📦 Model paths
    ML_MODEL_PATH: str = str(BASE_DIR / "models" / "ml" / "latest_model.pkl")
    LSTM_MODEL_PATH: str = str(BASE_DIR / "models" / "lstm" / "latest_model.h5")

    # 🌐 Ports
    STREAMLIT_PORT: int = 8501
    BACKEND_PORT: int = 8000

    # 🛠 App configs
    APP_NAME: str = "FinSense"
    ENVIRONMENT: str = "development"

    # 📊 Data Sources
    YAHOO_FINANCE_ENABLED: bool = True

    # 🧠 Model Settings
    LSTM_SEQUENCE_LENGTH: int = 60
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 32
    LSTM_VALIDATION_SPLIT: float = 0.2
    ML_TRAIN_TEST_SPLIT: float = 0.2

    # 🧪 Feature Engineering
    FEATURE_WINDOW_SIZE: int = 20
    TECHNICAL_INDICATORS: List[str] = [
        "SMA", "EMA", "RSI", "MACD", "Bollinger_Bands",
        "ATR", "OBV", "Stochastic", "ADX"
    ]

    # 📉 Risk Metrics
    RISK_FREE_RATE: float = 0.02
    CONFIDENCE_LEVEL: float = 0.95

    # 📝 Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

    # 📁 Directories
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


# ✅ Singleton instance
settings = Settings()

# ✅ Create required directories
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
