"""
Streamlit dashboard for FinSense Financial Intelligence System.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from dotenv import load_dotenv
# OPENAI_API_KEY must be set via environment or .env — do not hardcode secrets here.

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from src.data.data_ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineering
from src.services.forecasting_service import ForecastingService
from src.services.risk_service import RiskService
from src.models.ml_models import MLForecaster
from src.models.lstm_model import LSTMForecaster
from src.risk.risk_metrics import RiskMetrics
from data_ingestion.live_data import LiveDataFetcher
from chatbot.financial_rag import FinancialRAGChatbot
from chatbot.personal_finance_advisor import PersonalFinanceAdvisor
from config import settings

# Page configuration
st.set_page_config(
    page_title="FinSense - Financial Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global styling for a cleaner UI
st.markdown(
    """
    <style>
    .main { background: linear-gradient(180deg, #0b1220 0%, #111827 100%); }
    .block-container { padding-top: 1.2rem; }
    h1, h2, h3, .stMarkdown, .stTextInput label, .stSelectbox label, .stSlider label {
        color: #e5e7eb !important;
    }
    .finsense-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(17,24,39,0.92));
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .finsense-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #93c5fd;
        margin-bottom: 0.3rem;
    }
    .finsense-sub {
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache services."""
    return {
        'data_ingestion': DataIngestion(),
        'feature_engineering': FeatureEngineering(),
        'forecasting_service': ForecastingService(),
        'risk_service': RiskService(),
        'risk_metrics': RiskMetrics(),
        'live_fetcher': LiveDataFetcher(),
        # Lazy initialize chatbot later to avoid blocking initial screen load.
        'rag_chatbot': None
    }


@st.cache_resource
def get_chatbot():
    """Lazily initialize chatbot to keep UI responsive."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm_provider = os.getenv("LLM_PROVIDER")

        if groq_api_key:
            bot = FinancialRAGChatbot(
                groq_api_key=groq_api_key,
                provider=llm_provider or "groq",
                model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            )
            bot.initialize_with_default_knowledge()
            return bot
        if openai_api_key:
            bot = FinancialRAGChatbot(
                openai_api_key=openai_api_key,
                provider=llm_provider or "openai",
                model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            )
            bot.initialize_with_default_knowledge()
            return bot
    except Exception:
        pass

    # Fallback to local chatbots if LLM init fails
    try:
        return PersonalFinanceAdvisor()
    except Exception:
        try:
            from chatbot.free_chatbot import FreeFinancialChatbot
            return FreeFinancialChatbot()
        except Exception:
            return None

services = get_services()


def _safe_metric_value(info: dict, key: str, default: float = 0.0) -> float:
    """Safely read numeric metric values from market info payloads."""
    if not isinstance(info, dict):
        return default
    value = info.get(key, default)
    return value if isinstance(value, (int, float)) and value is not None else default

# Sidebar
st.sidebar.title("📈 FinSense")
st.sidebar.markdown("### Financial Intelligence System")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Dashboard",
        "Data Explorer",
        "Model Training",
        "Forecasting",
        "Risk Analysis",
        "Feature Importance",
        "Live & Chatbot"
    ]
)

# Main content
if page == "Dashboard":
    st.title("📊 Financial Intelligence Dashboard")
    st.markdown("---")
    
    # Symbol input
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                data = services['data_ingestion'].fetch_historical_data(symbol=symbol, period=period)
                if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                    raise ValueError("No data returned from data provider. Try another symbol or period.")
                if 'close' not in data.columns or 'volume' not in data.columns:
                    raise ValueError("Returned data is missing required price columns (close/volume).")
                
                # Display basic info
                info = services['data_ingestion'].get_market_info(symbol)
                if not isinstance(info, dict):
                    info = {}
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${_safe_metric_value(info, 'current_price'):.2f}")
                with col2:
                    st.metric("52W High", f"${_safe_metric_value(info, '52_week_high'):.2f}")
                with col3:
                    st.metric("52W Low", f"${_safe_metric_value(info, '52_week_low'):.2f}")
                with col4:
                    st.metric("Market Cap", f"${_safe_metric_value(info, 'market_cap'):,.0f}")
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title=f"{symbol} Stock Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color='rgba(31, 119, 180, 0.5)'
                ))
                fig_vol.update_layout(
                    title=f"{symbol} Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=300
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
                st.session_state['data'] = data
                st.session_state['symbol'] = symbol
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

elif page == "Data Explorer":
    st.title("🔍 Data Explorer")
    st.markdown("---")
    
    symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    if st.button("Load and Engineer Features"):
        with st.spinner("Processing data..."):
            try:
                # Fetch data
                data = services['data_ingestion'].fetch_historical_data(symbol=symbol, period=period)
                
                # Engineer features
                features = services['feature_engineering'].engineer_features(data)
                
                st.success(f"Generated {len(features.columns)} features")
                
                # Display data
                st.subheader("Engineered Features")
                st.dataframe(features.tail(20), use_container_width=True)
                
                # Feature statistics
                st.subheader("Feature Statistics")
                st.dataframe(features.describe(), use_container_width=True)
                
                st.session_state['features'] = features
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif page == "Model Training":
    st.title("🤖 Model Training")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
        model_type = st.selectbox(
            "Model Type",
            ["xgboost", "random_forest", "extra_trees", "gradient_boosting"],
            index=0
        )
    
    with col2:
        train_lstm = st.checkbox("Train LSTM Model", value=False)
        adaptive_mode = st.checkbox("Adaptive Mode (AEFDS)", value=True)
        period = st.selectbox("Training Period", ["6mo", "1y", "2y", "5y"], index=2)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                if train_lstm:
                    result = services['forecasting_service'].train_lstm_model(
                        symbol=symbol,
                        period=period
                    )
                    st.success("LSTM Model Trained Successfully!")
                    
                    metrics = result['metrics']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Test RMSE", f"${metrics['test_rmse']:.2f}")
                    with col2:
                        st.metric("Test MAE", f"${metrics['test_mae']:.2f}")
                    with col3:
                        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
                    with col4:
                        st.metric("Train R²", f"{metrics['train_r2']:.4f}")
                    
                    # Training history
                    if 'history' in metrics:
                        history = metrics['history']
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history['loss'],
                            mode='lines',
                            name='Training Loss'
                        ))
                        fig.add_trace(go.Scatter(
                            y=history['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))
                        fig.update_layout(
                            title="Training History",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    result = services['forecasting_service'].train_ml_model(
                        symbol=symbol,
                        model_type=model_type,
                        adaptive_mode=adaptive_mode,
                        period=period
                    )
                    selected_model = result.get("selected_model", model_type)
                    st.success(f"{selected_model.upper()} Model Trained Successfully!")
                    if adaptive_mode:
                        st.caption(f"Adaptive selector chose: `{selected_model}`")
                    
                    metrics = result['metrics']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Test RMSE", f"${metrics['test_rmse']:.2f}")
                    with col2:
                        st.metric("Test MAE", f"${metrics['test_mae']:.2f}")
                    with col3:
                        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
                    with col4:
                        st.metric("Train R²", f"{metrics['train_r2']:.4f}")
                    
                    # Feature importance
                    if 'feature_importance' in result:
                        importance_df = pd.DataFrame(result['feature_importance'])
                        fig = px.bar(
                            importance_df.head(15),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 15 Feature Importance"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")

elif page == "Forecasting":
    st.title("🔮 Price Forecasting")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["xgboost", "random_forest", "extra_trees", "gradient_boosting"],
            index=0
        )
    with col3:
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
    
    use_lstm = st.checkbox("Use LSTM Model", value=False)
    adaptive_mode_forecast = st.checkbox("Adaptive Mode (AEFDS)", value=True)
    sentiment_input = st.selectbox("Sentiment", ["bearish", "neutral", "bullish"], index=1)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                result = services['forecasting_service'].predict(
                    symbol=symbol,
                    model_type=model_type,
                    forecast_days=forecast_days,
                    use_lstm=use_lstm,
                    adaptive_mode=adaptive_mode_forecast,
                    sentiment=sentiment_input
                )
                
                # Get historical data for context
                historical = services['data_ingestion'].fetch_historical_data(
                    symbol=symbol,
                    period="3mo"
                )
                
                # Create forecast dates
                last_date = historical.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                
                # Plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical.index[-30:],
                    y=historical['close'].tail(30),
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=result['forecasts'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                selected_model = result.get("selected_model", result.get("model_type", model_type))
                st.info(f"Selected model: **{selected_model}**")
                if result.get("decision"):
                    decision = result["decision"]
                    signal = decision.get("decision", "HOLD")
                    color = "#16a34a" if signal == "BUY" else ("#dc2626" if signal == "SELL" else "#ca8a04")
                    st.markdown(
                        f"<div style='padding:10px 12px;border-radius:10px;border:1px solid #334155;background:#0f172a;'>"
                        f"<strong>Decision:</strong> <span style='color:{color};font-weight:700'>{signal}</span> "
                        f"| Confidence: {decision.get('confidence', 0)}% "
                        f"| Risk: {decision.get('risk_level', 'Medium')}</div>",
                        unsafe_allow_html=True
                    )
                
                # Forecast table
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Price': result['forecasts']
                })
                st.dataframe(forecast_df, use_container_width=True)
                st.session_state["latest_forecast_result"] = result
                
            except Exception as e:
                st.error(f"Forecast error: {str(e)}")

elif page == "Risk Analysis":
    st.title("⚠️ Risk Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
    with col2:
        benchmark = st.text_input("Benchmark Symbol (optional)", value="^GSPC", placeholder="e.g., ^GSPC for S&P 500")
    
    period = st.selectbox("Analysis Period", ["3mo", "6mo", "1y", "2y"], index=2)
    
    if st.button("Analyze Risk", type="primary"):
        with st.spinner("Analyzing risk..."):
            try:
                result = services['risk_service'].analyze_risk(
                    symbol=symbol,
                    benchmark_symbol=benchmark if benchmark else None,
                    period=period
                )
                
                metrics = result['metrics']
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Volatility", f"{metrics['volatility']:.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("VaR (95%)", f"{metrics['var_95']:.2%}")
                with col4:
                    st.metric("CVaR (95%)", f"{metrics['cvar_95']:.2%}")
                
                # Drawdown
                st.subheader("Maximum Drawdown")
                drawdown_info = metrics['max_drawdown']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Drawdown", f"{drawdown_info['max_drawdown']:.2%}")
                with col2:
                    st.metric("Drawdown Date", str(drawdown_info['max_drawdown_date']).split()[0])
                with col3:
                    st.metric("Current Drawdown", f"{drawdown_info['current_drawdown']:.2%}")
                
                # Additional metrics
                if 'beta' in metrics:
                    st.subheader("Market Comparison")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Beta", f"{metrics['beta']:.2f}")
                    with col2:
                        st.metric("Tracking Error", f"{metrics['tracking_error']:.2%}")
                    with col3:
                        st.metric("Information Ratio", f"{metrics['information_ratio']:.2f}")
                
                # Risk metrics table
                st.subheader("All Risk Metrics")
                risk_df = pd.DataFrame({
                    'Metric': [
                        'Volatility', 'VaR (95%)', 'VaR (99%)', 'CVaR (95%)', 'CVaR (99%)',
                        'Sharpe Ratio', 'Sortino Ratio', 'Mean Return', 'Total Return'
                    ],
                    'Value': [
                        f"{metrics['volatility']:.2%}",
                        f"{metrics['var_95']:.2%}",
                        f"{metrics['var_99']:.2%}",
                        f"{metrics['cvar_95']:.2%}",
                        f"{metrics['cvar_99']:.2%}",
                        f"{metrics['sharpe_ratio']:.2f}",
                        f"{metrics['sortino_ratio']:.2f}",
                        f"{metrics['mean_return']:.2%}",
                        f"{metrics['total_return']:.2f}%"
                    ]
                })
                st.dataframe(risk_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Risk analysis error: {str(e)}")

elif page == "Feature Importance":
    st.title("📊 Feature Importance")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["xgboost", "random_forest", "extra_trees", "gradient_boosting"],
            index=0
        )
    
    if st.button("Load Feature Importance", type="primary"):
        with st.spinner("Loading feature importance..."):
            try:
                importance_df = services['forecasting_service'].get_feature_importance(
                    symbol=symbol,
                    model_type=model_type
                )
                
                # Bar chart
                fig = px.bar(
                    importance_df.head(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f"Top 20 Feature Importance - {model_type.upper()}",
                    labels={'importance': 'Importance', 'feature': 'Feature'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.subheader("All Features")
                st.dataframe(importance_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading feature importance: {str(e)}")

# New page: Live prices, predictions, risk, and chatbot
elif page == "Live & Chatbot":
    st.title("⚡ Live Prices & 🤖 Chatbot")
    st.markdown("---")
    st.markdown(
        """
        <div class="finsense-card">
            <div class="finsense-title">FinSense AI Assistant</div>
            <div class="finsense-sub">
                Ask about stocks, strategy, risk, or any project feature.
                Choose response style to get formal/friendly and concise/detailed answers.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        live_symbol = st.text_input("Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT")
    with col2:
        forecast_days = st.slider("Forecast Days", 1, 14, 5)
    with col3:
        period_live = st.selectbox("History Period", ["3mo", "6mo", "1y"], index=1)

    # Live prices
    st.subheader("Live Price")
    if st.button("Refresh Live Price"):
        with st.spinner("Fetching live price..."):
            try:
                live_fetcher = services.get("live_fetcher")
                if not live_fetcher:
                    st.warning("Live data fetcher not available.")
                else:
                    live_data = live_fetcher.get_live_price(live_symbol.upper())
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${live_data['close']:.2f}")
                    col2.metric("High", f"${live_data['high']:.2f}")
                    col3.metric("Low", f"${live_data['low']:.2f}")
                    col4.metric("Volume", f"{live_data['volume']:,}")
                    st.caption(f"Timestamp: {live_data['timestamp']}")

                    # Recent intraday chart
                    recent = live_fetcher.get_recent_data(live_symbol.upper(), minutes=120)
                    if not recent.empty:
                        fig_live = go.Figure()
                        fig_live.add_trace(go.Scatter(
                            x=recent.index,
                            y=recent['close'],
                            mode='lines',
                            name='Live Price'
                        ))
                        fig_live.update_layout(
                            title=f"{live_symbol.upper()} Intraday (last 120 min)",
                            xaxis_title="Time",
                            yaxis_title="Price"
                        )
                        st.plotly_chart(fig_live, use_container_width=True)
            except Exception as e:
                st.error(f"Live price error: {str(e)}")

    # Predictions
    st.subheader("Predictions")
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            try:
                forecast_service = services['forecasting_service']
                result = forecast_service.predict(
                    symbol=live_symbol,
                    model_type="xgboost",
                    forecast_days=forecast_days,
                    use_lstm=False,
                    adaptive_mode=True,
                    sentiment="neutral"
                )
                # Historical for context
                historical = services['data_ingestion'].fetch_historical_data(
                    symbol=live_symbol,
                    period=period_live
                )
                last_date = historical.index[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=historical.index[-60:],
                    y=historical['close'].tail(60),
                    mode='lines',
                    name='Historical'
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=result['forecasts'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(dash='dash')
                ))
                fig_forecast.update_layout(
                    title=f"{live_symbol.upper()} Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                forecast_df = pd.DataFrame({
                    "Date": forecast_dates,
                    "Forecast": result['forecasts']
                })
                st.dataframe(forecast_df, use_container_width=True)
                st.session_state["latest_forecast_result"] = result
            except Exception as e:
                st.error(f"Forecast error: {str(e)}")

    # Risk metrics
    st.subheader("Risk Metrics")
    if st.button("Compute Risk"):
        with st.spinner("Computing risk metrics..."):
            try:
                risk_service = services['risk_service']
                risk_result = risk_service.analyze_risk(
                    symbol=live_symbol,
                    period=period_live
                )
                metrics = risk_result['metrics']
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
                col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                col3.metric("VaR (95%)", f"{metrics.get('var_95', 0):.2%}")
                col4.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.2%}")

                if 'max_drawdown' in metrics:
                    drawdown = metrics['max_drawdown']
                    st.metric("Max Drawdown", f"{drawdown.get('max_drawdown', 0):.2%}")

                st.json(metrics)
            except Exception as e:
                st.error(f"Risk analysis error: {str(e)}")

    # Chatbot - Personal Finance Advisor
    st.subheader("💬 FinSense AI Chat")
    st.info("💡 Ask anything: stocks, financial concepts, platform usage, or technical/project questions.")
    use_decision_engine = st.toggle("Use AI Decision Engine", value=True, key="use_decision_engine")
    style_col1, style_col2 = st.columns(2)
    with style_col1:
        chat_style = st.selectbox(
            "Answer Tone",
            options=["default", "friendly", "formal"],
            index=0,
            key="chat_style"
        )
    with style_col2:
        chat_verbosity = st.selectbox(
            "Answer Length",
            options=["normal", "concise", "detailed"],
            index=0,
            key="chat_verbosity"
        )
    
    question = st.text_input("Ask a financial question", value="What does a Sharpe ratio of 1.5 indicate?", key="chatbot_question")
    if st.button("Ask", key="chatbot_ask"):
        with st.spinner("Thinking..."):
            try:
                rag_chatbot = get_chatbot()
                if not rag_chatbot:
                    st.error("Chatbot is unavailable. Please check API keys and dependencies.")
                    st.stop()

                # Optionally pass latest indicators if available
                context = None
                if 'metrics' in locals() and metrics:
                    context = {
                        live_symbol.upper(): {
                            "volatility": metrics.get('volatility'),
                            "sharpe_ratio": metrics.get('sharpe_ratio'),
                            "trend_direction": metrics.get('trend_direction', 'neutral')
                        }
                    }
                latest_result = st.session_state.get("latest_forecast_result")
                if isinstance(latest_result, dict):
                    c = latest_result.get("context", {})
                    context = context or {live_symbol.upper(): {}}
                    context[live_symbol.upper()].update({
                        "prediction": c.get("prediction"),
                        "rsi": c.get("rsi"),
                        "moving_average": c.get("moving_average"),
                        "sentiment": c.get("sentiment", "neutral"),
                        "volatility": c.get("volatility"),
                        "trend_direction": c.get("trend", "neutral"),
                        "top_features": c.get("top_features", []),
                    })
                if context and hasattr(rag_chatbot, "add_financial_context"):
                    try:
                        rag_chatbot.add_financial_context(
                            ticker=live_symbol.upper(),
                            indicators=context[live_symbol.upper()]
                        )
                    except Exception:
                        pass
                
                # Use the query method (works for both RAG and fallback chatbots)
                try:
                    result = rag_chatbot.query(
                        question=question,
                        ticker=live_symbol if live_symbol else None,
                        k=4,
                        return_source_documents=True,
                        response_style=chat_style,
                        verbosity=chat_verbosity
                    )
                except TypeError:
                    result = rag_chatbot.query(
                        question=question,
                        ticker=live_symbol if live_symbol else None,
                        k=4,
                        return_source_documents=True
                    )
                
                st.success("✅ Answer Generated")
                if use_decision_engine and isinstance(result, dict) and result.get("recommendation"):
                    rec = result["recommendation"]
                    signal = rec.get("signal", "HOLD")
                    confidence = rec.get("confidence", 5)
                    badge_color = "#16a34a" if signal == "BUY" else ("#dc2626" if signal == "SELL" else "#ca8a04")
                    st.markdown(
                        f"<div style='padding:10px 12px;border-radius:10px;border:1px solid #334155;background:#0f172a;'>"
                        f"<strong>Decision Signal:</strong> "
                        f"<span style='color:{badge_color};font-weight:700'>{signal}</span> "
                        f"(confidence: {confidence}/10)</div>",
                        unsafe_allow_html=True
                    )
                st.markdown(f"**Answer:**\n\n{result['answer']}")
                
                if 'source_documents' in result and result['source_documents']:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(result['source_documents'], 1):
                            st.markdown(f"**Source {i}:** {doc.get('source', 'unknown')}")
                            source_type = doc.get("type")
                            if source_type:
                                st.caption(f"Type: {source_type}")
                            content = doc.get("content", "")
                            # Use code block rendering for clean formatting (no truncation)
                            st.code(content)
                
                if 'tokens_used' in result and result.get('tokens_used', 0) > 0:
                    st.caption(f"Tokens used: {result.get('tokens_used', 0)}")
                            
            except Exception as e:
                st.error(f"Chatbot error: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
                st.info("💡 **Tip:** Make sure all dependencies are installed: `pip install -r requirements.txt`")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>FinSense Financial Intelligence System v1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)

