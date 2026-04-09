"""
Retrieval-Augmented Generation (RAG) Chatbot for Financial Insights.

This module provides a RAG-based chatbot that combines financial indicators
with LLM responses to provide grounded and explainable financial insights.

Features:
- LangChain integration for RAG pipeline
- FAISS vector store for document retrieval
- Financial indicator integration
- Explainable and grounded responses
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import sys
from dotenv import load_dotenv
from src.models.decision_engine import compute_decision

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import LangChain components (LangChain 1.2.0+ uses langchain_core and langchain_openai)
try:
    # LangChain 1.2.0+ imports (correct structure)
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # Note: RetrievalQA and load_qa_chain are not needed - we build RAG manually
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths
        from langchain_core.documents import Document
        from langchain_core.prompts import PromptTemplate
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            ChatGroq = None
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            HuggingFaceEmbeddings = None
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        # Fallback: Create minimal Document class for type hints
        class Document:
            """Minimal Document class fallback when LangChain is not available."""
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                self.page_content = page_content
                self.metadata = metadata or {}
        
        # Create minimal fallback classes
        class PromptTemplate:
            """Minimal PromptTemplate fallback."""
            def __init__(self, template: str = "", input_variables: Optional[List[str]] = None):
                self.template = template
                self.input_variables = input_variables or []
            def format(self, **kwargs) -> str:
                return self.template.format(**kwargs)
        
        LANGCHAIN_AVAILABLE = False
        logging.warning("LangChain not available. Install with: pip install langchain-community langchain-openai faiss-cpu")

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are FinSense AI, an expert financial and product assistant for the FinSense platform.

## Core Role
You help users with:
1) Stock market questions (fundamentals, technicals, risk, trends, portfolio ideas, definitions).
2) FinSense project questions (features, workflows, APIs, dashboard, model training, troubleshooting, usage).
3) General knowledge questions, while staying concise, accurate, and practical.

## Behavior Rules
- Be accurate, clear, and action-oriented.
- If a question is ambiguous, ask 1 short clarifying question before proceeding.
- If user asks for recommendation, provide reasoning + risks + alternatives.
- Never claim certainty for uncertain outcomes.
- If data is missing, say what is missing and suggest how to get it.
- For stock predictions, include uncertainty and risk disclaimers.
- Do not provide illegal/unsafe financial manipulation advice.
- Do not fabricate platform capabilities; if unsure, say so.

## Response Style Control
The user can request style in plain words, such as:
- "formal"
- "friendly"
- "concise"
- "detailed"

Style policy:
- If user says "formal": professional tone, structured bullets, no slang.
- If user says "friendly": warm conversational tone, simple language.
- If user says "concise": short direct answer first, then optional details.
- If user says "detailed": full explanation with examples.
- If no style is given: default to clear, semi-friendly professional tone.

## FinSense Context Priority
When the user asks about FinSense, prioritize:
- How to use dashboard pages/features
- How predictions and risk metrics work
- How chatbot/explain endpoints work
- Practical troubleshooting steps (ports, env keys, model setup)
- Suggested next actions inside FinSense

## Stock Question Format
For stock-specific questions, usually answer in this structure:
1) Quick takeaway
2) Key signals (trend, volatility, risk, valuation/technical context)
3) Scenarios (bull/base/bear if relevant)
4) Risk warning and practical next step

## Recommendation Safety
- Avoid absolute language like “guaranteed” or “sure profit”.
- Use wording like “based on current signals…”, “higher probability…”, “risk remains…”.
- Encourage diversification and position sizing.

## If User Asks “Anything”
When user asks broad/open questions, behave like a high-quality general assistant:
- explain concepts clearly,
- offer practical examples,
- adapt depth to user level,
- and ask if they want deeper technical/advanced detail.

Always optimize for usefulness, honesty, and clarity.
"""

DEFAULT_DECISION_PROMPT = """You are FinSense AI, an AI-powered financial decision assistant.

STRICT RULES:
- Do NOT give generic advice
- Do NOT say "I am not a financial advisor"
- ALWAYS use the provided data
- ALWAYS give a clear recommendation

Stock Data:
- Prediction: {prediction}
- Trend: {trend}
- RSI: {rsi}
- Moving Average: {ma}
- Sentiment: {sentiment}

Based on this data, generate output in EXACT format:

📊 Recommendation: (BUY / SELL / HOLD)
Confidence: (0-100%)
Risk: (Low / Medium / High)

🧠 Key Reasons:
- point 1
- point 2
- point 3

📈 Summary:
(2-3 lines max, clear and direct)

IMPORTANT:
- Be confident
- Be concise
- Do NOT give long paragraphs
"""


class FinancialRAGChatbot:
    """
    RAG-based chatbot for explainable financial insights.
    
    Combines financial indicators with LLM knowledge to provide
    grounded and explainable financial advice and insights.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        vector_store_path: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the Financial RAG Chatbot.
        
        Args:
            openai_api_key: OpenAI API key. If None, reads from environment
            groq_api_key: Groq API key. If None, reads from environment
            provider: LLM provider ("groq" or "openai"). Auto-detected if None.
            model_name: LLM model name. Default: "gpt-3.5-turbo"
            temperature: Model temperature. Default: 0.7
            vector_store_path: Path to save/load FAISS vector store
            knowledge_base_path: Path to financial knowledge base documents
            chunk_size: Text chunk size for splitting. Default: 1000
            chunk_overlap: Overlap between chunks. Default: 200
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required. Install with: pip install langchain openai faiss-cpu"
            )

        # Ensure .env variables are available when launched from IDE/dev servers.
        load_dotenv()
        
        # Set API keys and provider
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
        if not self.provider:
            if self.groq_api_key:
                self.provider = "groq"
            elif self.openai_api_key:
                self.provider = "openai"
            else:
                raise ValueError(
                    "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY."
                )
        
        if self.provider == "groq":
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when provider='groq'.")
            os.environ["GROQ_API_KEY"] = self.groq_api_key
        elif self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when provider='openai'.")
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            if model_name == "llama-3.3-70b-versatile":
                model_name = "gpt-4o-mini"
        else:
            raise ValueError("Unsupported provider. Use 'groq' or 'openai'.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.vector_store_path = Path(vector_store_path) if vector_store_path else Path("chatbot/vector_store")
        self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else Path("chatbot/knowledge_base")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.text_splitter = None
        
        # Financial context storage
        self.financial_context = {}
        self.system_prompt = os.getenv("CHATBOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        self.decision_prompt = os.getenv("CHATBOT_DECISION_PROMPT", DEFAULT_DECISION_PROMPT)
        
        logger.info("FinancialRAGChatbot initialized")
        logger.info(f"  Provider: {self.provider}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Vector store path: {self.vector_store_path}")
        logger.info(f"  Knowledge base path: {self.knowledge_base_path}")
    
    def _initialize_components(self):
        """Initialize LangChain components."""
        logger.info("Initializing LangChain components...")
        
        # Initialize embeddings
        if self.provider == "groq":
            if HuggingFaceEmbeddings is None:
                raise ImportError(
                    "HuggingFace embeddings are required for Groq mode. "
                    "Install with: pip install sentence-transformers"
                )
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Initialize LLM
        if self.provider == "groq":
            if ChatGroq is None:
                raise ImportError(
                    "Groq integration not installed. Install with: pip install langchain-groq"
                )
            self.llm = ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                groq_api_key=self.groq_api_key
            )
        elif "gpt" in self.model_name.lower():
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.openai_api_key
            )
        else:
            # Fallback for non-GPT models
            try:
                from langchain_community.llms import OpenAI
                self.llm = OpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=self.openai_api_key
                )
            except ImportError:
                # Use ChatOpenAI as fallback
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=self.openai_api_key
                )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        logger.info("Components initialized successfully")
    
    def load_knowledge_base(
        self,
        documents: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load financial knowledge base documents.
        
        Args:
            documents: List of document texts
            file_paths: List of file paths to load
        
        Returns:
            List of LangChain Document objects
        """
        all_documents = []
        
        # Load from provided documents
        if documents:
            for i, doc_text in enumerate(documents):
                all_documents.append(Document(
                    page_content=doc_text,
                    metadata={"source": f"document_{i}", "type": "financial_knowledge"}
                ))
        
        # Load from files
        if file_paths:
            for file_path in file_paths:
                path = Path(file_path)
                if path.exists():
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        all_documents.append(Document(
                            page_content=content,
                            metadata={"source": str(path), "type": "financial_knowledge"}
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {str(e)}")
        
        # Load from knowledge base directory
        if self.knowledge_base_path.exists():
            for file_path in self.knowledge_base_path.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    all_documents.append(Document(
                        page_content=content,
                        metadata={"source": str(file_path), "type": "financial_knowledge"}
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(all_documents)} documents from knowledge base")
        return all_documents
    
    def create_vector_store(
        self,
        documents: Optional[List[Document]] = None,
        force_recreate: bool = False
    ):
        """
        Create or load FAISS vector store.
        
        Args:
            documents: Documents to index. If None, loads existing store
            force_recreate: Whether to recreate even if exists. Default: False
        """
        if self.embeddings is None:
            self._initialize_components()
        
        # Check if vector store exists
        if not force_recreate and (self.vector_store_path / "index.faiss").exists():
            logger.info("Loading existing vector store...")
            try:
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings
                )
                logger.info("Vector store loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load vector store: {str(e)}. Recreating...")
        
        # Create new vector store
        if documents is None:
            documents = self.load_knowledge_base()
        
        if not documents:
            logger.warning("No documents provided. Creating empty vector store.")
            # Create empty vector store with a dummy document
            dummy_doc = Document(
                page_content="Financial markets involve buying and selling of securities.",
                metadata={"source": "dummy", "type": "financial_knowledge"}
            )
            documents = [dummy_doc]
        
        # Split documents
        logger.info("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(split_docs)} document chunks")
        
        # Create vector store
        logger.info("Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # Save vector store
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_path))
        logger.info(f"Vector store saved to {self.vector_store_path}")
    
    def add_financial_context(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        prices: Optional[pd.Series] = None
    ):
        """
        Add financial indicators and context for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            indicators: Dictionary of financial indicators
            prices: Optional price series
        """
        self.financial_context[ticker] = {
            "indicators": indicators,
            "prices": prices.to_dict() if prices is not None else None,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        logger.info(f"Added financial context for {ticker}")
    
    def _create_financial_context_string(self, ticker: Optional[str] = None) -> str:
        """
        Create context string from financial indicators.
        
        Args:
            ticker: Optional ticker to include context for
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        if ticker and ticker in self.financial_context:
            context = self.financial_context[ticker]
            context_parts.append(f"\nFinancial Indicators for {ticker}:")
            
            indicators = context.get("indicators", {})
            for key, value in indicators.items():
                if isinstance(value, (int, float)):
                    context_parts.append(f"  - {key}: {value:.4f}")
                else:
                    context_parts.append(f"  - {key}: {value}")
        
        elif self.financial_context:
            context_parts.append("\nAvailable Financial Context:")
            for ticker_key, context in self.financial_context.items():
                context_parts.append(f"  {ticker_key}: {len(context.get('indicators', {}))} indicators")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _create_style_instruction(
        self,
        response_style: Optional[str] = None,
        verbosity: Optional[str] = None
    ) -> str:
        """Create style instructions for the current response."""
        style_map = {
            "formal": "Use a formal professional tone and structured bullets.",
            "friendly": "Use a friendly conversational tone with simple language.",
            "default": "Use a balanced professional-friendly tone."
        }
        verbosity_map = {
            "concise": "Keep the response concise. Lead with a direct answer.",
            "detailed": "Provide a detailed response with examples and practical guidance.",
            "normal": "Provide a clear medium-length response."
        }

        style_key = (response_style or "default").strip().lower()
        verbosity_key = (verbosity or "normal").strip().lower()
        style_instruction = style_map.get(style_key, style_map["default"])
        verbosity_instruction = verbosity_map.get(verbosity_key, verbosity_map["normal"])
        return f"{style_instruction} {verbosity_instruction}"

    def _is_recommendation_query(self, question: str) -> bool:
        """Detect whether user is asking for a buy/sell/hold decision."""
        q = (question or "").lower()
        keywords = [
            "buy", "sell", "hold", "recommendation", "should i invest",
            "should i buy", "should i sell", "decision", "entry", "exit",
            "is it good stock", "is this stock good"
        ]
        return any(k in q for k in keywords)

    def _rule_based_recommendation(self, ticker: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Generate a simple rule-based BUY/SELL/HOLD signal from available context.
        Returns None if insufficient context is available.
        """
        if not ticker or ticker not in self.financial_context:
            return None

        indicators = self.financial_context.get(ticker, {}).get("indicators", {})
        if not isinstance(indicators, dict) or not indicators:
            return None

        decision = compute_decision(
            prediction=indicators.get("prediction", 0.0),
            rsi=indicators.get("rsi", 50.0),
            sentiment=indicators.get("sentiment", "neutral"),
            volatility=indicators.get("volatility", 0.2),
            trend=indicators.get("trend_direction", "stable")
        )
        signal = decision.get("decision", "HOLD")
        confidence_pct = float(decision.get("confidence", 50.0))
        confidence = max(1, min(10, int(round(confidence_pct / 10))))
        reasons = [
            f"RSI contribution score: {decision['components']['rsi_score']}",
            f"Prediction contribution score: {decision['components']['prediction_score']}",
            f"Sentiment contribution score: {decision['components']['sentiment_score']}",
        ]
        return {
            "signal": signal,
            "confidence": confidence,
            "confidence_pct": confidence_pct,
            "risk_level": decision.get("risk_level", "Medium"),
            "reasons": reasons,
            "top_features": indicators.get("top_features", []),
        }

    def _risk_from_indicators(self, indicators: Dict[str, Any]) -> str:
        """Map indicators to Low/Medium/High risk bucket."""
        volatility = indicators.get("volatility")
        if isinstance(volatility, (int, float)):
            vol_pct = volatility * 100 if volatility <= 2 else volatility
            if vol_pct >= 30:
                return "High"
            if vol_pct >= 15:
                return "Medium"
            return "Low"
        return "Medium"

    def _build_decision_answer(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> str:
        """Render strict recommendation output format."""
        signal = decision.get("signal", "HOLD")
        confidence_100 = max(0, min(100, int(round(float(decision.get("confidence_pct", 50.0))))))
        reasons = decision.get("reasons", [])[:3]
        if len(reasons) < 3:
            defaults = [
                "Signal combines trend, volatility, and Sharpe profile.",
                "Current setup favors disciplined position sizing.",
                "Watch for trend reversal or volatility expansion."
            ]
            for d in defaults:
                if len(reasons) >= 3:
                    break
                reasons.append(d)

        trend = indicators.get("trend_direction", "Neutral")
        rsi = indicators.get("rsi", "N/A")
        ma = indicators.get("moving_average", indicators.get("sma_20", "N/A"))
        prediction = indicators.get("prediction", "N/A")
        sentiment = indicators.get("sentiment", "Neutral")
        risk = decision.get("risk_level", self._risk_from_indicators(indicators))
        top_features = decision.get("top_features", [])[:3]

        summary = (
            f"{ticker}: {signal} signal with {confidence_100}% confidence based on "
            f"trend ({trend}) and risk profile.\n"
            f"Risk is {risk}; maintain appropriate position sizing and monitor for setup changes."
        )

        return (
            f"📊 Recommendation: {signal}\n"
            f"Confidence: {confidence_100}%\n"
            f"Risk: {risk}\n\n"
            f"🧠 Key Reasons:\n"
            f"- {reasons[0]}\n"
            f"- {reasons[1]}\n"
            f"- {reasons[2]}\n\n"
            f"Top factors influencing decision: {', '.join(top_features) if top_features else 'RSI, trend, volatility'}\n\n"
            f"📈 Summary:\n"
            f"{summary}"
        )

    def _create_prompt_template(self) -> PromptTemplate:
        """Create custom prompt template for financial insights."""
        template = """{system_prompt}

Use the following financial context and retrieved documents to answer questions accurately and explainably.

Financial Context:
{financial_context}

Retrieved Context:
{context}

Question: {question}
Response Style:
{style_instruction}

Instructions:
1. Provide a clear, accurate answer based on the retrieved context and financial indicators
2. Explain your reasoning and cite specific indicators or data points
3. If the question involves specific financial metrics, reference the provided indicators
4. If information is not available in the context, say so clearly
5. Provide actionable insights when possible
6. If the user asks for a recommendation/decision, include:
   - Recommendation: BUY / SELL / HOLD
   - Confidence: X/10
   - Top reasons (2-4 bullets)
   - Risk note

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["system_prompt", "financial_context", "context", "question", "style_instruction"]
        )
    
    def query(
        self,
        question: str,
        ticker: Optional[str] = None,
        k: int = 4,
        return_source_documents: bool = True,
        response_style: Optional[str] = None,
        verbosity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a financial question.
        
        Args:
            question: User question
            ticker: Optional ticker symbol for context
            k: Number of documents to retrieve. Default: 4
            return_source_documents: Whether to return source documents. Default: True
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        if self.llm is None:
            self._initialize_components()
        
        # Get financial context
        financial_context = self._create_financial_context_string(ticker)
        rule_signal = None
        if self._is_recommendation_query(question):
            rule_signal = self._rule_based_recommendation(ticker)
            if rule_signal:
                indicators = self.financial_context.get(ticker, {}).get("indicators", {})
                strict_answer = self._build_decision_answer(ticker, indicators, rule_signal)
                return {
                    "answer": strict_answer,
                    "question": question,
                    "ticker": ticker,
                    "tokens_used": 0,
                    "cost": 0.0,
                    "retrieved_documents_count": 0,
                    "recommendation": rule_signal,
                    "source_documents": []
                }
        
        # Retrieve relevant documents
        logger.info(f"Retrieving documents for query: {question[:50]}...")
        relevant_docs = self.vector_store.similarity_search(question, k=k)
        
        # Combine retrieved context
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt_template = self._create_prompt_template()
        style_instruction = self._create_style_instruction(response_style, verbosity)
        prompt = prompt_template.format(
            system_prompt=self.system_prompt,
            financial_context=financial_context,
            context=context_text,
            question=question,
            style_instruction=style_instruction
        )
        
        # Generate answer
        logger.info("Generating answer with LLM...")
        tokens_used = 0
        cost = 0.0
        try:
            # Use LangChain 1.2.0+ API (invoke)
            if hasattr(self.llm, 'invoke'):
                from langchain_core.messages import HumanMessage
                result = self.llm.invoke([HumanMessage(content=prompt)])
                response = result.content if hasattr(result, 'content') else str(result)
            elif hasattr(self.llm, 'predict'):
                # Fallback to predict if available
                response = self.llm.predict(prompt)
            else:
                # Last resort: call directly
                response = str(self.llm(prompt))
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            tokens_used = len(prompt) // 4 + len(str(response)) // 4
            cost = tokens_used * 0.000002  # Rough estimate for GPT-3.5-turbo
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response = f"I encountered an error while processing your question: {str(e)}"
            tokens_used = 0
            cost = 0.0
        
        # Prepare result
        result = {
            "answer": response,
            "question": question,
            "ticker": ticker,
            "tokens_used": tokens_used,
            "cost": cost,
            "retrieved_documents_count": len(relevant_docs)
        }
        if rule_signal:
            result["recommendation"] = rule_signal
        
        if return_source_documents:
            result["source_documents"] = [
                {
                    "content": doc.page_content[:500],  # First 500 chars
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("type", "unknown")
                }
                for doc in relevant_docs
            ]
        
        logger.info(f"Query completed. Tokens used: {tokens_used}")
        
        return result
    
    def chat(
        self,
        question: str,
        ticker: Optional[str] = None,
        k: int = 4
    ) -> str:
        """
        Simple chat interface that returns just the answer.
        
        Args:
            question: User question
            ticker: Optional ticker symbol for context
            k: Number of documents to retrieve
        
        Returns:
            Answer string
        """
        result = self.query(question, ticker=ticker, k=k, return_source_documents=False)
        return result["answer"]
    
    def initialize_with_default_knowledge(self):
        """Initialize with default financial knowledge base."""
        default_documents = [
            """
            Stock Market Basics:
            - Stocks represent ownership in a company
            - Stock prices fluctuate based on supply and demand
            - Key metrics include P/E ratio, market cap, and dividend yield
            - Technical analysis uses price patterns and indicators
            - Fundamental analysis examines company financials
            """,
            """
            Financial Indicators:
            - Volatility measures price variability over time
            - Sharpe Ratio indicates risk-adjusted returns
            - Moving averages smooth price data to identify trends
            - RSI (Relative Strength Index) identifies overbought/oversold conditions
            - MACD shows momentum and trend changes
            """,
            """
            Risk Management:
            - Diversification reduces portfolio risk
            - Stop-loss orders limit potential losses
            - Position sizing is crucial for risk management
            - Correlation between assets affects portfolio risk
            - Risk-free rate is used in Sharpe ratio calculations
            """,
            """
            Investment Strategies:
            - Buy and hold: Long-term investment approach
            - Dollar-cost averaging: Regular investments over time
            - Value investing: Buying undervalued stocks
            - Growth investing: Focusing on high-growth companies
            - Momentum investing: Following price trends
            """
        ]
        
        self.create_vector_store(documents=self.load_knowledge_base(documents=default_documents))
        logger.info("Initialized with default financial knowledge base")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Financial RAG Chatbot - Example Usage")
    print("=" * 60)
    
    # Note: Requires OPENAI_API_KEY environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️  Warning: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("\nExample code:")
        print("""
    from chatbot import FinancialRAGChatbot
    
    # Initialize chatbot
    chatbot = FinancialRAGChatbot(
        openai_api_key="your-api-key",
        model_name="gpt-3.5-turbo"
    )
    
    # Initialize with default knowledge
    chatbot.initialize_with_default_knowledge()
    
    # Add financial context
    chatbot.add_financial_context(
        ticker="AAPL",
        indicators={
            "volatility": 0.25,
            "sharpe_ratio": 1.5,
            "current_price": 150.0,
            "trend": "upward"
        }
    )
    
    # Query
    response = chatbot.chat(
        "What does a Sharpe ratio of 1.5 mean for AAPL?",
        ticker="AAPL"
    )
    print(response)
    
    # Detailed query
    result = chatbot.query(
        "Explain the relationship between volatility and risk",
        ticker="AAPL"
    )
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['source_documents'])} documents retrieved")
        """)
    else:
        try:
            # Initialize chatbot
            chatbot = FinancialRAGChatbot(
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo"
            )
            
            # Initialize with default knowledge
            print("\n1. Initializing with default knowledge base...")
            chatbot.initialize_with_default_knowledge()
            
            # Add financial context
            print("\n2. Adding financial context...")
            chatbot.add_financial_context(
                ticker="AAPL",
                indicators={
                    "volatility": 0.25,
                    "sharpe_ratio": 1.5,
                    "current_price": 150.0,
                    "trend": "upward",
                    "pe_ratio": 28.5
                }
            )
            
            # Example queries
            print("\n3. Example Queries:")
            
            questions = [
                "What does a Sharpe ratio of 1.5 mean?",
                "Explain volatility in stock markets",
                "What is the relationship between risk and return?"
            ]
            
            for question in questions:
                print(f"\nQ: {question}")
                response = chatbot.chat(question, ticker="AAPL")
                print(f"A: {response[:200]}...")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Make sure OPENAI_API_KEY is set correctly and you have API credits.")




