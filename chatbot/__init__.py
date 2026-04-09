"""Financial RAG Chatbot for explainable financial insights."""

# Try to import the full RAG chatbot, fallback to free version
try:
    from .financial_rag import FinancialRAGChatbot
    HAS_OPENAI = True
except (ImportError, ValueError):
    # Fallback to free version if OpenAI not available
    from .free_chatbot import FreeFinancialChatbot as FinancialRAGChatbot
    HAS_OPENAI = False

# Import Personal Finance Advisor (enhanced chatbot with predictions)
try:
    from .personal_finance_advisor import PersonalFinanceAdvisor
except ImportError:
    PersonalFinanceAdvisor = None

__all__ = ["FinancialRAGChatbot", "FreeFinancialChatbot", "PersonalFinanceAdvisor", "HAS_OPENAI"]
