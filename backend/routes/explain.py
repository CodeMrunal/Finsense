"""
LLM explanation endpoints using RAG chatbot.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging
import os

from ..schemas import ExplainRequest, ExplainResponse, ErrorResponse
from ..dependencies import services

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explanation"])


@router.post("", response_model=ExplainResponse)
async def explain_financial_insight(request: ExplainRequest):
    """
    Get LLM-powered explanation for financial questions.
    
    Uses RAG chatbot to provide grounded and explainable answers.
    
    Args:
        request: Explanation request with question and optional context
    
    Returns:
        ExplainResponse with answer and sources
    """
    try:
        logger.info(f"Explanation request: {request.question[:50]}...")
        
        # Check if RAG chatbot is available
        rag_chatbot = services.get_rag_chatbot()
        if not rag_chatbot:
            # Check if any supported API key is available
            openai_key = os.getenv("OPENAI_API_KEY")
            groq_key = os.getenv("GROQ_API_KEY")
            if not (openai_key or groq_key):
                raise HTTPException(
                    status_code=503,
                    detail="RAG chatbot not available. Set GROQ_API_KEY or OPENAI_API_KEY environment variable."
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="RAG chatbot not initialized. Please check configuration."
                )
        
        # Add financial context if provided
        if request.context:
            for ticker, indicators in request.context.items():
                rag_chatbot.add_financial_context(
                    ticker=ticker,
                    indicators=indicators
                )
        
        # Query the chatbot
        result = rag_chatbot.query(
            question=request.question,
            ticker=request.symbol,
            k=4,
            return_source_documents=request.include_sources
        )
        
        # Determine confidence based on sources
        confidence = "high" if result.get("retrieved_documents_count", 0) >= 3 else "medium"
        
        return ExplainResponse(
            answer=result["answer"],
            question=request.question,
            symbol=request.symbol,
            sources=result.get("source_documents", []) if request.include_sources else None,
            tokens_used=result.get("tokens_used"),
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.get("/health")
async def check_rag_health():
    """
    Check if RAG chatbot is available.
    
    Returns:
        Health status
    """
    rag_chatbot = services.get_rag_chatbot()
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    return {
        "rag_available": rag_chatbot is not None,
        "api_key_set": (openai_key is not None) or (groq_key is not None),
        "provider": "groq" if groq_key else ("openai" if openai_key else None),
        "status": "healthy" if rag_chatbot else "unavailable"
    }













