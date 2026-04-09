"""
FastAPI Backend for FinSense Financial Intelligence System.

Modular API design with separate routes for prediction, risk analysis, and explanations.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from datetime import datetime

from .routes import predict_router, risk_router, explain_router
from .schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FinSense Financial Intelligence API",
    version="1.0.0",
    description="""
    Financial Intelligence System API providing:
    - Stock price predictions using ML and LSTM models
    - Comprehensive risk analysis and metrics
    - LLM-powered explainable financial insights
    
    Built with FastAPI, LangChain, and advanced ML models.
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router)
app.include_router(risk_router)
app.include_router(explain_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FinSense Financial Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "prediction": "/predict",
            "risk": "/risk",
            "explain": "/explain",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "prediction": "available",
            "risk_analysis": "available",
            "rag_chatbot": "available" if (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")) else "requires_api_key"
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"HTTP {exc.status_code} error"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True
    )













