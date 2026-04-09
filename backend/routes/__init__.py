"""API routes module."""

from .predict import router as predict_router
from .risk import router as risk_router
from .explain import router as explain_router

__all__ = ["predict_router", "risk_router", "explain_router"]













