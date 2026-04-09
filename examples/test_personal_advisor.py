"""
Test script for Personal Finance Advisor chatbot.

This script demonstrates the enhanced chatbot capabilities including:
- Price predictions
- Creative financial advice
- Investment recommendations
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.personal_finance_advisor import PersonalFinanceAdvisor


def test_personal_advisor():
    """Test the Personal Finance Advisor chatbot."""
    print("=" * 70)
    print("Personal Finance Advisor - Test Script")
    print("=" * 70)
    
    # Initialize advisor
    print("\n1. Initializing Personal Finance Advisor...")
    try:
        advisor = PersonalFinanceAdvisor()
        print("[OK] Advisor initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return
    
    # Test questions
    test_questions = [
        "Hello!",
        "What is volatility?",
        "Explain Sharpe ratio",
        "Predict AAPL price for next 7 days",
        "What will TSLA be next week?",
        "Should I invest in MSFT?",
        "What is a good investment strategy?",
        "Explain RSI indicator",
    ]
    
    print("\n2. Testing chatbot responses...")
    print("-" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}] {question}")
        print("-" * 70)
        try:
            answer = advisor.chat(question)
            print(f"[Answer]\n{answer}")
        except Exception as e:
            print(f"[ERROR] {e}")
        print("-" * 70)
    
    # Test price prediction specifically
    print("\n3. Testing price prediction feature...")
    print("-" * 70)
    
    prediction_questions = [
        "Predict AAPL for next 5 days",
        "What will GOOGL be next week?",
        "Forecast MSFT price for next 10 days",
    ]
    
    for question in prediction_questions:
        print(f"\n[Question] {question}")
        try:
            answer = advisor.chat(question)
            print(f"[Answer]\n{answer[:500]}...")  # Show first 500 chars
        except Exception as e:
            print(f"[ERROR] {e}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_personal_advisor()










