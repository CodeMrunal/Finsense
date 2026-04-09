"""
Test script for Financial RAG Chatbot.
Run this to diagnose chatbot issues.
"""
import os
import sys

print("=" * 60)
print("Chatbot Diagnostic Test")
print("=" * 60)

# Check 1: Python version
print("\n1. Checking Python version...")
print(f"   Python: {sys.version}")

# Check 2: Required packages
print("\n2. Checking required packages...")
try:
    import langchain
    print(f"   [OK] LangChain: {langchain.__version__}")
except ImportError:
    print("   [ERROR] LangChain: NOT INSTALLED")
    print("   Fix: pip install langchain")

try:
    import openai
    print(f"   [OK] OpenAI: {openai.__version__}")
except ImportError:
    print("   [ERROR] OpenAI: NOT INSTALLED")
    print("   Fix: pip install openai")

try:
    import faiss
    print(f"   [OK] FAISS: Installed")
except ImportError:
    print("   [ERROR] FAISS: NOT INSTALLED")
    print("   Fix: pip install faiss-cpu")

# Check 3: API Key
print("\n3. Checking OpenAI API Key...")
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
    print(f"   [OK] API Key: {masked_key}")
else:
    print("   [ERROR] API Key: NOT SET")
    print("   Fix: Set OPENAI_API_KEY environment variable")
    print("   Windows: set OPENAI_API_KEY=sk-your-key")
    print("   Mac/Linux: export OPENAI_API_KEY=sk-your-key")

# Check 4: Import chatbot
print("\n4. Testing chatbot import...")
try:
    from chatbot.financial_rag import FinancialRAGChatbot
    print("   [OK] Chatbot module imported successfully")
except Exception as e:
    print(f"   [ERROR] Import failed: {str(e)}")
    sys.exit(1)

# Check 5: Initialize chatbot
if api_key:
    print("\n5. Testing chatbot initialization...")
    try:
        chatbot = FinancialRAGChatbot(openai_api_key=api_key)
        print("   [OK] Chatbot initialized")
        
        # Check 6: Initialize knowledge base
        print("\n6. Testing knowledge base initialization...")
        try:
            chatbot.initialize_with_default_knowledge()
            print("   [OK] Knowledge base initialized")
        except Exception as e:
            print(f"   [ERROR] Knowledge base failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Check 7: Test query
        print("\n7. Testing chatbot query...")
        try:
            response = chatbot.chat("What is volatility?", ticker=None)
            print(f"   [OK] Query successful!")
            print(f"   Response: {response[:100]}...")
        except Exception as e:
            print(f"   [ERROR] Query failed: {str(e)}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"   [ERROR] Initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("\n5. Skipping initialization (no API key)")

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)

