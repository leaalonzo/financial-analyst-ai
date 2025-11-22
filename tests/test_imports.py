# tests/test_imports.py - FIXED VERSION
import sys
import os

# Add parent directory to path so Python can find 'src'
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

print("Testing imports...")
print(f"Python path: {sys.path[0]}\n")

try:
    from langchain_openai import ChatOpenAI
    print("✅ ChatOpenAI imported")
except Exception as e:
    print(f"❌ ChatOpenAI failed: {e}")

try:
    from langchain.prompts import PromptTemplate
    print("✅ PromptTemplate imported")
except Exception as e:
    print(f"❌ PromptTemplate failed: {e}")

try:
    from langchain_core.output_parsers import StrOutputParser
    print("✅ StrOutputParser imported")
except Exception as e:
    print(f"❌ StrOutputParser failed: {e}")

try:
    from src.llm_chain import FinancialAnalystChain
    print("✅ FinancialAnalystChain imported")
except Exception as e:
    print(f"❌ FinancialAnalystChain failed: {e}")

print("\nAll imports successful!")