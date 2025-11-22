# src/llm_chain.py - FIXED FOR TESLA-NVIDIA COMPARISON
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, List
import os

class FinancialAnalystChain:
    def __init__(self, vector_store, model_name: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.1,
            openai_api_key=api_key
        )
        self.vector_store = vector_store
        
        self.prompt = PromptTemplate.from_template(
            """You are an expert financial analyst. Analyze the provided context from financial documents to answer the question thoroughly.

CRITICAL INSTRUCTIONS:
1. **Use ALL relevant information from the context**
2. **For comparative questions**: Provide details for EACH company mentioned
3. **Cite specific numbers and data points**
4. **Structure comparative answers** with clear sections for each company
5. **If data is missing**: Explicitly state what's missing
6. **Always attempt an answer** based on available context

CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

QUESTION: {question}

DETAILED ANALYSIS:"""
        )
    
    def format_docs(self, docs):
        """Format retrieved documents with clear separation"""
        formatted = []
        for i, doc in enumerate(docs):
            company = doc.metadata.get('company', 'Unknown')
            doc_type = doc.metadata.get('type', 'Unknown')
            formatted.append(f"[Document {i+1} - {company} - {doc_type}]:\n{doc.page_content}\n")
        return "\n---\n".join(formatted)
    
    def _enhance_query_for_retrieval(self, query: str) -> List[str]:
        """Generate multiple search queries for better coverage"""
        queries = [query]
        
        if any(word in query.lower() for word in ["compare", "versus", "vs"]):
            companies = []
            for word in ["Tesla", "Nvidia", "Apple", "Microsoft", "Google", "Amazon", "Meta"]:
                if word.lower() in query.lower():
                    companies.append(word)
            
            metric = ""
            for m in ["revenue", "r&d", "margin", "profit", "risk", "growth"]:
                if m in query.lower():
                    metric = m
                    break
            
            if len(companies) >= 2 and metric:
                for company in companies:
                    queries.append(f"{company} {metric}")
        
        return queries
    
    def analyze_query(self, query: str, force_companies: List[str] = None) -> Dict:
        """
        Run analysis with optional forced balanced retrieval from specific companies
        """
        try:
            all_docs = []
            
            # ← CRITICAL FIX: Force balanced retrieval for comparisons
            if force_companies and len(force_companies) >= 2:
                seen_content = set()
                
                # Get documents from EACH company separately
                for company in force_companies:
                    # Search with company-specific query
                    company_query = f"{company} revenue financial performance"
                    
                    retriever = self.vector_store.vectorstore.as_retriever(
                        search_kwargs={"k": 10}
                    )
                    docs = retriever.get_relevant_documents(company_query)
                    
                    # Filter to only this company's docs
                    company_docs = [
                        d for d in docs 
                        if d.metadata.get('company', '').lower() == company.lower()
                    ]
                    
                    # Add unique documents
                    for doc in company_docs[:7]:  # Max 7 per company
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            all_docs.append(doc)
                            seen_content.add(content_hash)
                
            else:
                # Normal retrieval
                search_queries = self._enhance_query_for_retrieval(query)
                seen_content = set()
                
                for search_query in search_queries:
                    retriever = self.vector_store.vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )
                    docs = retriever.get_relevant_documents(search_query)
                    
                    for doc in docs:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            all_docs.append(doc)
                            seen_content.add(content_hash)
                
                all_docs = all_docs[:15]
            
            if not all_docs:
                return {
                    "answer": "I couldn't find relevant information in the uploaded documents.",
                    "sources": []
                }
            
            # Format context
            context = self.format_docs(all_docs)
            
            # Create prompt
            formatted_prompt = self.prompt.format(
                context=context,
                question=query
            )
            
            # Get LLM response
            response = self.llm.invoke(formatted_prompt)
            
            return {
                "answer": response.content,
                "sources": [
                    {
                        "content": doc.page_content[:300] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in all_docs[:10]
                ]
            }
        
        except Exception as e:
            import traceback
            return {
                "answer": f"Error: {str(e)}\n\n{traceback.format_exc()}",
                "sources": []
            }