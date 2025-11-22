# src/vector_store.py - PRODUCTION READY
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict
import os
import streamlit as st

class FinancialVectorStore:
    """
    In-memory vector store for financial documents.
    No disk persistence - state is maintained via st.session_state.processed_chunks
    """
    
    def __init__(self):
        # Get API key from Streamlit secrets or environment
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY in .env or Streamlit secrets."
            )
        
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # Cost-effective embedding model
        )
        
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Dict]):
        """
        Create in-memory vector store from documents.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            
        Returns:
            Chroma vectorstore instance
        """
        # Validation
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        # Extract texts and metadata
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        if not texts:
            raise ValueError("No text content in documents")
        
        # Create in-memory vector store (no persistence)
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
            # Note: NO persist_directory - keeps everything in memory
        )
        
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 4, filter_dict: Dict = None):
        """
        Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        if filter_dict:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        
        return self.vectorstore.similarity_search(query, k=k)