# src/document_processor.py
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text

    def create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        return [
            {
                "content": chunk,
                "metadata": {**metadata, "chunk_id": i}
            }
            for i, chunk in enumerate(chunks)
        ]

    def process_document(self, file_path: str, doc_type: str, company: str) -> List[Dict]:
        """Main processing pipeline"""
        text = self.extract_text_from_pdf(file_path)
        metadata = {"source": file_path, "type": doc_type, "company": company}
        return self.create_chunks(text, metadata)
