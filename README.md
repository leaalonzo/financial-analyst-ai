# 🤖 AI Financial Analyst Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> LLM-powered financial document analysis using Retrieval-Augmented Generation (RAG) architecture

Upload financial documents (10-Ks, earnings calls, analyst reports) and ask questions in natural language. The AI analyzes them and provides insights with cited sources.

---

## ✨ Features

### Core Capabilities
- **📄 Multi-Document Analysis**: Upload and analyze multiple financial reports simultaneously
- **🎯 Source Citations**: Every answer includes references to specific documents and page numbers
- **🔍 Semantic Search**: Advanced RAG with ChromaDB vector database for accurate information retrieval
- **📊 Comparative Analysis**: Compare metrics across different companies (revenue, R&D, margins, risks)
- **⚡ Real-Time Processing**: Fast document processing and query responses (<3s average)
- **💾 Persistent Sessions**: Documents remain loaded across page refreshes until explicitly cleared
- **📥 Export Reports**: Download conversation history as formatted text files

---

## 🏗️ Architecture
```
┌─────────────┐
│ User Upload │ PDF Files
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Document Processing │ PyPDF2
│  - Text Extraction  │
│  - Chunking (1000)  │
│  - Metadata         │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Embeddings        │ OpenAI text-embedding-3-small
│  - Vector Creation  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Vector Store      │ ChromaDB (in-memory)
│  - Semantic Search  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   User Query        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  RAG Retrieval      │ Top-k similarity search
│  - Query Embedding  │ Forced balanced retrieval
│  - Doc Retrieval    │ for comparisons
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   LLM Analysis      │ GPT-4o-mini
│  - Context + Query  │
│  - Generate Answer  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Cited Response     │ Answer + Sources
└─────────────────────┘
```

**Technology Stack:**
- **LLM**: OpenAI GPT-4o-mini (cost-optimized, fast)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: ChromaDB (in-memory for zero-latency)
- **Framework**: LangChain 0.1+
- **Frontend**: Streamlit 1.28+
- **PDF Processing**: PyPDF2

---

## 📊 Use Cases

### Investment Research
Quickly analyze company filings and extract key financial metrics across multiple quarters or competitors.

**Example Query:**
> "Compare Tesla and Nvidia's revenue growth over the past 3 years"

### Due Diligence
Identify risks, opportunities, and strategic initiatives mentioned in corporate documents.

**Example Query:**
> "What are the main risks Tesla mentioned in their latest 10-K?"

### Financial Modeling
Extract specific numbers and data points for model building.

**Example Query:**
> "What was Apple's R&D spending in 2024 and how did it change from 2023?"

### Competitive Analysis
Compare key metrics across industry competitors.

**Example Query:**
> "Compare the profit margins of Tesla, Apple, and Nvidia"

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key 

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/leaalonzo/financial-analyst-ai.git
cd financial-analyst-ai
```

2. **Create virtual environment**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open in browser**
```
Navigate to: http://localhost:8501
```

---

## 📖 Usage

### Quick Start

1. **Upload Documents**
   - Click "Upload Financial Documents (PDF)" in the sidebar
   - Select one or more PDF files (max 10MB each)
   - Click "Process Documents"

2. **Ask Questions**
   - Type your question in the main input box
   - Click "🔍 Analyze" to get an answer
   - View sources by clicking "📄 View Sources"

3. **Compare Companies**
   - Expand "🔬 Comparative Analysis"
   - Enter two company names
   - Select a metric (Revenue, R&D, Margins, Risks)
   - Click "⚖️ Compare"

### Demo Mode

Click **"Load Samples"** to load pre-configured demo documents (Tesla, Apple, Nvidia 10-Ks).

### Example Queries

**Single Company Analysis:**
- "What were Tesla's main revenue drivers in 2024?"
- "Summarize Nvidia's AI strategy"
- "What risks did Apple mention in their latest 10-K?"

**Comparative Analysis:**
- "Compare Tesla and Apple's R&D spending trends"
- "Which company has higher profit margins: Nvidia or Tesla?"
- "Compare the revenue growth of all three companies"

**Deep Dives:**
- "What percentage of Nvidia's revenue comes from data centers?"
- "How has Tesla's automotive revenue changed year-over-year?"
- "What strategic initiatives did Apple announce?"

---

## 💡 Technical Highlights

### RAG Architecture
Implements retrieval-augmented generation to ensure factual, grounded responses with source attribution.

### Forced Balanced Retrieval
For comparative queries, uses separate searches per company to ensure balanced representation in retrieved context.

### Smart Chunking
Documents are split into 1000-character chunks with 200-character overlap for optimal context preservation.

### In-Memory Vector Store
Uses ChromaDB without disk persistence for:
- Zero file permission issues
- Faster operation (no I/O)
- Cleaner session management
- Cross-platform compatibility

### Cost Optimization
- GPT-4o-mini: ~$0.01 per query
- text-embedding-3-small: ~$0.002 per 10 documents
- Total cost: ~$0.02 per comparative analysis

### Production Features
- Rate limiting (10 queries/hour)
- Error handling with detailed feedback
- Session state management
- Document deduplication
- Intelligent company name extraction

---

## 📁 Project Structure
```
financial-analyst-ai/
├── app.py                      # Main Streamlit application
├── src/
│   ├── document_processor.py   # PDF processing & chunking
│   ├── vector_store.py         # ChromaDB vector store
│   └── llm_chain.py           # LangChain RAG chain
├── data/
│   └── 10k_reports/           # Demo financial documents
│       ├── 10-K TESLA.pdf
│       ├── 10-K APPLE.pdf
│       └── 10-K NVIDIA.pdf
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in git)
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 🔒 Privacy & Security

### Data Handling
- Documents processed in real-time (not stored permanently on disk)
- Vector database exists only in memory during session
- All data cleared when "Start New Analysis" is clicked
- No user data retention beyond active session

### API Security
- OpenAI API calls encrypted in transit (HTTPS)
- API keys stored in environment variables (never in code)
- Rate limiting prevents abuse
- No credentials stored in logs

### Compliance Notes
- Suitable for non-sensitive financial documents
- For confidential documents, consider self-hosted LLM alternatives
- Always review company data policies before uploading proprietary information

---

## 🎯 Roadmap

### Planned Features
- [ ] Streaming responses (real-time text generation)
- [ ] Multi-modal support (extract tables, charts from PDFs)
- [ ] Chat history persistence (save across sessions)
- [ ] Document comparison view (side-by-side)
- [ ] Export to PDF (formatted reports with charts)
- [ ] Custom prompt templates (user-configurable)
- [ ] Multiple language support
- [ ] Excel/CSV data visualization
- [ ] User authentication (personalized document collections)
- [ ] API endpoint (programmatic access)

### Known Limitations
- Max file size: 10MB per document
- PDF format only (no Word, Excel, HTML)
- Text-based PDFs only (scanned images not supported)
- Rate limit: 10 queries per hour
- English language only

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **OpenAI** for GPT-4 and embeddings API
- **Streamlit** for the web framework
- **ChromaDB** for vector storage

---

## 👤 Author

- LinkedIn: [linkedin.com/in/leaalonzo](https://linkedin.com/in/leaalonzo)
- GitHub: [@leaalonzo](https://github.com/leaalonzo)


---

**⭐ If you find this project helpful, please consider giving it a star!**
