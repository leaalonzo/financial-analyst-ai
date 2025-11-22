import streamlit as st
import os
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor
from src.vector_store import FinancialVectorStore
from src.llm_chain import FinancialAnalystChain
from datetime import datetime, timedelta
import traceback  # ← Already imported here, don't import again
import shutil  # ← Add this here too

load_dotenv()

st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_company_name(filename: str) -> str:
    """Extract company name from filename intelligently"""
    name = filename.replace('.pdf', '').replace('.PDF', '')
    
    company_keywords = {
        'TESLA': 'Tesla', 'TSLA': 'Tesla',
        'NVIDIA': 'Nvidia', 'NVDA': 'Nvidia',
        'APPLE': 'Apple', 'AAPL': 'Apple',
        'MICROSOFT': 'Microsoft', 'MSFT': 'Microsoft',
        'GOOGLE': 'Google', 'GOOGL': 'Google',
        'AMAZON': 'Amazon', 'AMZN': 'Amazon',
        'META': 'Meta', 'FB': 'Meta',
    }
    
    name_upper = name.upper()
    for keyword, proper_name in company_keywords.items():
        if keyword in name_upper:
            return proper_name
    
    for separator in ['_', '-', ' ']:
        if separator in name:
            parts = name.split(separator)
            skip_words = {'10K', '10-K', 'EC', 'AR', 'EARNINGS', 'CALL', 
                         'REPORT', 'ANNUAL', 'Q1', 'Q2', 'Q3', 'Q4', 
                         '2023', '2024', '2025', 'FY'}
            
            for part in parts:
                part_clean = part.strip().upper()
                if part_clean and part_clean not in skip_words:
                    for keyword, proper_name in company_keywords.items():
                        if keyword in part_clean:
                            return proper_name
                    return part.strip().title()
    
    first_word = name.split()[0].strip() if name.split() else name
    return first_word.title()

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'last_reset' not in st.session_state:
    st.session_state.last_reset = datetime.now()
if 'total_documents_processed' not in st.session_state:
    st.session_state.total_documents_processed = 0
if 'total_queries_asked' not in st.session_state:
    st.session_state.total_queries_asked = 0
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = datetime.now()
if 'processed_chunks' not in st.session_state:
    st.session_state.processed_chunks = []
if 'loaded_file_names' not in st.session_state:
    st.session_state.loaded_file_names = set()

def check_rate_limit():
    """Rate limiting: 10 queries per hour"""
    if datetime.now() - st.session_state.last_reset > timedelta(hours=1):
        st.session_state.query_count = 0
        st.session_state.last_reset = datetime.now()
    
    if st.session_state.query_count >= 10:
        st.error("⏳ Rate limit reached (10 queries/hour). Please try again later.")
        return False
    
    st.session_state.query_count += 1
    st.session_state.total_queries_asked += 1
    return True

def main():
    # Header
    st.title("🤖 AI Financial Analyst Assistant")
    st.markdown("""
    Upload financial documents (10-Ks, earnings calls, analyst reports) and ask questions.
    The AI will analyze them and provide insights with citations.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        if st.session_state.loaded_file_names:
            st.info(f"📎 {len(st.session_state.loaded_file_names)} files currently loaded")
        
        uploaded_files = st.file_uploader(
            "Upload Financial Documents (PDF)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Max 10MB per file. Already loaded files will be skipped."
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("📄 Processing documents..."):
                    try:
                        processor = DocumentProcessor()
                        all_chunks = []
                        processed_files = []
                        failed_files = []
                        skipped_files = []
                        
                        for file in uploaded_files:
                            if file.name in st.session_state.loaded_file_names:
                                skipped_files.append(file.name)
                                continue
                            
                            if file.size > 10 * 1024 * 1024:
                                failed_files.append((file.name, "File too large"))
                                continue
                            
                            try:
                                temp_path = f"temp_{file.name}"
                                with open(temp_path, "wb") as f:
                                    f.write(file.getbuffer())
                                
                                company = extract_company_name(file.name)
                                chunks = processor.process_document(
                                    temp_path, "Financial Report", company
                                )
                                
                                if chunks:
                                    all_chunks.extend(chunks)
                                    processed_files.append(file.name)
                                    st.session_state.loaded_file_names.add(file.name)
                                else:
                                    failed_files.append((file.name, "No text extracted"))
                                
                                os.remove(temp_path)
                            except Exception as e:
                                failed_files.append((file.name, str(e)))
                        
                        if all_chunks:
                            st.session_state.processed_chunks.extend(all_chunks)
                        
                        # Document upload section
                        if st.session_state.processed_chunks:
                            with st.spinner("🔢 Building vector database..."):
                                # ← SIMPLIFIED: No disk cleanup needed
                                vector_store = FinancialVectorStore()
                                vector_store.create_vectorstore(st.session_state.processed_chunks)
                                st.session_state.vector_store = vector_store
                                st.session_state.qa_chain = FinancialAnalystChain(vector_store)
                                
                                st.write(f"✓ Vector store created with {len(st.session_state.processed_chunks)} chunks")
                            
                            if processed_files:
                                st.session_state.total_documents_processed += len(processed_files)
                                st.success(f"✅ Processed {len(processed_files)} new documents!")
                            
                            with st.expander("📋 Details"):
                                if processed_files:
                                    st.write("**New files:**")
                                    for f in processed_files:
                                        st.write(f"✓ {f}")
                                if skipped_files:
                                    st.write("\n**Already loaded:**")
                                    for f in skipped_files:
                                        st.write(f"↻ {f}")
                                st.caption(f"Total: {len(st.session_state.processed_chunks)} chunks")
                        
                        if failed_files:
                            st.warning(f"⚠️ Failed: {len(failed_files)} files")
                            for fname, reason in failed_files:
                                st.caption(f"✗ {fname}: {reason}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.error(traceback.format_exc())  # ← Fixed: no import statement
        
        # Demo documents
        st.markdown("---")
        st.subheader("🎯 Demo Documents")
        if st.button("Load Samples"):
            with st.spinner("Loading demo documents..."):
                try:
                    sample_files = [
                        ("data/10k_reports/10-K TESLA.pdf", "10-K", "Tesla"),
                        ("data/10k_reports/10-K APPLE.pdf", "10-K", "Apple"),
                        ("data/10k_reports/10-K NVIDIA.pdf", "10-K", "Nvidia"),
                        ("data/earnings_calls/EC-TESLA.pdf", "Earnings Call", "Tesla"),
                    ]
                    
                    processor = DocumentProcessor()
                    all_chunks = []
                    loaded_count = 0
                    
                    for file_path, doc_type, company in sample_files:
                        if os.path.exists(file_path):
                            filename = os.path.basename(file_path)
                            
                            if filename in st.session_state.loaded_file_names:
                                continue
                            
                            chunks = processor.process_document(file_path, doc_type, company)
                            all_chunks.extend(chunks)
                            loaded_count += 1
                            st.session_state.loaded_file_names.add(filename)
                    
                    if all_chunks:
                        st.session_state.processed_chunks.extend(all_chunks)
                    
                    # Demo documents section
                    if st.session_state.processed_chunks:
                        # ← SIMPLIFIED: No disk cleanup needed
                        vector_store = FinancialVectorStore()
                        vector_store.create_vectorstore(st.session_state.processed_chunks)
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = FinancialAnalystChain(vector_store)
                        st.session_state.total_documents_processed += loaded_count
                                
                        if loaded_count > 0:
                            st.success(f"✅ Loaded {loaded_count} new demo documents!")
                        else:
                            st.info("ℹ️ Demo documents already loaded")
                    else:
                        st.warning("⚠️ No demo documents found")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error(traceback.format_exc())  # ← Fixed: no import statement
        
        # Loaded documents display
        if st.session_state.vector_store:
            st.markdown("---")
            st.subheader("📚 Loaded Documents")
            
            try:
                all_docs = st.session_state.vector_store.vectorstore.get()
                
                if all_docs and 'metadatas' in all_docs:
                    companies = {}
                    doc_types = {}
                    sources = set()
                    
                    for metadata in all_docs['metadatas']:
                        company = metadata.get('company', 'Unknown')
                        doc_type = metadata.get('type', 'Unknown')
                        source = metadata.get('source', 'Unknown')
                        
                        companies[company] = companies.get(company, 0) + 1
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                        
                        if source != 'Unknown':
                            filename = os.path.basename(source)
                            if filename.startswith('temp_'):
                                filename = filename[5:]
                            sources.add(filename)
                    
                    st.markdown("**📊 By Company:**")
                    for company, count in sorted(companies.items()):
                        percentage = (count / len(all_docs['metadatas'])) * 100
                        st.write(f"• **{company}**: {count} chunks ({percentage:.0f}%)")
                    
                    st.markdown("\n**📄 By Document Type:**")
                    for dtype, count in sorted(doc_types.items()):
                        st.write(f"• {dtype}: {count} chunks")
                    
                    if sources:
                        with st.expander("📁 View Source Files"):
                            for source in sorted(sources):
                                st.caption(f"• {source}")
                    
                    st.markdown("---")
                    st.info(f"**Total:** {len(all_docs['metadatas'])} chunks from {len(sources)} files")
                else:
                    st.info("✅ Documents loaded and ready")
            
            except Exception as e:
                st.info("✅ Documents loaded and ready")
        
        # About section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **Technology:**
        - 🤖 GPT-4o-mini
        - 🗄️ ChromaDB
        - 🔗 LangChain
        - 📊 Streamlit
        
        **Limits:**
        - 10 queries/hour
        - 10MB max file size
        - PDF format only
        """)
        
        progress = st.session_state.query_count / 10
        st.progress(progress)
        st.caption(f"Queries: {st.session_state.query_count}/10 this hour")
        
        st.markdown("---")
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Docs", st.session_state.total_documents_processed)
        col2.metric("Queries", st.session_state.total_queries_asked)
        
        session_duration = datetime.now() - st.session_state.session_start_time
        st.caption(f"Duration: {str(session_duration).split('.')[0]}")
    
    # Main content
    if st.session_state.qa_chain:
        st.header("💬 Ask Questions")
        
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - What were the main revenue drivers for Tesla in 2023?
            - Compare Apple and Tesla's R&D spending trends
            - What risks did the companies mention?
            - Summarize key strategic initiatives
            """)
        
        user_query = st.text_input(
            "Ask a question:",
            placeholder="e.g., What were Tesla's revenue growth drivers?",
            key="main_query"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        if analyze_button and user_query:
            if not check_rate_limit():
                st.stop()
            
            with st.spinner("🧠 Analyzing..."):
                try:
                    result = st.session_state.qa_chain.analyze_query(user_query)
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "result": result,
                        "timestamp": datetime.now()
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error(traceback.format_exc())  # ← Fixed: no import statement
                
        # Add this RIGHT BEFORE the comparative analysis section in app.py:

        # ============================================
        # DEBUG SECTION (TEMPORARY)
        # ============================================
        if st.checkbox("🔍 Enable Debug Mode"):
            st.write("### Debug Information")
            
            # Test retrieval
            if st.session_state.vector_store:
                test_query = "Tesla Apple revenue"
                st.write(f"**Test Query:** {test_query}")
                
                try:
                    retriever = st.session_state.vector_store.vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    )
                    docs = retriever.invoke(query)
                    
                    st.write(f"**Retrieved {len(docs)} documents:**")
                    for i, doc in enumerate(docs):
                        with st.expander(f"Document {i+1}"):
                            st.write(f"**Company:** {doc.metadata.get('company', 'Unknown')}")
                            st.write(f"**Type:** {doc.metadata.get('type', 'Unknown')}")
                            st.write(f"**Content preview:**")
                            st.text(doc.page_content[:300])
                except Exception as e:
                    st.error(f"Debug error: {e}")
            else:
                st.warning("No vector store loaded")

        st.markdown("---")

        # Comparative analysis
        with st.expander("🔬 Comparative Analysis"):
            col1, col2, col3 = st.columns(3)
            with col1:
                company1 = st.text_input("Company 1", "Tesla", key="comp1")
            with col2:
                company2 = st.text_input("Company 2", "Nvidia", key="comp2")
            with col3:
                metric = st.selectbox("Metric", ["Revenue", "R&D", "Margins", "Risks"])
            
            # Test retrieval button
            if st.button("🧪 Test Retrieval"):
                st.write("### Testing Retrieval")
                
                # Test 1: Basic combined search (old way - shows the problem)
                st.write("**Method 1: Combined Search** (old way)")
                test_query = f"{company1} {company2} {metric}"
                st.write(f"Query: `{test_query}`")
                
                retriever = st.session_state.vector_store.vectorstore.as_retriever(
                    search_kwargs={"k": 10}
                )
                docs = retriever.get_relevant_documents(test_query)
                
                companies_found = {}
                for doc in docs:
                    company = doc.metadata.get('company', 'Unknown')
                    companies_found[company] = companies_found.get(company, 0) + 1
                
                st.write(f"Documents by company: {companies_found}")
                
                # Test 2: Separate searches (new way - shows the fix)
                st.write("\n**Method 2: Separate Searches** (new way)")
                
                all_docs = []
                for company in [company1, company2]:
                    st.write(f"\nSearching for: `{company} {metric}`")
                    
                    company_query = f"{company} {metric}"
                    docs = retriever.get_relevant_documents(company_query)
                    
                    # Filter to this company
                    company_docs = [d for d in docs if d.metadata.get('company', '').lower() == company.lower()]
                    all_docs.extend(company_docs[:5])
                    
                    st.write(f"✓ Found {len(company_docs)} documents from {company}")
                
                # Summary
                combined_companies = {}
                for doc in all_docs:
                    company = doc.metadata.get('company', 'Unknown')
                    combined_companies[company] = combined_companies.get(company, 0) + 1
                
                st.write(f"\n**Final combined results:** {combined_companies}")
                st.success(f"✅ Both companies represented: {list(combined_companies.keys())}")

            if st.button("⚖️ Compare"):
                if not check_rate_limit():
                    st.stop()
                
                query = f"Compare {company1} and {company2} in terms of {metric}. Be specific."
                
                with st.spinner("Comparing..."):
                    try:
                        # ← CRITICAL: Pass companies for forced balanced retrieval
                        result = st.session_state.qa_chain.analyze_query(
                            query,
                            force_companies=[company1, company2]
                        )
                        
                        st.session_state.chat_history.append({
                            "query": query,
                            "result": result,
                            "timestamp": datetime.now()
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error(traceback.format_exc())
        
        # Display chat history
        def format_timestamp(timestamp):
            try:
                now = datetime.now()
                diff = now - timestamp
                seconds = diff.total_seconds()
                
                if seconds < 60:
                    return "Just now"
                elif seconds < 3600:
                    minutes = int(seconds / 60)
                    return f"{minutes} min ago" if minutes > 1 else "1 min ago"
                elif seconds < 86400:
                    hours = int(seconds / 3600)
                    return f"{hours} hr ago" if hours > 1 else "1 hr ago"
                else:
                    return timestamp.strftime('%b %d at %H:%M')
            except:
                return timestamp.strftime('%H:%M:%S')
        
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📜 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    timestamp = chat['timestamp']
                    time_str = format_timestamp(timestamp)
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**🙋 Question {len(st.session_state.chat_history) - i}**")
                    with col2:
                        st.markdown(f"*{time_str}*")
                    
                    st.markdown(f"> {chat['query']}")
                    st.markdown("**🤖 Analysis:**")
                    st.info(chat['result']['answer'])
                    
                    with st.expander("📄 View Sources"):
                        for j, source in enumerate(chat['result']['sources']):
                            st.markdown(f"**Source {j+1}:**")
                            st.text(source['content'])
                            st.caption(f"Metadata: {source['metadata']}")
                    
                    st.markdown("---")
        
        # Download conversation
        if st.session_state.chat_history:
            col1, col2 = st.columns(2)
            
            with col1:
                report = "AI FINANCIAL ANALYST - CONVERSATION REPORT\n"
                report += "=" * 60 + "\n\n"
                report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                report += f"Total Questions: {len(st.session_state.chat_history)}\n\n"
                
                for i, chat in enumerate(st.session_state.chat_history):
                    ts = chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    report += f"\n{'='*60}\n"
                    report += f"QUESTION {i+1} (Asked at: {ts})\n"
                    report += f"{'='*60}\n\n"
                    report += f"Q: {chat['query']}\n\n"
                    report += f"A: {chat['result']['answer']}\n\n"
                    report += "SOURCES:\n"
                    for j, source in enumerate(chat['result']['sources']):
                        report += f"\nSource {j+1}:\n{source['content']}\n"
                
                st.download_button(
                    "📥 Download Report",
                    report,
                    f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
            
            with col2:
                # Start New Analysis button
                if st.button("🔄 Start New Analysis"):
                    st.session_state.vector_store = None
                    st.session_state.qa_chain = None
                    st.session_state.chat_history = []
                    st.session_state.query_count = 0
                    st.session_state.processed_chunks = []
                    st.session_state.loaded_file_names = set()
                    
                    # ← NO DISK CLEANUP NEEDED
                    
                    st.success("✅ All data cleared!")
                    st.rerun()  
    
    else:
        st.info("👈 Upload documents to begin")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📊 Multi-Document")
            st.write("Analyze multiple reports")
        with col2:
            st.subheader("🎯 Cited Sources")
            st.write("Every answer has sources")
        with col3:
            st.subheader("🧠 AI-Powered")
            st.write("GPT-4 analysis")

if __name__ == "__main__":
    main()