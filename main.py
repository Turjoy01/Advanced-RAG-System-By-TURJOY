"""
Advanced RAG System with Streamlit Interface
Supports 500+ page documents with semantic chunking and retrieval
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

# Set page config FIRST, before any other streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import our modules with fallbacks
try:
    from utils import setup_logging, format_sources, validate_api_key
    # Removed: from document_compare import DocumentComparer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all files are in the same directory and all modules are complete.")
    st.stop()

def handle_streamlit_error(error: Exception, context: str = "") -> str:
    """Handle and format errors for Streamlit display"""

    error_msg = str(error)

    # Common error patterns and user-friendly messages
    if "api_key" in error_msg.lower():
        return "❌ Invalid or missing OpenAI API key. Please check your API key and try again."

    elif "connection" in error_msg.lower() or "network" in error_msg.lower():
        return "❌ Network connection error. Please check your internet connection and try again."

    elif "permission" in error_msg.lower():
        return "❌ Permission denied. Please check file permissions or API key permissions."

    elif "not found" in error_msg.lower():
        return "❌ Resource not found. The file or document may have been moved or deleted."

    elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
        return "❌ Memory error. The document may be too large. Try processing a smaller document."

    elif "timeout" in error_msg.lower():
        return "❌ Operation timed out. Please try again or try with a smaller document."

    else:
        # Generic error with context
        if context:
            return f"❌ Error in {context}: {error_msg}"
        else:
            return f"❌ An error occurred: {error_msg}"

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
# Reduced logging level for chromadb telemetry to avoid noise
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

class RAGSystemManager:
    """Manages the RAG system components with a single Q&A processing pipeline"""

    def __init__(self):
        self.api_key = None
        self.components_loaded = False
        self.document_processor = None
        # Removed: self.comparison_processor = None
        self.vector_store = None
        self.retrieval_system = None
        self.llm_interface = None
        # Removed: self.document_comparer = None

    def initialize_components(self, api_key: str) -> Tuple[bool, str]:
        """Initialize all RAG components (Q&A pipeline only)"""
        try:
            # Set API key
            self.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key

            # Test OpenAI connection first with simple initialization
            import openai
            try:
                # Simple client initialization without extra parameters
                client = openai.OpenAI(api_key=api_key)

                # Test with a simple call
                test_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                logger.info("OpenAI connection test successful")

            except Exception as openai_error:
                logger.error(f"OpenAI connection test failed: {openai_error}")
                return False, f"❌ OpenAI connection failed: {str(openai_error)}"

            # Import and initialize components
            try:
                from document_processor import DocumentProcessor
                from vector_store import VectorStore
                from retrieval_system import RetrievalSystem
                from llm_interface import LLMInterface
                # Removed: from comparison_processor import ComparisonDocumentProcessor

                self.document_processor = DocumentProcessor()
                logger.info("DocumentProcessor initialized")

                # Removed: self.comparison_processor = ComparisonDocumentProcessor()
                # Removed: logger.info("ComparisonDocumentProcessor initialized")

                self.vector_store = VectorStore()
                logger.info("VectorStore initialized") # Message changed

                self.retrieval_system = RetrievalSystem(self.vector_store)
                logger.info("RetrievalSystem initialized")

                self.llm_interface = LLMInterface(api_key)
                logger.info("LLMInterface initialized")

                # Removed: self.document_comparer = DocumentComparer(self.vector_store, self.llm_interface)
                # Removed: logger.info("DocumentComparer with dual pipeline initialized")

            except Exception as component_error:
                logger.error(f"Component initialization failed: {component_error}")
                return False, f"❌ Component initialization failed: {str(component_error)}"

            self.components_loaded = True
            return True, "✅ RAG system initialized successfully!" # Message changed

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False, handle_streamlit_error(e, "system initialization")

    def process_document_sync(self, file_path: str, progress_callback=None) -> Tuple[bool, str, Dict]:
        """Process document for Q&A pipeline only"""
        try:
            if not self.components_loaded:
                return False, "System not initialized", {}

            # Create and run event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Extract metadata
                metadata = loop.run_until_complete(
                    self.document_processor.extract_metadata(file_path)
                )
                if progress_callback:
                    progress_callback(0.1, "Metadata extracted")

                # Process document for Q&A
                qa_chunks = loop.run_until_complete(
                    self.document_processor.process_document(file_path,
                        lambda p, m: progress_callback(0.1 + p * 0.8, f"Processing: {m}") if progress_callback else None) # Adjusted progress range for single pipeline
                )
                if progress_callback:
                    progress_callback(0.9, f"Generated {len(qa_chunks)} Q&A chunks")

                # Store Q&A chunks in vector database
                doc_id = loop.run_until_complete(
                    self.vector_store.add_document(qa_chunks, metadata,
                        lambda p, m: progress_callback(0.9 + p * 0.1, f"Storing: {m}") if progress_callback else None) # Adjusted progress range
                )
                if progress_callback:
                    progress_callback(1.0, "Processing complete!") # Message changed

                stats = {
                    "doc_id": doc_id,
                    "num_qa_chunks": len(qa_chunks),
                    # Removed: "comparison_processed": comparison_success,
                    "title": metadata.title,
                    "pages": metadata.page_count,
                    "size_mb": round(metadata.file_size / (1024*1024), 2),
                    "processing_method": "qa_pipeline" # Changed
                }

                success_msg = f"Document '{metadata.title}' processed successfully for Q&A!" # Message changed

                return True, success_msg, stats

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Document processing failed: {e}") # Message changed
            return False, handle_streamlit_error(e, "document processing"), {}

    def query_documents_sync(self, query: str, num_results: int = 5) -> Tuple[bool, str, List[Dict]]:
        """Query documents synchronously for Streamlit"""
        try:
            if not self.components_loaded:
                return False, "System not initialized", []

            # Create and run event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Retrieve relevant chunks
                retrieved_chunks = loop.run_until_complete(
                    self.retrieval_system.retrieve(query, k=num_results * 2)
                )

                if not retrieved_chunks:
                    return True, "No relevant documents found for your query.", []

                # Rerank using MMR
                reranked_chunks = loop.run_until_complete(
                    self.retrieval_system.rerank_mmr(query, retrieved_chunks, k=num_results)
                )

                # Generate response
                response = loop.run_until_complete(
                    self.llm_interface.generate_response(query, reranked_chunks)
                )

                # Format sources
                sources = format_sources(reranked_chunks)

                result = {
                    "answer": response,
                    "sources": sources,
                    "num_sources": len(sources)
                }

                return True, response, [result]

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return False, handle_streamlit_error(e, "query processing"), []

    # Removed: def compare_documents_sync(...) method

    def get_document_list(self) -> List[Dict]:
        """Get list of indexed documents"""
        if not self.components_loaded:
            return []

        try:
            return self.vector_store.get_document_list()
        except Exception as e:
            logger.error(f"Failed to get document list: {e}")
            return []

def initialize_session_state():
    """Initialize all session state variables"""
    # Ensure initialization happens only once per session
    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = RAGSystemManager()

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

# Removed: render_document_comparison_tab function

def main():
    """Main Streamlit application"""

    # Initialize session state first
    initialize_session_state()

    # Header
    st.title("🚀 Advanced RAG System By TURJOY")
    st.markdown("""
    **Retrieval-Augmented Generation**

    Features: 📄 Process large PDF documents • 🔍 Semantic search • 🎯 AI-powered responses
    """) # Features updated

    # Sidebar for system setup
    with st.sidebar:
        st.header("🔑 System Setup")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key to initialize the system"
        )

        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            if api_key and validate_api_key(api_key):
                with st.spinner("Initializing RAG system..."):
                    success, message = st.session_state.rag_manager.initialize_components(api_key)

                if success:
                    st.session_state.initialized = True
                    st.success(message)
                    # Use experimental_rerun for older Streamlit versions
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter a valid OpenAI API key (starts with 'sk-')")

        # System status
        if st.session_state.initialized:
            st.success("✅ System Ready")

            # Document stats
            docs = st.session_state.rag_manager.get_document_list()
            st.metric("Documents Indexed", len(docs))

            if docs:
                total_qa_chunks = sum(doc.get('qa_chunk_count', doc.get('chunk_count', 0)) for doc in docs)
                # Removed: total_comp_chunks = sum(doc.get('comparison_chunk_count', 0) for doc in docs)
                st.metric("Total Chunks", total_qa_chunks) # Changed metric name
                # Removed: st.metric("Comparison Chunks", total_comp_chunks)
        else:
            st.warning("❌ System Not Initialized")

    # Main content area
    if not st.session_state.initialized:
        st.info("👈 Please initialize the system with your OpenAI API key in the sidebar.")

        # Getting started guide
        st.markdown("""
        ### 🚀 Getting Started

        1.  **Get an OpenAI API Key**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
        2.  **Enter your API key** in the sidebar
        3.  **Click Initialize System** to start
        4.  **Upload PDF documents** to begin processing

        ### 📋 System Requirements

        Make sure you have installed all required packages:

        ```bash
        pip install streamlit openai chromadb pypdf python-dotenv pandas numpy scikit-learn tiktoken
        ```
        """) # Updated requirements
        return

    # Main tabs - Adjusted tabs (removed comparison tab)
    tab1, tab2, tab3 = st.tabs([
        "📄 Document Processing",
        "🔍 Query Documents",
        "ℹ️ System Info"
    ])

    with tab1:
        st.header("📄 Document Processing")
        st.markdown("Upload and process PDF documents for AI-powered analysis")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload PDF documents for processing (supports large documents 500+ pages)"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([2, 1])

            with col1:
                file_size_mb = uploaded_file.size / (1024*1024)
                st.info(f"📎 **File**: {uploaded_file.name}")
                st.info(f"📊 **Size**: {file_size_mb:.2f} MB")

                if file_size_mb > 50:
                    st.warning("⚠️ Large file detected. Processing may take several minutes.")

            with col2:
                if st.button("🔄 Process Document", type="primary"):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    try:
                        # Progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(progress, message):
                            progress_bar.progress(progress)
                            status_text.text(message)

                        # Process document
                        with st.spinner("Processing document..."):
                            success, message, stats = st.session_state.rag_manager.process_document_sync(
                                tmp_file_path,
                                progress_callback=progress_callback
                            )

                        if success:
                            st.success(message)

                            # Display stats
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📄 Pages", stats['pages'])
                            with col2:
                                st.metric("🧩 Chunks", stats['num_qa_chunks'])
                            with col3:
                                st.metric("💾 Size (MB)", stats['size_mb'])
                            with col4:
                                st.metric("🆔 Doc ID", stats['doc_id'][:8] + "...")

                            st.info("✅ Document is now ready for querying!") # Message changed
                        else:
                            st.error(message)

                    except Exception as e:
                        st.error(handle_streamlit_error(e, "document processing"))

                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)

        # Show existing documents
        docs = st.session_state.rag_manager.get_document_list()
        if docs:
            st.subheader("📚 Indexed Documents")

            # Create DataFrame for display
            df_data = []
            for doc in docs:
                df_data.append({
                    "Title": doc.get('title', 'Unknown')[:50] + "..." if len(doc.get('title', '')) > 50 else doc.get('title', 'Unknown'),
                    "Pages": doc.get('page_count', 0),
                    "Chunks": doc.get('qa_chunk_count', doc.get('chunk_count', 0)), # Renamed to 'Chunks'
                    # Removed: "Comparison Chunks": doc.get('comparison_chunk_count', 0),
                    "Size (MB)": round(doc.get('file_size', 0) / (1024*1024), 2),
                    "Processed": doc.get('processing_date', '')[:10] if doc.get('processing_date') else 'Unknown'
                })

            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)

    with tab2:
        st.header("🔍 Query Documents")
        st.markdown("Ask questions about your processed documents")

        # Check if documents exist
        docs = st.session_state.rag_manager.get_document_list()
        if not docs:
            st.warning("📝 No documents have been processed yet. Please upload and process documents in the 'Document Processing' tab first.")
            return

        # Query input
        col1, col2 = st.columns([4, 1])

        with col1:
            query = st.text_area(
                "💬 Enter your query",
                placeholder="What is the main topic discussed in the documents?",
                height=100,
                help="Ask specific questions about your documents for best results"
            )

        with col2:
            num_results = st.slider(
                "📊 Sources",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of source chunks to retrieve"
            )

            search_button = st.button("🔍 Search", type="primary")

        # Example queries
        st.markdown("**💡 Example queries:**")
        example_queries = [
            "What are the main findings?",
            "Can you summarize the methodology?",
            "What are the key recommendations?",
            "What data sources were used?",
            "What are the limitations of this study?"
        ]

        cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            with cols[i]:
                if st.button(f"📌 {example}", key=f"example_{i}"):
                    st.session_state.selected_query = example

        # Use selected query if available
        if 'selected_query' in st.session_state:
            query = st.session_state.selected_query
            del st.session_state.selected_query

        if search_button and query:
            with st.spinner("🔍 Searching documents and generating response..."):
                success, response, results = st.session_state.rag_manager.query_documents_sync(
                    query, num_results
                )

                if success and results:
                    result = results[0]

                    # Display answer
                    st.subheader("📋 Answer")
                    st.markdown(result['answer'])

                    # Display sources
                    st.subheader("📚 Sources")
                    st.markdown(f"*Based on {result['num_sources']} source(s):*")

                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"📄 Source {i}: {source['title']} (Page {source['page']}) - Relevance: {source['score']:.3f}"):
                            st.markdown(f"**Section:** {source['section']}")
                            st.markdown("**Content:**")
                            st.write(source['content_preview'])
                else:
                    if success:
                        st.warning("🤔 " + response)
                    else:
                        st.error(response)

        elif search_button:
            st.warning("⚠️ Please enter a query to search")

    # Removed: with tab3: (Document Comparison tab content)

    with tab3: # This is now the "System Info" tab, previously tab4
        st.header("ℹ️ System Information")

        # Technology stack
        st.subheader("🛠️ Technology Stack")
        tech_info = {
            "LLM": "GPT-4o-mini (cost-effective, high-quality responses)",
            "Embeddings": "OpenAI text-embedding-3-small (1536 dimensions)",
            "Vector DB": "ChromaDB (persistent storage)",
            "Chunking": "Token-based semantic splitter",
            "Retrieval": "Hybrid search with MMR reranking",
            # Removed: "Comparison": "Semantic similarity with AI-powered conflict detection",
            "Interface": "Streamlit web application"
        }

        for tech, desc in tech_info.items():
            st.markdown(f"- **{tech}**: {desc}")

        # System capabilities
        st.subheader("📊 Capabilities")
        capabilities = [
            "Process documents up to 500+ pages",
            "Semantic chunking preserving context",
            "Hybrid retrieval (semantic + text search)",
            "MMR reranking for diversity",
            "Source traceability with page numbers",
            # Removed: "Document comparison with conflict detection",
            # Removed: "AI-powered similarity analysis",
            "Batch processing for efficiency"
        ]

        for capability in capabilities:
            st.markdown(f"✅ {capability}")

        # Performance metrics
        st.subheader("🚀 Performance")
        perf_info = {
            "Batch processing": "100 pages per batch",
            "Token management": "8000 context tokens",
            "Embedding efficiency": "50 chunks per batch",
            "Response optimization": "Speed and accuracy balanced",
            # Removed: "Comparison threshold": "0.8 similarity for conflict detection"
        }

        for metric, value in perf_info.items():
            st.markdown(f"- **{metric}**: {value}")

        # Usage tips
        st.subheader("💡 Usage Tips")
        tips = [
            "📄 Upload PDF documents using the Document Processing tab",
            "⏳ Wait for processing to complete before querying",
            "🎯 Use specific questions for better results",
            "📚 Check Sources section for page numbers and confidence scores",
            "🔄 Try different phrasings if results aren't satisfactory",
            # Removed: "🆚 Compare similar documents for best comparison results",
            # Removed: "⚖️ Review conflicts carefully as they require human judgment"
        ]

        for tip in tips:
            st.markdown(f"{tip}")

        # Current system stats
        if st.session_state.initialized:
            st.subheader("📈 Current System Stats")
            docs = st.session_state.rag_manager.get_document_list()

            if docs:
                # Summary stats
                total_pages = sum(doc.get('page_count', 0) for doc in docs)
                total_qa_chunks = sum(doc.get('qa_chunk_count', doc.get('chunk_count', 0)) for doc in docs)
                # Removed: total_comp_chunks = sum(doc.get('comparison_chunk_count', 0) for doc in docs)
                total_size_mb = sum(doc.get('file_size', 0) for doc in docs) / (1024*1024)

                col1, col2, col3 = st.columns(3) # Adjusted columns
                with col1:
                    st.metric("📄 Documents", len(docs))
                with col2:
                    st.metric("📑 Total Pages", total_pages)
                with col3:
                    st.metric("🧩 Total Chunks", total_qa_chunks) # Renamed

                # Additional stats
                st.markdown("**📊 System Statistics:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("💾 Total Size (MB)", f"{total_size_mb:.1f}")
                # Removed: with col2: (for comparison ready count)
                # Removed: st.metric("🔄 Comparison Ready", f"{comparison_ready_count}/{len(docs)}")

                # Document details table
                st.markdown("**📋 Document Details:**")
                df_data = []
                for doc in docs:
                    df_data.append({
                        "Title": doc.get('title', 'Unknown'),
                        "Pages": doc.get('page_count', 0),
                        "Chunks": doc.get('qa_chunk_count', doc.get('chunk_count', 0)), # Renamed
                        # Removed: "Comparison Chunks": doc.get('comparison_chunk_count', 0),
                        "Processing Date": doc.get('processing_date', '')[:10] if doc.get('processing_date') else 'Unknown'
                    })

                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)

                # Removed: Comparison readiness indicator
                # if len(docs) >= 2: ...
            else:
                st.info("📝 No documents processed yet")

        # Troubleshooting section
        st.subheader("🔧 Troubleshooting")

        with st.expander("🚨 Common Issues and Solutions"):
            st.markdown("""
            **Issue: "Failed to initialize OpenAI client"**
            - ✅ Check your API key is correct and starts with 'sk-'
            - ✅ Verify you have sufficient credits in your OpenAI account
            - ✅ Ensure your API key has the necessary permissions

            **Issue: "ChromaDB initialization failed"**
            - ✅ Check you have write permissions in the current directory
            - ✅ Ensure sufficient disk space is available
            - ✅ Try deleting the `chroma_db` folder and restarting

            **Issue: "PDF processing failed"**
            - ✅ Ensure the PDF is not password-protected
            - ✅ Check the PDF is not corrupted
            - ✅ Try with a smaller PDF first to test the system

            **Issue: "No results found"**
            - ✅ Make sure documents are fully processed before querying
            - ✅ Try different query phrasings
            - ✅ Check that your query is related to the document content

            **Issue: "Memory errors"**
            - ✅ Try processing smaller documents
            - ✅ Restart the application to clear memory
            - ✅ Ensure sufficient RAM is available
            """) # Removed comparison specific troubleshooting

        # Removed: System requirements check button and content

if __name__ == "__main__":
    main()