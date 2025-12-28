"""
Document Processor with Semantic Chunking
Fixed for OpenAI 1.51.0 compatibility
"""

import os
import asyncio
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

# Document processing
import pypdf
from llama_index.core import Document
import tiktoken

# For embeddings - we'll handle OpenAI directly instead of through LlamaIndex
import openai

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    title: str
    file_path: str
    file_size: int
    page_count: int
    creation_date: datetime
    processing_date: datetime
    document_type: str
    language: str
    doc_id: str
    sections: List[str]
    
@dataclass
class DocumentChunk:
    """Document chunk structure"""
    chunk_id: str
    content: str
    page_number: int
    section: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    token_count: int = 0

class DocumentProcessor:
    """Advanced document processor with semantic chunking"""
    
    def __init__(self):
        self.embedding_model = None
        self.openai_client = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_chunk_size = 1000  # tokens
        self.chunk_overlap = 200   # tokens
        self.max_pages_per_batch = 100
        
    def _initialize_embedding_model(self):
        """Initialize OpenAI embedding model directly"""
        if self.openai_client is None:
            try:
                # Use the new OpenAI client directly
                self.openai_client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY")
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
    
    async def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract comprehensive metadata from document"""
        try:
            file_stats = os.stat(file_path)
            
            # Read PDF metadata
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Extract basic info
                num_pages = len(pdf_reader.pages)
                
                # Try to extract title from metadata or first page
                title = "Unknown Document"
                if pdf_reader.metadata and pdf_reader.metadata.title:
                    title = pdf_reader.metadata.title
                else:
                    # Extract from first page text
                    first_page_text = pdf_reader.pages[0].extract_text()
                    lines = first_page_text.split('\n')
                    for line in lines[:5]:  # Check first 5 lines
                        if len(line.strip()) > 10 and len(line.strip()) < 100:
                            title = line.strip()
                            break
                
                # Generate document ID
                doc_id = hashlib.md5(
                    f"{file_path}_{file_stats.st_mtime}".encode()
                ).hexdigest()[:12]
                
                # Extract sections (simplified - could be enhanced)
                sections = await self._extract_sections(pdf_reader)
                
                metadata = DocumentMetadata(
                    title=title,
                    file_path=file_path,
                    file_size=file_stats.st_size,
                    page_count=num_pages,
                    creation_date=datetime.fromtimestamp(file_stats.st_ctime),
                    processing_date=datetime.now(),
                    document_type="PDF",
                    language="en",  # Could be enhanced with language detection
                    doc_id=doc_id,
                    sections=sections
                )
                
                logger.info(f"Extracted metadata for {title}: {num_pages} pages")
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}")
            raise
    
    async def _extract_sections(self, pdf_reader) -> List[str]:
        """Extract section headers from PDF"""
        sections = []
        
        try:
            # Simple section detection based on formatting patterns
            for page_num, page in enumerate(pdf_reader.pages[:10]):  # Check first 10 pages
                text = page.extract_text()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # Look for potential section headers
                    if (len(line) > 5 and len(line) < 80 and
                        (line.isupper() or 
                         any(keyword in line.lower() for keyword in 
                             ['chapter', 'section', 'introduction', 'conclusion', 
                              'abstract', 'summary', 'background', 'methodology']))):
                        if line not in sections:
                            sections.append(line)
                            
        except Exception as e:
            logger.warning(f"Section extraction failed: {str(e)}")
        
        return sections[:20]  # Limit to 20 sections
    
    async def process_document(
        self, 
        file_path: str, 
        progress_callback=None
    ) -> List[DocumentChunk]:
        """Process document with semantic chunking"""
        
        self._initialize_embedding_model()
        
        try:
            # Extract text from PDF
            document_text = await self._extract_text_from_pdf(
                file_path, 
                progress_callback
            )
            
            if progress_callback:
                progress_callback(0.3, "Text extracted, starting chunking...")
            
            # Create chunks (using simple chunking instead of semantic for now)
            chunks = await self._create_simple_chunks(
                document_text, 
                file_path,
                progress_callback
            )
            
            if progress_callback:
                progress_callback(0.8, "Generating embeddings...")
            
            # Generate embeddings for chunks
            chunks = await self._generate_embeddings(chunks, progress_callback)
            
            logger.info(f"Generated {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
    
    async def _extract_text_from_pdf(
        self, 
        file_path: str, 
        progress_callback=None
    ) -> List[Tuple[str, int]]:
        """Extract text from PDF in batches"""
        
        text_pages = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Process in batches to handle large documents
                for batch_start in range(0, total_pages, self.max_pages_per_batch):
                    batch_end = min(batch_start + self.max_pages_per_batch, total_pages)
                    
                    # Extract text from current batch
                    for page_num in range(batch_start, batch_end):
                        try:
                            page_text = pdf_reader.pages[page_num].extract_text()
                            
                            # Clean and validate text
                            cleaned_text = self._clean_text(page_text)
                            if cleaned_text.strip():
                                text_pages.append((cleaned_text, page_num + 1))
                                
                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                            continue
                    
                    if progress_callback:
                        progress = 0.1 + (batch_end / total_pages) * 0.2
                        progress_callback(progress, f"Extracted {batch_end}/{total_pages} pages")
                
                logger.info(f"Extracted text from {len(text_pages)} pages")
                return text_pages
                
        except Exception as e:
            logger.error(f"PDF text extraction failed: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:  # Filter out very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    async def _create_simple_chunks(
        self, 
        text_pages: List[Tuple[str, int]], 
        file_path: str,
        progress_callback=None
    ) -> List[DocumentChunk]:
        """Create simple token-based chunks"""
        
        chunks = []
        
        try:
            total_pages = len(text_pages)
            
            for idx, (page_text, page_num) in enumerate(text_pages):
                # Simple token-based chunking
                tokens = self.tokenizer.encode(page_text)
                
                for i in range(0, len(tokens), self.max_chunk_size - self.chunk_overlap):
                    chunk_tokens = tokens[i:i + self.max_chunk_size]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    
                    chunk_id = f"{Path(file_path).stem}_{page_num}_{i // (self.max_chunk_size - self.chunk_overlap)}"
                    
                    # Create chunk
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        page_number=page_num,
                        section=self._identify_section(chunk_text),
                        metadata={
                            "source": file_path,
                            "page_number": page_num,
                            "chunk_index": i // (self.max_chunk_size - self.chunk_overlap),
                            "token_count": len(chunk_tokens),
                            "token_start": i,
                            "token_end": i + len(chunk_tokens),
                            "chunking_method": "simple"
                        },
                        token_count=len(chunk_tokens)
                    )
                    
                    chunks.append(chunk)
                
                if progress_callback and idx % 10 == 0:
                    progress = 0.3 + (idx / total_pages) * 0.5
                    progress_callback(progress, f"Processed {idx}/{total_pages} pages")
            
            logger.info(f"Created {len(chunks)} simple chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Simple chunking failed: {str(e)}")
            raise
    
    def _identify_section(self, text: str) -> str:
        """Identify which section this chunk belongs to"""
        text_lower = text.lower()
        
        # Simple section identification
        if any(keyword in text_lower for keyword in ['abstract', 'summary']):
            return "abstract"
        elif any(keyword in text_lower for keyword in ['introduction', 'intro']):
            return "introduction"
        elif any(keyword in text_lower for keyword in ['methodology', 'method']):
            return "methodology"
        elif any(keyword in text_lower for keyword in ['results', 'findings']):
            return "results"
        elif any(keyword in text_lower for keyword in ['conclusion', 'summary']):
            return "conclusion"
        elif any(keyword in text_lower for keyword in ['references', 'bibliography']):
            return "references"
        else:
            return "content"
    
    async def _generate_embeddings(
        self, 
        chunks: List[DocumentChunk], 
        progress_callback=None
    ) -> List[DocumentChunk]:
        """Generate embeddings for chunks using OpenAI directly"""
        
        if not self.openai_client:
            logger.warning("OpenAI client not available, skipping embeddings")
            return chunks
        
        try:
            # Batch process embeddings for efficiency
            batch_size = 50  # Process 50 chunks at a time
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch_chunks]
                
                try:
                    # Generate embeddings using OpenAI client directly
                    response = self.openai_client.embeddings.create(
                        input=batch_texts,
                        model="text-embedding-3-small"
                    )
                    
                    # Assign embeddings to chunks
                    for chunk, embedding_data in zip(batch_chunks, response.data):
                        chunk.embedding = embedding_data.embedding
                    
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings for batch {i}: {e}")
                    # Continue without embeddings for this batch
                
                if progress_callback:
                    progress = 0.8 + (min(i + batch_size, total_chunks) / total_chunks) * 0.2
                    progress_callback(progress, f"Embedded {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            # Return chunks without embeddings (will be generated later)
            return chunks