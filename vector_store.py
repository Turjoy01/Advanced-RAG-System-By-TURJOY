"""
Vector Store with ChromaDB
Supports only Q&A chunks (comparison pipeline removed)
"""

import os
import json
import uuid
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path

import chromadb
import numpy as np

from document_processor import DocumentChunk, DocumentMetadata
# Removed: from comparison_processor import ComparisonChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB-based vector store supporting Q&A pipeline only"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.qa_collection = None          # For Q&A chunks
        # Removed: self.comparison_collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and Q&A collection"""
        try:
            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client (Updated for 0.4.18+)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            # Create or get Q&A collection
            self.qa_collection = self.client.get_or_create_collection(
                name="document_chunks",
                metadata={"hnsw:space": "cosine", "description": "Q&A optimized chunks"}
            )

            # Removed: comparison collection creation
            logger.info(f"Initialized ChromaDB with Q&A collection at {self.persist_directory}") # Message changed

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    # ============================================================================
    # Q&A COLLECTION METHODS
    # ============================================================================

    async def add_document(
        self,
        chunks: List[DocumentChunk],
        metadata: DocumentMetadata,
        progress_callback=None
    ) -> str:
        """Add Q&A document chunks to vector store"""

        try:
            # Prepare data for ChromaDB
            documents = []
            embeddings = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                # Prepare chunk data
                documents.append(chunk.content)
                ids.append(chunk.chunk_id)

                # Prepare metadata - ensure all values are JSON serializable
                chunk_metadata = {
                    "doc_id": metadata.doc_id,
                    "doc_title": metadata.title,
                    "page_number": int(chunk.page_number),
                    "section": chunk.section,
                    "token_count": int(chunk.token_count),
                    "chunk_index": int(i),
                    "file_path": str(metadata.file_path),
                    "document_type": metadata.document_type,
                    "processing_date": metadata.processing_date.isoformat(),
                    "chunk_type": "qa_optimized"
                }

                # Add additional metadata from chunk
                for key, value in chunk.metadata.items():
                    if key not in chunk_metadata:
                        # Ensure value is JSON serializable
                        if isinstance(value, (str, int, float, bool)):
                            chunk_metadata[key] = value
                        else:
                            chunk_metadata[key] = str(value)

                metadatas.append(chunk_metadata)

                # Handle embeddings
                if chunk.embedding and len(chunk.embedding) > 0:
                    embeddings.append(chunk.embedding)
                else:
                    # Generate placeholder embedding (will be replaced)
                    embeddings.append([0.0] * 1536)

                if progress_callback and i % 50 == 0:
                    progress = 0.6 + (i / len(chunks)) * 0.3
                    progress_callback(progress, f"Storing chunk {i}/{len(chunks)}") # Message changed

            # Add to Q&A ChromaDB collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))

                try:
                    self.qa_collection.add(
                        documents=documents[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                        ids=ids[i:batch_end]
                    )
                except Exception as e:
                    logger.error(f"Failed to add batch {i}-{batch_end}: {e}") # Message changed
                    # Try adding one by one
                    for j in range(i, batch_end):
                        try:
                            self.qa_collection.add(
                                documents=[documents[j]],
                                embeddings=[embeddings[j]],
                                metadatas=[metadatas[j]],
                                ids=[ids[j]]
                            )
                        except Exception as e2:
                            logger.warning(f"Failed to add chunk {j}: {e2}") # Message changed

            # Store document metadata separately
            await self._store_document_metadata(metadata)

            logger.info(f"Added {len(chunks)} Q&A chunks for document {metadata.doc_id}") # Message changed
            return metadata.doc_id

        except Exception as e:
            logger.error(f"Failed to add document to vector store: {str(e)}") # Message changed
            raise

    # Removed: COMPARISON COLLECTION METHODS (add_comparison_chunks, get_comparison_chunks, search_comparison_similar)

    # ============================================================================
    # SHARED METHODS (USED BY Q&A COLLECTION)
    # ============================================================================

    async def search_similar(
        self,
        query_embedding: List[float],
        k: int = 10,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search Q&A chunks using embeddings"""

        try:
            # Prepare where clause for metadata filtering
            where_clause = {}
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value

            # Query Q&A collection
            results = self.qa_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        "score": 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []

    async def search_by_text(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search Q&A chunks by text query"""

        try:
            # Prepare where clause
            where_clause = {}
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value

            # Query Q&A collection
            results = self.qa_collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'])):
                    result = {
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        "score": 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Text search failed: {str(e)}")
            return []

    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Get all Q&A chunks for a specific document"""

        try:
            results = self.qa_collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"]
            )

            formatted_results = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    result = {
                        "id": results['ids'][i],
                        "content": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    formatted_results.append(result)

            # Sort by chunk index
            formatted_results.sort(key=lambda x: x['metadata'].get('chunk_index', 0))

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get document chunks: {str(e)}")
            return []

    async def _store_document_metadata(self, metadata: DocumentMetadata):
        """Store document metadata in a separate collection"""
        try:
            # Get or create metadata collection
            metadata_collection = self.client.get_or_create_collection(
                name="document_metadata"
            )

            # Prepare metadata document - ensure JSON serializable
            metadata_doc = {
                "title": str(metadata.title),
                "file_path": str(metadata.file_path),
                "file_size": int(metadata.file_size),
                "page_count": int(metadata.page_count),
                "creation_date": metadata.creation_date.isoformat(),
                "processing_date": metadata.processing_date.isoformat(),
                "document_type": str(metadata.document_type),
                "language": str(metadata.language),
                "sections": json.dumps(metadata.sections)
            }

            # Add or update metadata
            metadata_collection.upsert(
                documents=[json.dumps(metadata_doc)],
                metadatas=[metadata_doc],
                ids=[metadata.doc_id]
            )

        except Exception as e:
            logger.warning(f"Failed to store document metadata: {str(e)}")

    def get_document_list(self) -> List[Dict]:
        """Get list of all indexed documents with Q&A collection stats""" # Message changed

        try:
            # Get metadata collection
            metadata_collection = self.client.get_or_create_collection(
                name="document_metadata"
            )

            # Get all documents
            results = metadata_collection.get(include=["metadatas"])

            documents = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]

                    # Count Q&A chunks
                    try:
                        qa_results = self.qa_collection.get(
                            where={"doc_id": doc_id},
                            include=["ids"]
                        )
                        qa_chunk_count = len(qa_results['ids']) if qa_results['ids'] else 0
                    except:
                        qa_chunk_count = 0

                    # Removed: Count comparison chunks
                    doc_info = {
                        "id": doc_id,
                        "title": metadata.get("title", "Unknown"),
                        "page_count": metadata.get("page_count", 0),
                        "chunk_count": qa_chunk_count,  # Q&A chunks for backward compatibility
                        "qa_chunk_count": qa_chunk_count,
                        # Removed: "comparison_chunk_count": comp_chunk_count,
                        "processing_date": metadata.get("processing_date", ""),
                        "file_size": metadata.get("file_size", 0),
                        # Removed: "comparison_ready": comp_chunk_count > 0
                    }
                    documents.append(doc_info)

            # Sort by processing date (newest first)
            documents.sort(
                key=lambda x: x.get("processing_date", ""),
                reverse=True
            )

            return documents

        except Exception as e:
            logger.error(f"Failed to get document list: {str(e)}")
            return []

    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a specific document"""

        try:
            metadata_collection = self.client.get_or_create_collection(
                name="document_metadata"
            )

            results = metadata_collection.get(
                ids=[doc_id],
                include=["metadatas"]
            )

            if results['ids'] and len(results['ids']) > 0:
                return results['metadatas'][0]
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get document metadata: {str(e)}")
            return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from Q&A collection (comparison removed)""" # Message changed

        try:
            # Delete Q&A chunks
            qa_results = self.qa_collection.get(
                where={"doc_id": doc_id},
                include=["ids"]
            )

            if qa_results['ids']:
                self.qa_collection.delete(ids=qa_results['ids'])
                logger.info(f"Deleted {len(qa_results['ids'])} Q&A chunks for {doc_id}")

            # Removed: Delete comparison chunks
            # Delete metadata
            try:
                metadata_collection = self.client.get_or_create_collection(
                    name="document_metadata"
                )
                metadata_collection.delete(ids=[doc_id])
            except:
                pass

            logger.info(f"Deleted document {doc_id} from collections") # Message changed
            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Q&A vector store collection""" # Message changed

        try:
            qa_count = self.qa_collection.count()
            # Removed: comp_count = self.comparison_collection.count()

            # Get document count
            try:
                metadata_collection = self.client.get_or_create_collection(
                    name="document_metadata"
                )
                doc_count = metadata_collection.count()
            except:
                doc_count = 0

            # Get sample of Q&A metadata to calculate averages
            qa_stats = {"avg_tokens_per_chunk": 0}
            if qa_count > 0:
                try:
                    qa_sample = self.qa_collection.get(
                        limit=min(1000, qa_count),
                        include=["metadatas"]
                    )

                    if qa_sample['metadatas']:
                        token_counts = [
                            meta.get("token_count", 0)
                            for meta in qa_sample['metadatas']
                        ]
                        qa_stats["avg_tokens_per_chunk"] = sum(token_counts) / len(token_counts) if token_counts else 0
                except:
                    pass

            # Removed: Get sample of comparison metadata
            stats = {
                "total_documents": doc_count,
                "qa_chunks": qa_count,
                # Removed: "comparison_chunks": comp_count,
                "total_chunks": qa_count, # Adjusted total chunks
                "database_path": self.persist_directory,
                "collections": {
                    "qa_collection": {
                        "name": "document_chunks",
                        "count": qa_count,
                        "avg_tokens_per_chunk": qa_stats["avg_tokens_per_chunk"],
                        "purpose": "Q&A and retrieval"
                    }
                    # Removed: "comparison_collection" entry
                },
                "dual_pipeline_enabled": False # Changed to False
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                "total_documents": 0,
                "qa_chunks": 0,
                # Removed: "comparison_chunks": 0,
                "total_chunks": 0,
                "database_path": self.persist_directory,
                "dual_pipeline_enabled": False # Changed to False
            }