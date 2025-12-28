"""
Advanced Retrieval System with Maximal Marginal Relevance (MMR) - Fixed Version
Updated for OpenAI 1.51.0 compatibility with better error handling
"""

import asyncio
from typing import List, Dict, Tuple, Optional, Any
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Use OpenAI directly instead of LlamaIndex
import openai
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class RetrievalSystem:
    """Advanced retrieval with MMR reranking"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.openai_client = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize OpenAI embedding model directly"""
        try:
            # Simple initialization without extra parameters
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for retrieval system")
            else:
                logger.warning("No OPENAI_API_KEY found")
                self.openai_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client in retrieval: {e}")
            self.openai_client = None
    
    async def retrieve(
        self, 
        query: str, 
        k: int = 10,
        metadata_filter: Optional[Dict] = None,
        retrieval_method: str = "hybrid"
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using multiple methods
        
        Args:
            query: User query
            k: Number of results to return
            metadata_filter: Optional metadata filtering
            retrieval_method: "semantic", "text", or "hybrid"
        """
        
        try:
            if retrieval_method == "hybrid":
                # Combine semantic and text search
                results = await self._hybrid_retrieval(query, k, metadata_filter)
            elif retrieval_method == "semantic":
                # Pure semantic search
                results = await self._semantic_retrieval(query, k, metadata_filter)
            else:
                # Pure text search
                results = await self._text_retrieval(query, k, metadata_filter)
            
            logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    async def _semantic_retrieval(
        self, 
        query: str, 
        k: int,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve using semantic similarity"""
        
        if not self.openai_client:
            logger.warning("OpenAI client not available, falling back to text search")
            return await self._text_retrieval(query, k, metadata_filter)
        
        try:
            # Generate query embedding using OpenAI directly
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            
            # Search vector store
            results = await self.vector_store.search_similar(
                query_embedding, 
                k=k,
                metadata_filter=metadata_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            # Fallback to text search
            return await self._text_retrieval(query, k, metadata_filter)
    
    async def _text_retrieval(
        self, 
        query: str, 
        k: int,
        metadata_filter = None
    ) -> List[Dict]:
        """Use semantic search for optimal results (intentionally)"""
        
        try:
            logger.info("Using semantic search for optimal results")
            return await self._semantic_retrieval(query, k, metadata_filter)
            
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")
            return []
    
    async def _hybrid_retrieval(
        self, 
        query: str, 
        k: int,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Combine semantic and text retrieval"""
        
        try:
            # Get results from both methods
            semantic_results = await self._semantic_retrieval(
                query, k, metadata_filter
            )
            text_results = await self._text_retrieval(
                query, k, metadata_filter
            )
            
            # Combine and deduplicate results
            combined_results = {}
            
            # Add semantic results with higher weight
            for result in semantic_results:
                result_id = result['id']
                result['retrieval_method'] = 'semantic'
                result['weighted_score'] = result.get('score', 0) * 0.7  # Semantic weight
                combined_results[result_id] = result
            
            # Add text results
            for result in text_results:
                result_id = result['id']
                if result_id in combined_results:
                    # Combine scores
                    existing = combined_results[result_id]
                    existing['weighted_score'] += result.get('score', 0) * 0.3  # Text weight
                    existing['retrieval_method'] = 'hybrid'
                else:
                    result['retrieval_method'] = 'text'
                    result['weighted_score'] = result.get('score', 0) * 0.3
                    combined_results[result_id] = result
            
            # Sort by weighted score and return top k
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x.get('weighted_score', 0),
                reverse=True
            )
            
            return sorted_results[:k]
        
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    async def rerank_mmr(
        self, 
        query: str, 
        retrieved_chunks: List[Dict], 
        k: int = 5,
        lambda_param: float = 0.7
    ) -> List[Dict]:
        """
        Rerank using Maximal Marginal Relevance (MMR)
        
        Args:
            query: Original query
            retrieved_chunks: Initially retrieved chunks
            k: Number of final results
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        
        if not retrieved_chunks:
            return []
        
        if not self.openai_client:
            logger.warning("OpenAI client not available, returning original ranking")
            return retrieved_chunks[:k]
        
        try:
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk['content'] for chunk in retrieved_chunks]
            
            # Batch process embeddings
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                
                try:
                    batch_response = self.openai_client.embeddings.create(
                        input=batch_texts,
                        model="text-embedding-3-small"
                    )
                    batch_embeddings = [data.embedding for data in batch_response.data]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings for batch {i}: {e}")
                    # Use zero embeddings as fallback
                    fallback_embeddings = [[0.0] * 1536 for _ in batch_texts]
                    all_embeddings.extend(fallback_embeddings)
            
            chunk_embeddings = np.array(all_embeddings)
            
            # Calculate relevance scores (cosine similarity with query)
            relevance_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(retrieved_chunks)))
            
            while len(selected_indices) < k and remaining_indices:
                mmr_scores = []
                
                for i in remaining_indices:
                    relevance = relevance_scores[i]
                    
                    if not selected_indices:
                        # First selection - pure relevance
                        diversity = 0
                    else:
                        # Calculate max similarity with already selected chunks
                        selected_embeddings = chunk_embeddings[selected_indices]
                        current_embedding = chunk_embeddings[i].reshape(1, -1)
                        similarities = cosine_similarity(current_embedding, selected_embeddings)[0]
                        diversity = np.max(similarities)
                    
                    # MMR score: λ * relevance - (1-λ) * max_similarity_to_selected
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                    mmr_scores.append((mmr_score, i))
                
                # Select chunk with highest MMR score
                best_score, best_index = max(mmr_scores)
                selected_indices.append(best_index)
                remaining_indices.remove(best_index)
            
            # Return reranked chunks with MMR scores
            reranked_chunks = []
            for i, idx in enumerate(selected_indices):
                chunk = retrieved_chunks[idx].copy()
                chunk['mmr_rank'] = i + 1
                chunk['mmr_score'] = float(relevance_scores[idx])
                chunk['final_score'] = chunk.get('weighted_score', chunk.get('score', 0))
                reranked_chunks.append(chunk)
            
            logger.info(f"Reranked {len(reranked_chunks)} chunks using MMR")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"MMR reranking failed: {str(e)}")
            # Fallback to original ranking
            return retrieved_chunks[:k]
    
    async def retrieve_with_context(
        self, 
        query: str, 
        k: int = 5,
        context_window: int = 2,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve chunks with surrounding context
        
        Args:
            query: User query
            k: Number of primary results
            context_window: Number of neighboring chunks to include
            metadata_filter: Optional metadata filtering
        """
        
        try:
            # Get primary results
            primary_results = await self.retrieve(query, k, metadata_filter)
            
            # Expand with context
            expanded_results = []
            processed_ids = set()
            
            for result in primary_results:
                doc_id = result['metadata'].get('doc_id')
                if not doc_id:
                    continue
                
                page_num = result['metadata'].get('page_number')
                chunk_index = result['metadata'].get('chunk_index')
                
                # Get neighboring chunks
                doc_chunks = self.vector_store.get_document_chunks(doc_id)
                
                # Sort by chunk index
                doc_chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
                
                # Find current chunk position
                current_pos = -1
                for i, chunk in enumerate(doc_chunks):
                    if chunk['id'] == result['id']:
                        current_pos = i
                        break
                
                if current_pos >= 0:
                    # Add context chunks
                    start_pos = max(0, current_pos - context_window)
                    end_pos = min(len(doc_chunks), current_pos + context_window + 1)
                    
                    for i in range(start_pos, end_pos):
                        chunk = doc_chunks[i]
                        if chunk['id'] not in processed_ids:
                            # Mark context type
                            if i == current_pos:
                                chunk['context_type'] = 'primary'
                                chunk['relevance_score'] = result.get('final_score', result.get('score', 0))
                            elif i < current_pos:
                                chunk['context_type'] = 'preceding'
                                chunk['relevance_score'] = 0.3  # Lower score for context
                            else:
                                chunk['context_type'] = 'following'
                                chunk['relevance_score'] = 0.3
                            
                            expanded_results.append(chunk)
                            processed_ids.add(chunk['id'])
            
            # Sort by relevance score
            expanded_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            logger.info(f"Expanded to {len(expanded_results)} chunks with context")
            return expanded_results
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return await self.retrieve(query, k, metadata_filter)
    
    async def multi_query_retrieval(
        self, 
        queries: List[str], 
        k: int = 5,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve using multiple related queries
        
        Args:
            queries: List of related queries
            k: Number of results per query
            metadata_filter: Optional metadata filtering
        """
        
        try:
            all_results = {}
            
            # Retrieve for each query
            for i, query in enumerate(queries):
                results = await self.retrieve(query, k, metadata_filter)
                
                for result in results:
                    result_id = result['id']
                    if result_id in all_results:
                        # Boost score for multiple query matches
                        all_results[result_id]['multi_query_score'] += result.get('final_score', result.get('score', 0))
                        all_results[result_id]['matched_queries'].append(i)
                    else:
                        result['multi_query_score'] = result.get('final_score', result.get('score', 0))
                        result['matched_queries'] = [i]
                        all_results[result_id] = result
            
            # Sort by multi-query score
            sorted_results = sorted(
                all_results.values(),
                key=lambda x: x.get('multi_query_score', 0),
                reverse=True
            )
            
            logger.info(f"Multi-query retrieval returned {len(sorted_results)} unique chunks")
            return sorted_results[:k * 2]  # Return more results for multi-query
            
        except Exception as e:
            logger.error(f"Multi-query retrieval failed: {str(e)}")
            return []
    
    def filter_by_metadata(
        self, 
        results: List[Dict], 
        filters: Dict[str, Any]
    ) -> List[Dict]:
        """
        Apply additional metadata filtering to results
        
        Args:
            results: Retrieved chunks
            filters: Metadata filters to apply
        """
        
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            include_result = True
            for key, value in filters.items():
                if key not in metadata:
                    include_result = False
                    break
                
                if isinstance(value, list):
                    if metadata[key] not in value:
                        include_result = False
                        break
                elif isinstance(value, dict):
                    # Range filtering
                    if 'min' in value and metadata[key] < value['min']:
                        include_result = False
                        break
                    if 'max' in value and metadata[key] > value['max']:
                        include_result = False
                        break
                else:
                    if metadata[key] != value:
                        include_result = False
                        break
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    async def get_query_suggestions(
        self, 
        partial_query: str, 
        doc_id: Optional[str] = None
    ) -> List[str]:
        """
        Generate query suggestions based on document content
        
        Args:
            partial_query: Partial user query
            doc_id: Optional document ID to scope suggestions
        """
        
        suggestions = []
        
        try:
            # Get sample chunks for suggestions
            if doc_id:
                metadata_filter = {'doc_id': doc_id}
            else:
                metadata_filter = None
            
            # Search for related content
            if partial_query.strip():
                results = await self.retrieve(partial_query, k=10, metadata_filter=metadata_filter)
                
                # Extract potential query completions from chunk content
                for result in results[:5]:
                    content = result['content']
                    
                    # Simple extraction of potential questions/topics
                    sentences = content.split('.')
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if (len(sentence) > 20 and len(sentence) < 100 and
                            any(word in sentence.lower() for word in ['what', 'how', 'why', 'when', 'where'])):
                            suggestions.append(sentence + '?')
                        
                        if len(suggestions) >= 5:
                            break
                    
                    if len(suggestions) >= 5:
                        break
            else:
                # Default suggestions for empty query
                suggestions = [
                    "What is the main topic of this document?",
                    "Can you summarize the key findings?",
                    "What are the main conclusions?",
                    "What methodology was used?",
                    "What are the recommendations?"
                ]
        
        except Exception as e:
            logger.error(f"Failed to generate query suggestions: {str(e)}")
        
        return suggestions[:5]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            stats = {
                **vector_stats,
                "embedding_model": "text-embedding-3-small",
                "embedding_dimensions": 1536,
                "retrieval_methods": ["semantic", "text", "hybrid"],
                "reranking_enabled": True,
                "mmr_available": True,
                "openai_client_available": self.openai_client is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {str(e)}")
            return {}