from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio

from app.models.document import DocumentChunk
from app.models.config_options import RAGConfig, RerankingConfig
from app.models.retrieval import (
    RetrievalRequest, 
    RetrievalResponse, 
    RetrievedChunk,
    MetadataFilter
)
from app.services.embeddings.factory import EmbeddingFactory
from app.services.vectorstores.factory import VectorStoreFactory


class RetrievalService:
    """
    Service for retrieval operations in the RAG pipeline.
    Handles query embedding, vector search, and optional reranking.
    """
    
    async def retrieve(
        self, 
        request: RetrievalRequest, 
        config: RAGConfig,
        user_id: str
    ) -> RetrievalResponse:
        """
        Perform a retrieval operation based on the request and configuration.
        
        Args:
            request: Retrieval request parameters
            config: RAG configuration to use
            user_id: ID of the requesting user
            
        Returns:
            RetrievalResponse with results
        """
        try:
            # Get embedding model
            embedding_model = await EmbeddingFactory.create_embedding_model(config.embedding_model_config)
            
            # Generate embedding for query
            query_embeddings = await embedding_model.get_embeddings([request.query])
            if not query_embeddings or len(query_embeddings) == 0:
                return RetrievalResponse(
                    query=request.query,
                    results=[],
                    error="Failed to generate query embedding"
                )
            
            query_embedding = query_embeddings[0]
            
            # Get vector store
            vector_store = await VectorStoreFactory.create_vector_store(config.vector_store_config)
            
            # Prepare filters
            filters = request.filters
            
            # Add document filter if specified
            if request.document_id:
                # Create a filter for the specified document
                doc_filter = MetadataFilter(
                    field="document_id",
                    operator="eq",
                    value=request.document_id
                )
                
                if filters:
                    filters.append(doc_filter)
                else:
                    filters = [doc_filter]
            
            # Search vector store
            search_results = await vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k or 5,
                filters=filters
            )
            
            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            for chunk, score in search_results:
                retrieved_chunk = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=score,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index
                )
                retrieved_chunks.append(retrieved_chunk)
            
            # Apply reranking if configured
            if config.reranking_config and config.reranking_config.enabled and retrieved_chunks:
                retrieved_chunks = await self._rerank_results(
                    query=request.query,
                    chunks=retrieved_chunks,
                    reranking_config=config.reranking_config
                )
            
            # Return response
            return RetrievalResponse(
                query=request.query,
                results=retrieved_chunks
            )
            
        except Exception as e:
            return RetrievalResponse(
                query=request.query,
                results=[],
                error=f"Retrieval error: {str(e)}"
            )
    
    async def _rerank_results(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        reranking_config: RerankingConfig
    ) -> List[RetrievedChunk]:
        """
        Rerank retrieved chunks using specified reranking method.
        
        Args:
            query: Original query string
            chunks: Retrieved chunks to rerank
            reranking_config: Reranking configuration
            
        Returns:
            List of reranked chunks
        """
        if not chunks:
            return chunks
        
        try:
            # Determine which reranking method to use
            if reranking_config.method == "cross_encoder":
                return await self._rerank_cross_encoder(query, chunks, reranking_config)
            elif reranking_config.method == "bm25":
                return await self._rerank_bm25(query, chunks, reranking_config)
            elif reranking_config.method == "mmr":
                return await self._rerank_mmr(query, chunks, reranking_config)
            else:
                # Default to returning original chunks
                return chunks
                
        except Exception as e:
            print(f"Reranking error: {str(e)}")
            # On error, return original chunks
            return chunks
    
    async def _rerank_cross_encoder(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        config: RerankingConfig
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks using a cross-encoder model.
        
        Args:
            query: Original query string
            chunks: Retrieved chunks to rerank
            config: Reranking configuration
            
        Returns:
            List of reranked chunks
        """
        try:
            from sentence_transformers import CrossEncoder
            
            # Load cross-encoder model
            model_name = config.cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            model = CrossEncoder(model_name, device=config.device)
            
            # Create input pairs
            pairs = [[query, chunk.content] for chunk in chunks]
            
            # Compute scores
            scores = model.predict(pairs)
            
            # Update scores and sort
            for i, chunk in enumerate(chunks):
                # Replace original score with cross-encoder score
                chunk.score = float(scores[i])
            
            # Sort by new score (descending)
            chunks.sort(key=lambda x: x.score, reverse=True)
            
            return chunks
            
        except ImportError:
            print("sentence-transformers not installed. Skipping cross-encoder reranking.")
            return chunks
        except Exception as e:
            print(f"Cross-encoder reranking error: {str(e)}")
            return chunks
    
    async def _rerank_bm25(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        config: RerankingConfig
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks using BM25 algorithm.
        
        Args:
            query: Original query string
            chunks: Retrieved chunks to rerank
            config: Reranking configuration
            
        Returns:
            List of reranked chunks
        """
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            
            # Try to download nltk data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Tokenize chunks
            tokenized_chunks = [nltk.word_tokenize(chunk.content.lower()) for chunk in chunks]
            
            # Create BM25 model
            bm25 = BM25Okapi(tokenized_chunks)
            
            # Tokenize query
            tokenized_query = nltk.word_tokenize(query.lower())
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Apply weights to combine original scores with BM25 scores
            bm25_weight = config.bm25_weight or 0.5
            vector_weight = 1.0 - bm25_weight
            
            for i, chunk in enumerate(chunks):
                # Combine scores
                combined_score = (vector_weight * chunk.score) + (bm25_weight * bm25_scores[i])
                chunk.score = float(combined_score)
            
            # Sort by new score (descending)
            chunks.sort(key=lambda x: x.score, reverse=True)
            
            return chunks
            
        except ImportError:
            print("rank_bm25 not installed. Skipping BM25 reranking.")
            return chunks
        except Exception as e:
            print(f"BM25 reranking error: {str(e)}")
            return chunks
    
    async def _rerank_mmr(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        config: RerankingConfig
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks using Maximum Marginal Relevance (MMR) to optimize for relevance and diversity.
        
        Args:
            query: Original query string
            chunks: Retrieved chunks to rerank
            config: Reranking configuration
            
        Returns:
            List of reranked chunks
        """
        try:
            import numpy as np
            
            # Check if we have embeddings for all chunks
            all_have_embeddings = all(hasattr(chunk, 'embedding') and chunk.embedding is not None for chunk in chunks)
            
            # If not, we'll need to generate embeddings
            if not all_have_embeddings:
                # Get embedding model config from main config
                embedding_model_config = config.embedding_model_config
                
                # If none provided, create a default sentence transformer model
                if not embedding_model_config:
                    from app.models.config_options import EmbeddingModelConfig, EmbeddingModelType
                    embedding_model_config = EmbeddingModelConfig(
                        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
                        model_name="all-MiniLM-L6-v2"
                    )
                
                # Create embedding model
                embedding_model = await EmbeddingFactory.create_embedding_model(embedding_model_config)
                
                # Get embeddings for all chunk contents
                contents = [chunk.content for chunk in chunks]
                embeddings = await embedding_model.get_embeddings(contents)
                
                # Assign embeddings to chunks
                for i, embedding in enumerate(embeddings):
                    if i < len(chunks):
                        chunks[i].embedding = embedding
            
            # MMR parameters
            lambda_param = config.mmr_lambda or 0.5  # Balance between relevance and diversity
            
            # Convert to numpy arrays for faster computation
            doc_embeddings = np.array([chunk.embedding for chunk in chunks])
            
            # Original retrieval scores
            retrieval_scores = np.array([chunk.score for chunk in chunks])
            
            # Sort indices by original score
            sorted_indices = np.argsort(-retrieval_scores)
            
            # Start with highest-scoring document
            selected = [sorted_indices[0]]
            remaining = sorted_indices[1:].tolist()
            
            # Select documents using MMR
            while remaining and len(selected) < len(chunks):
                best_score = -float('inf')
                best_idx = -1
                
                # Calculate MMR score for each remaining document
                for i, idx in enumerate(remaining):
                    # Relevance score (using original vector score)
                    relevance = retrieval_scores[idx]
                    
                    # Diversity score (max similarity to any selected document)
                    similarities = [
                        np.dot(doc_embeddings[idx], doc_embeddings[sel]) / 
                        (np.linalg.norm(doc_embeddings[idx]) * np.linalg.norm(doc_embeddings[sel]))
                        for sel in selected
                    ]
                    diversity = max(similarities) if similarities else 0
                    
                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                # Add the best document to selected
                if best_idx >= 0:
                    selected.append(remaining[best_idx])
                    remaining.pop(best_idx)
                else:
                    break
            
            # Re-order chunks based on MMR selection
            mmr_reranked = [chunks[i] for i in selected]
            
            return mmr_reranked
            
        except Exception as e:
            print(f"MMR reranking error: {str(e)}")
            return chunks
