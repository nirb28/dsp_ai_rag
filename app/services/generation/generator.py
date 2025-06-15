from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
import asyncio
import logging
from datetime import datetime

from app.models.config_options import RAGConfig
from app.models.retrieval import (
    GenerationRequest, 
    GenerationResponse, 
    RetrievalRequest,
    RetrievedChunk,
    StreamGenerationResponse,
    GenerationSource
)
from app.services.retrieval.retriever import RetrievalService
from app.services.completion.factory import CompletionFactory


class GenerationService:
    """
    Service for generation operations in the RAG pipeline.
    Integrates retrieval and completion services.
    """
    
    def __init__(self):
        self.retrieval_service = RetrievalService()
    
    async def generate(
        self, 
        request: GenerationRequest, 
        config: RAGConfig,
        user_id: str
    ) -> GenerationResponse:
        """
        Generate a response for the given request using RAG.
        
        Args:
            request: Generation request parameters
            config: RAG configuration to use
            user_id: ID of the requesting user
            
        Returns:
            GenerationResponse with results
        """
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Retrieve relevant chunks if retrieval is enabled
            retrieved_chunks = []
            if request.enable_retrieval:
                # Create retrieval request
                retrieval_request = RetrievalRequest(
                    query=request.query,
                    top_k=request.retrieval_top_k,
                    filters=request.filters,
                    document_id=request.document_id
                )
                
                # Perform retrieval
                retrieval_response = await self.retrieval_service.retrieve(
                    retrieval_request,
                    config,
                    user_id
                )
                
                if retrieval_response.error:
                    return GenerationResponse(
                        query=request.query,
                        response="",
                        error=f"Retrieval error: {retrieval_response.error}",
                        retrieved_chunks=[],
                        generation_time=0.0
                    )
                
                # Get retrieved chunks
                retrieved_chunks = retrieval_response.results
            
            # Step 2: Create completion model
            completion_model = await CompletionFactory.create_completion_model(
                config.completion_model_config
            )
            
            # Step 3: Generate completion
            completion_text, generation_info = await completion_model.generate(
                prompt=request.query,
                context=retrieved_chunks if request.enable_retrieval else None,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            )
            
            # Calculate generation time
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Return response
            return GenerationResponse(
                query=request.query,
                response=completion_text,
                retrieved_chunks=retrieved_chunks if request.include_sources else [],
                generation_info=generation_info,
                generation_time=generation_time
            )
            
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return GenerationResponse(
                query=request.query,
                response="",
                error=f"Generation error: {str(e)}",
                retrieved_chunks=[],
                generation_time=0.0
            )
    
    async def generate_stream(
        self, 
        request: GenerationRequest, 
        config: RAGConfig,
        user_id: str
    ) -> AsyncIterator[StreamGenerationResponse]:
        """
        Stream a response for the given request using RAG.
        
        Args:
            request: Generation request parameters
            config: RAG configuration to use
            user_id: ID of the requesting user
            
        Returns:
            AsyncIterator yielding StreamGenerationResponse objects
        """
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Retrieve relevant chunks if retrieval is enabled
            retrieved_chunks = []
            if request.enable_retrieval:
                # Create retrieval request
                retrieval_request = RetrievalRequest(
                    query=request.query,
                    top_k=request.retrieval_top_k,
                    filters=request.filters,
                    document_id=request.document_id
                )
                
                # Perform retrieval
                retrieval_response = await self.retrieval_service.retrieve(
                    retrieval_request,
                    config,
                    user_id
                )
                
                if retrieval_response.error:
                    yield StreamGenerationResponse(
                        chunk="",
                        error=f"Retrieval error: {retrieval_response.error}",
                        is_done=True,
                        sources=[]
                    )
                    return
                
                # Get retrieved chunks
                retrieved_chunks = retrieval_response.results
                
                # Emit sources if requested
                if request.include_sources and retrieved_chunks:
                    yield StreamGenerationResponse(
                        chunk="",
                        sources=retrieved_chunks,
                        is_done=False
                    )
            
            # Step 2: Create completion model
            completion_model = await CompletionFactory.create_completion_model(
                config.completion_model_config
            )
            
            # Step 3: Generate streaming completion
            async for token, generation_info in completion_model.generate_stream(
                prompt=request.query,
                context=retrieved_chunks if request.enable_retrieval else None,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p
            ):
                # Yield each token
                is_done = generation_info.get("finish_reason") is not None
                
                yield StreamGenerationResponse(
                    chunk=token,
                    is_done=is_done,
                    sources=[]  # Sources are only sent in the first chunk
                )
                
                # If done, break
                if is_done:
                    break
            
            # Ensure we send a final message with is_done=True
            yield StreamGenerationResponse(
                chunk="",
                is_done=True,
                sources=[]
            )
            
        except Exception as e:
            logging.error(f"Streaming generation error: {str(e)}")
            yield StreamGenerationResponse(
                chunk="",
                error=f"Streaming generation error: {str(e)}",
                is_done=True,
                sources=[]
            )
