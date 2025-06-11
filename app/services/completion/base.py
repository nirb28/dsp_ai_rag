from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple

from app.models.config_options import CompletionModelConfig
from app.models.retrieval import RetrievedChunk, GenerationSource


class BaseCompletion(ABC):
    """Base class for completion models"""
    
    def __init__(self, config: CompletionModelConfig):
        self.config = config
        self._model = None
    
    @abstractmethod
    async def initialize(self):
        """
        Initialize the completion model.
        Should be called before using the model.
        """
        pass
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        context: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a completion for the given prompt and context.
        
        Args:
            prompt: The user query or prompt
            context: Optional list of retrieved chunks to use as context
            kwargs: Additional parameters for the model
            
        Returns:
            Tuple of (generated_text, generation_info)
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str, 
        context: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream a completion for the given prompt and context.
        
        Args:
            prompt: The user query or prompt
            context: Optional list of retrieved chunks to use as context
            kwargs: Additional parameters for the model
            
        Returns:
            AsyncIterator yielding Tuples of (partial_text, generation_info)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the completion model"""
        pass
    
    def format_prompt(self, prompt: str, context: List[RetrievedChunk]) -> str:
        """
        Format a prompt with context for the model.
        
        Args:
            prompt: The user query or prompt
            context: Retrieved chunks to use as context
            
        Returns:
            Formatted prompt with context
        """
        # Default formatting, can be overridden by subclasses
        formatted_prompt = "Please answer the question based on the following context:\n\n"
        
        # Add context sections
        for i, chunk in enumerate(context):
            # Add source information
            source_info = f"Document ID: {chunk.document_id}"
            if chunk.page_number:
                source_info += f", Page: {chunk.page_number}"
            
            formatted_prompt += f"[{i+1}] {source_info}\n{chunk.content}\n\n"
            
        # Add the query
        formatted_prompt += f"Question: {prompt}\n\nAnswer:"
        
        return formatted_prompt
