from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from app.models.document import DocumentChunk
from app.models.config_options import VectorStoreConfig
from app.models.retrieval import MetadataFilter


class BaseVectorStore(ABC):
    """Base class for vector stores"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
    
    @abstractmethod
    async def initialize(self):
        """
        Initialize the vector store.
        Should be called before using the vector store.
        """
        pass
    
    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: Optional[List[MetadataFilter]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for document chunks similar to the query embedding.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of (document_chunk, score) tuples
        """
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of unique documents
        """
        pass
    
    @abstractmethod
    async def get_chunk_count(self) -> int:
        """
        Get the number of chunks in the vector store.
        
        Returns:
            Number of chunks
        """
        pass
