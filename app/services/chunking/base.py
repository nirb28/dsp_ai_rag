from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from app.models.document import DocumentChunk
from app.models.config_options import ChunkingConfig


class BaseChunker(ABC):
    """Base class for document chunking strategies"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    @abstractmethod
    async def chunk_text(self, text: str, metadata: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """
        Split the text into chunks based on the chunking strategy.
        
        Args:
            text: The text to split into chunks
            metadata: Metadata to attach to each chunk
            document_id: ID of the document being chunked
            
        Returns:
            List of DocumentChunk objects
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the chunking strategy"""
        pass
