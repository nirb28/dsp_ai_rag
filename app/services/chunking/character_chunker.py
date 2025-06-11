from typing import List, Dict, Any
import uuid

from app.models.document import DocumentChunk
from app.models.config_options import ChunkingConfig
from app.services.chunking.base import BaseChunker


class CharacterChunker(BaseChunker):
    """Split text into chunks based on character count"""
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """
        Split the text into chunks based on character count.
        
        Args:
            text: The text to split into chunks
            metadata: Metadata to attach to each chunk
            document_id: ID of the document being chunked
            
        Returns:
            List of DocumentChunk objects
        """
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        # If text is shorter than chunk size, return as a single chunk
        if len(text) <= chunk_size:
            return [
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=text,
                    metadata=metadata,
                    chunk_index=0
                )
            ]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position with overlap
            end = min(start + chunk_size, len(text))
            
            # Create a chunk from the text
            chunk_content = text[start:end]
            
            # Add the chunk to the list
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk_content,
                    metadata=metadata,
                    chunk_index=chunk_index
                )
            )
            
            # Move to the next position, accounting for overlap
            start = end - chunk_overlap if end < len(text) else end
            chunk_index += 1
        
        return chunks
    
    def get_strategy_name(self) -> str:
        """Get the name of the chunking strategy"""
        return "character"
