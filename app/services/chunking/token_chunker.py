from typing import List, Dict, Any, Optional
import uuid
import tiktoken

from app.models.document import DocumentChunk
from app.models.config_options import ChunkingConfig
from app.services.chunking.base import BaseChunker


class TokenChunker(BaseChunker):
    """Split text into chunks based on token count"""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """
        Split the text into chunks based on token count.
        
        Args:
            text: The text to split into chunks
            metadata: Metadata to attach to each chunk
            document_id: ID of the document being chunked
            
        Returns:
            List of DocumentChunk objects
        """
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        
        # If tokens are fewer than chunk size, return as a single chunk
        if len(tokens) <= chunk_size:
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
        
        while start < len(tokens):
            # Calculate end position with respect to token count
            end = min(start + chunk_size, len(tokens))
            
            # Decode tokens back to text
            chunk_tokens = tokens[start:end]
            chunk_content = self.encoding.decode(chunk_tokens)
            
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
            start = end - chunk_overlap if end < len(tokens) else end
            chunk_index += 1
        
        return chunks
    
    def get_strategy_name(self) -> str:
        """Get the name of the chunking strategy"""
        return "token"
