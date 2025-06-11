from typing import List, Dict, Any, Optional
import uuid
import re

from app.models.document import DocumentChunk
from app.models.config_options import ChunkingConfig
from app.services.chunking.base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Split text into chunks based on semantic meaning.
    This attempts to keep semantically related content together.
    """
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """
        Split the text into chunks based on semantic boundaries like paragraphs,
        headings, and other structural elements.
        
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
        
        # Split text by semantic boundaries (paragraphs, headings)
        # This is a simplistic approach - a more sophisticated approach would
        # use NLP to identify semantic boundaries
        paragraph_pattern = r'\n\s*\n'
        heading_pattern = r'\n#{1,6}\s+[^\n]+\n'
        
        # Combine patterns to split on either paragraphs or headings
        split_pattern = f"({paragraph_pattern}|{heading_pattern})"
        
        # Split the text into semantic elements
        elements = re.split(split_pattern, text)
        
        # Filter out empty elements
        elements = [e for e in elements if e.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        # Build chunks based on semantic elements while respecting max chunk size
        for element in elements:
            # If adding this element exceeds chunk size, create a new chunk
            if len(current_chunk) + len(element) > chunk_size and current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        content=current_chunk.strip(),
                        metadata=metadata,
                        chunk_index=chunk_index
                    )
                )
                # Include overlap from previous chunk if possible
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:] + element
                else:
                    current_chunk = element
                chunk_index += 1
            else:
                current_chunk += element
        
        # Add the final chunk if there's anything left
        if current_chunk.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=current_chunk.strip(),
                    metadata=metadata,
                    chunk_index=chunk_index
                )
            )
        
        return chunks
    
    def get_strategy_name(self) -> str:
        """Get the name of the chunking strategy"""
        return "semantic"
