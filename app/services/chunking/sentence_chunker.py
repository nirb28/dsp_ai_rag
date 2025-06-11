from typing import List, Dict, Any
import uuid
import re

from app.models.document import DocumentChunk
from app.models.config_options import ChunkingConfig
from app.services.chunking.base import BaseChunker


class SentenceChunker(BaseChunker):
    """Split text into chunks based on sentences"""
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """
        Split the text into chunks based on sentences.
        
        Args:
            text: The text to split into chunks
            metadata: Metadata to attach to each chunk
            document_id: ID of the document being chunked
            
        Returns:
            List of DocumentChunk objects
        """
        # Pattern for sentence splitting - handles common punctuation and preserves the delimiter
        sentence_pattern = r'(?<=[.!?])\s+'
        
        # Split text by sentence pattern
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size and we already have content,
            # create a new chunk and start a new one
            if len(current_chunk) + len(sentence) > self.config.chunk_size and current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        content=current_chunk.strip(),
                        metadata=metadata,
                        chunk_index=chunk_index
                    )
                )
                
                # Handle overlap if specified
                if self.config.chunk_overlap > 0 and len(current_chunk) > self.config.chunk_overlap:
                    # Get the last few sentences for overlap
                    overlap_sentences = current_chunk.split('. ')
                    overlap_text = '. '.join(overlap_sentences[-3:]) if len(overlap_sentences) > 3 else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                    
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the final chunk if there's content
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
        return "sentence"
