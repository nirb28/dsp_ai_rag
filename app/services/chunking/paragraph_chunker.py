from typing import List, Dict, Any
import uuid
import re

from app.models.document import DocumentChunk
from app.models.config_options import ChunkingConfig
from app.services.chunking.base import BaseChunker


class ParagraphChunker(BaseChunker):
    """Split text into chunks based on paragraphs"""
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """
        Split the text into chunks based on paragraphs.
        
        Args:
            text: The text to split into chunks
            metadata: Metadata to attach to each chunk
            document_id: ID of the document being chunked
            
        Returns:
            List of DocumentChunk objects
        """
        # Split text by paragraphs (consecutive newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the chunk size and we already have content,
            # create a new chunk and start a new one
            if len(current_chunk) + len(paragraph) > self.config.chunk_size and current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        content=current_chunk.strip(),
                        metadata=metadata,
                        chunk_index=chunk_index
                    )
                )
                current_chunk = paragraph + "\n\n"
                chunk_index += 1
            else:
                current_chunk += paragraph + "\n\n"
        
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
        return "paragraph"
