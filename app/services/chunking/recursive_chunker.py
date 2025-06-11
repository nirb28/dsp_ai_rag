from typing import List, Dict, Any
import uuid
import re

from app.models.document import DocumentChunk
from app.models.config_options import ChunkingConfig
from app.services.chunking.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Split text into chunks using a recursive hierarchy.
    Creates parent chunks (larger sections) and child chunks (smaller units).
    """
    
    async def chunk_text(self, text: str, metadata: Dict[str, Any], document_id: str) -> List[DocumentChunk]:
        """
        Split the text into a hierarchy of chunks with parent and child relationships.
        
        Args:
            text: The text to split into chunks
            metadata: Metadata to attach to each chunk
            document_id: ID of the document being chunked
            
        Returns:
            List of DocumentChunk objects
        """
        # Get config parameters with fallbacks
        parent_chunk_size = self.config.parent_chunk_size or self.config.chunk_size * 3
        child_chunk_size = self.config.child_chunk_size or self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        # If text is shorter than parent chunk size, return as a single chunk
        if len(text) <= parent_chunk_size:
            return [
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=text,
                    metadata={**metadata, "level": "single"},
                    chunk_index=0
                )
            ]
        
        # First level: Split by major sections (headings, chapters, etc.)
        section_pattern = r'(?<=\n)#{1,3}\s+[^\n]+\n'  # Match level 1-3 headings
        sections = re.split(section_pattern, text)
        
        # If no major sections found or just one, use paragraph splitting
        if len(sections) <= 2:  # Account for potential empty first element
            paragraph_pattern = r'\n\s*\n'
            sections = re.split(paragraph_pattern, text)
        
        # Filter out empty sections
        sections = [s for s in sections if s.strip()]
        
        all_chunks = []
        parent_index = 0
        child_overall_index = 0
        
        for section_idx, section in enumerate(sections):
            # If section exceeds parent chunk size, split further
            if len(section) > parent_chunk_size:
                # Split this section into parent chunks
                start = 0
                while start < len(section):
                    # Calculate end position with overlap
                    end = min(start + parent_chunk_size, len(section))
                    
                    # Create parent chunk
                    parent_content = section[start:end]
                    parent_id = str(uuid.uuid4())
                    
                    parent_chunk = DocumentChunk(
                        chunk_id=parent_id,
                        document_id=document_id,
                        content=parent_content,
                        metadata={**metadata, "level": "parent", "section": section_idx},
                        chunk_index=parent_index
                    )
                    all_chunks.append(parent_chunk)
                    
                    # Now create child chunks from this parent
                    child_start = 0
                    child_index = 0
                    
                    while child_start < len(parent_content):
                        # Calculate child end position with overlap
                        child_end = min(child_start + child_chunk_size, len(parent_content))
                        
                        # Create child chunk
                        child_content = parent_content[child_start:child_end]
                        child_chunk = DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            content=child_content,
                            metadata={
                                **metadata, 
                                "level": "child", 
                                "parent_id": parent_id,
                                "section": section_idx,
                                "parent_index": parent_index
                            },
                            chunk_index=child_overall_index
                        )
                        all_chunks.append(child_chunk)
                        
                        # Move to next child position with overlap
                        child_start = child_end - chunk_overlap if child_end < len(parent_content) else child_end
                        child_index += 1
                        child_overall_index += 1
                    
                    # Move to next parent position with overlap
                    start = end - chunk_overlap if end < len(section) else end
                    parent_index += 1
            else:
                # Section fits in a single parent chunk
                parent_id = str(uuid.uuid4())
                parent_chunk = DocumentChunk(
                    chunk_id=parent_id,
                    document_id=document_id,
                    content=section,
                    metadata={**metadata, "level": "parent", "section": section_idx},
                    chunk_index=parent_index
                )
                all_chunks.append(parent_chunk)
                
                # Create child chunks from this parent
                child_start = 0
                child_index = 0
                
                while child_start < len(section):
                    # Calculate child end position with overlap
                    child_end = min(child_start + child_chunk_size, len(section))
                    
                    # Create child chunk
                    child_content = section[child_start:child_end]
                    child_chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        content=child_content,
                        metadata={
                            **metadata, 
                            "level": "child", 
                            "parent_id": parent_id,
                            "section": section_idx,
                            "parent_index": parent_index
                        },
                        chunk_index=child_overall_index
                    )
                    all_chunks.append(child_chunk)
                    
                    # Move to next child position with overlap
                    child_start = child_end - chunk_overlap if child_end < len(section) else child_end
                    child_index += 1
                    child_overall_index += 1
                
                parent_index += 1
        
        return all_chunks
    
    def get_strategy_name(self) -> str:
        """Get the name of the chunking strategy"""
        return "recursive"
