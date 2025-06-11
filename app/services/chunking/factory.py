from typing import Dict, Type

from app.models.config_options import ChunkingConfig, ChunkingStrategy
from app.services.chunking.base import BaseChunker
from app.services.chunking.character_chunker import CharacterChunker
from app.services.chunking.token_chunker import TokenChunker
from app.services.chunking.semantic_chunker import SemanticChunker
from app.services.chunking.recursive_chunker import RecursiveChunker
from app.services.chunking.sentence_chunker import SentenceChunker
from app.services.chunking.paragraph_chunker import ParagraphChunker


class ChunkerFactory:
    """Factory for creating chunker instances"""
    
    # Registry of chunker classes
    _chunkers: Dict[ChunkingStrategy, Type[BaseChunker]] = {
        ChunkingStrategy.CHARACTER: CharacterChunker,
        ChunkingStrategy.TOKEN: TokenChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
    }
    
    @classmethod
    def create_chunker(cls, config: ChunkingConfig) -> BaseChunker:
        """
        Create a chunker instance based on the provided configuration.
        
        Args:
            config: Chunking configuration
            
        Returns:
            An instance of the appropriate chunker
            
        Raises:
            ValueError: If the requested chunking strategy is not supported
        """
        chunker_class = cls._chunkers.get(config.strategy)
        
        if not chunker_class:
            raise ValueError(f"Unsupported chunking strategy: {config.strategy}")
        
        return chunker_class(config)
