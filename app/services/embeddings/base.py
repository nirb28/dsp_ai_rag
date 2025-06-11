from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from app.models.config_options import EmbeddingModelConfig


class BaseEmbedding(ABC):
    """Base class for embedding models"""
    
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self._model = None
    
    @abstractmethod
    async def initialize(self):
        """
        Initialize the embedding model.
        Should be called before using the model.
        """
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model"""
        pass
