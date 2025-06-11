from typing import Dict, Type

from app.models.config_options import EmbeddingModelConfig, EmbeddingModelType
from app.services.embeddings.base import BaseEmbedding
from app.services.embeddings.openai_embedding import OpenAIEmbedding
from app.services.embeddings.sentence_transformers_embedding import SentenceTransformersEmbedding
from app.services.embeddings.local_embedding import LocalEmbedding


class EmbeddingFactory:
    """Factory for creating embedding model instances"""
    
    # Registry of embedding model classes
    _embedding_models: Dict[EmbeddingModelType, Type[BaseEmbedding]] = {
        EmbeddingModelType.OPENAI: OpenAIEmbedding,
        EmbeddingModelType.SENTENCE_TRANSFORMERS: SentenceTransformersEmbedding,
        EmbeddingModelType.LOCAL: LocalEmbedding,
    }
    
    @classmethod
    async def create_embedding_model(cls, config: EmbeddingModelConfig) -> BaseEmbedding:
        """
        Create an embedding model instance based on the provided configuration.
        
        Args:
            config: Embedding model configuration
            
        Returns:
            An instance of the appropriate embedding model
            
        Raises:
            ValueError: If the requested embedding model type is not supported
        """
        model_class = cls._embedding_models.get(config.model_type)
        
        if not model_class:
            raise ValueError(f"Unsupported embedding model type: {config.model_type}")
        
        model = model_class(config)
        await model.initialize()
        
        return model
