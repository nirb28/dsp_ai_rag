from typing import Dict, Type

from app.models.config_options import VectorStoreConfig, VectorStoreType
from app.services.vectorstores.base import BaseVectorStore
from app.services.vectorstores.faiss_store import FAISSVectorStore
from app.services.vectorstores.chroma_store import ChromaVectorStore
from app.services.vectorstores.pinecone_store import PineconeVectorStore
from app.services.vectorstores.weaviate_store import WeaviateVectorStore
from app.services.vectorstores.memory_store import InMemoryVectorStore


class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    # Registry of vector store classes
    _vector_stores: Dict[VectorStoreType, Type[BaseVectorStore]] = {
        VectorStoreType.FAISS: FAISSVectorStore,
        VectorStoreType.CHROMA: ChromaVectorStore,
        VectorStoreType.PINECONE: PineconeVectorStore,
        VectorStoreType.WEAVIATE: WeaviateVectorStore,
        VectorStoreType.IN_MEMORY: InMemoryVectorStore,
    }
    
    @classmethod
    async def create_vector_store(cls, config: VectorStoreConfig) -> BaseVectorStore:
        """
        Create a vector store instance based on the provided configuration.
        
        Args:
            config: Vector store configuration
            
        Returns:
            An instance of the appropriate vector store
            
        Raises:
            ValueError: If the requested vector store type is not supported
        """
        store_class = cls._vector_stores.get(config.store_type)
        
        if not store_class:
            raise ValueError(f"Unsupported vector store type: {config.store_type}")
        
        store = store_class(config)
        await store.initialize()
        
        return store
