from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    CHARACTER = "character"
    TOKEN = "token"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class VectorStoreType(str, Enum):
    """Available vector store types"""
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    IN_MEMORY = "in_memory"


class EmbeddingModelType(str, Enum):
    """Available embedding model types"""
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    LOCAL = "local"


class CompletionModelType(str, Enum):
    """Available completion model types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL_LLAMA = "local_llama"
    LOCAL_TRANSFORMERS = "local_transformers"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    strategy: ChunkingStrategy = ChunkingStrategy.CHARACTER
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Additional parameters for specific chunking strategies
    separators: Optional[List[str]] = None
    keep_separator: bool = False
    
    # Additional parameters for recursive chunking
    parent_chunk_size: Optional[int] = None
    child_chunk_size: Optional[int] = None
    

class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    store_type: VectorStoreType = VectorStoreType.FAISS
    collection_name: str = "default"
    
    # FAISS specific parameters
    faiss_index_type: Optional[str] = None
    
    # Chroma specific parameters
    chroma_persist_directory: Optional[str] = None
    
    # Pinecone specific parameters
    pinecone_index_name: Optional[str] = None
    pinecone_namespace: Optional[str] = None
    
    # Weaviate specific parameters
    weaviate_class_name: Optional[str] = None
    
    # Generic parameters
    metadata_fields: List[str] = Field(default_factory=list)
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding model"""
    model_type: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # OpenAI specific parameters
    openai_api_key: Optional[str] = None
    
    # Cohere specific parameters
    cohere_api_key: Optional[str] = None
    
    # Local model parameters
    local_model_path: Optional[str] = None
    device: str = "cpu"  # "cpu", "cuda", "mps", etc.
    
    # Generic parameters
    batch_size: int = 32
    dimensions: Optional[int] = None
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class CompletionModelConfig(BaseModel):
    """Configuration for completion model"""
    model_type: CompletionModelType = CompletionModelType.OPENAI
    model_name: str = "gpt-3.5-turbo"
    
    # OpenAI specific parameters
    openai_api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0
    
    # Anthropic specific parameters
    anthropic_api_key: Optional[str] = None
    
    # Local model parameters
    local_model_path: Optional[str] = None
    device: str = "cpu"  # "cpu", "cuda", "mps", etc.
    max_new_tokens: int = 512
    
    # Generic parameters
    max_tokens: Optional[int] = None
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class RerankingConfig(BaseModel):
    """Configuration for reranking retrieved documents"""
    enabled: bool = False
    model_name: Optional[str] = None
    top_n: int = 5
    strategy: str = "default"
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    """Main configuration for RAG pipeline"""
    user_id: Optional[str] = None
    config_name: str = "default"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    completion: CompletionModelConfig = Field(default_factory=CompletionModelConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    additional_config: Dict[str, Any] = Field(default_factory=dict)
