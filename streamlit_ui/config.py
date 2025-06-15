import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import the models from the main app if possible, otherwise define simpler versions
try:
    from app.models.config_options import (
        RAGConfig, 
        CompletionModelConfig,
        EmbeddingModelConfig,
        VectorStoreConfig,
        ChunkingConfig,
        RerankingConfig,
        CompletionModelType,
        EmbeddingModelType,
        VectorStoreType,
        ChunkingStrategy
    )
except ImportError:
    # Define simplified versions for standalone operation
    from enum import Enum
    from pydantic import BaseModel, Field
    
    class ChunkingStrategy(str, Enum):
        CHARACTER = "character"
        TOKEN = "token"
        SENTENCE = "sentence"
        PARAGRAPH = "paragraph"
        SEMANTIC = "semantic"
        RECURSIVE = "recursive"
    
    class VectorStoreType(str, Enum):
        FAISS = "faiss"
        CHROMA = "chroma"
        PINECONE = "pinecone"
        WEAVIATE = "weaviate"
        IN_MEMORY = "in_memory"
    
    class EmbeddingModelType(str, Enum):
        OPENAI = "openai"
        SENTENCE_TRANSFORMERS = "sentence_transformers"
        LOCAL = "local"
    
    class CompletionModelType(str, Enum):
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        LOCAL_LLAMA = "local_llama"
    
    class ChunkingConfig(BaseModel):
        strategy: ChunkingStrategy = ChunkingStrategy.CHARACTER
        chunk_size: int = 1000
        chunk_overlap: int = 200
    
    class VectorStoreConfig(BaseModel):
        store_type: VectorStoreType = VectorStoreType.FAISS
        collection_name: str = "default"
    
    class EmbeddingModelConfig(BaseModel):
        model_type: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    class CompletionModelConfig(BaseModel):
        model_type: CompletionModelType = CompletionModelType.OPENAI
        model_name: str = "gpt-3.5-turbo"
        temperature: float = 0.7
        top_p: float = 1.0
    
    class RerankingConfig(BaseModel):
        enabled: bool = False
    
    class RAGConfig(BaseModel):
        config_name: str = "default"
        chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
        vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
        embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
        completion: CompletionModelConfig = Field(default_factory=CompletionModelConfig)
        reranking: RerankingConfig = Field(default_factory=RerankingConfig)

# Configuration file paths
DEFAULT_CONFIG_PATH = Path("streamlit_ui/config/default_config.json")
USER_CONFIG_PATH = Path("streamlit_ui/config/user_config.json")

# Create config directory if it doesn't exist
os.makedirs(DEFAULT_CONFIG_PATH.parent, exist_ok=True)

def get_default_rag_config() -> RAGConfig:
    """Return a default RAG configuration"""
    return RAGConfig(
        config_name="streamlit-ui-default",
        completion=CompletionModelConfig(
            model_type=CompletionModelType.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        ),
        embedding=EmbeddingModelConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        ),
        vectorstore=VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            collection_name="default",
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
        ),
        reranking=RerankingConfig(
            enabled=False,
        )
    )

def save_default_config():
    """Save the default configuration to a file"""
    config = get_default_rag_config()
    
    try:
        with open(DEFAULT_CONFIG_PATH, 'w') as f:
            # Convert to dict and save as JSON
            if hasattr(config, "model_dump"):  # Pydantic v2
                json.dump(config.model_dump(), f, indent=2)
            else:  # Pydantic v1
                json.dump(config.dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving default config: {e}")
        return False

def load_config(config_path: Optional[Path] = None) -> RAGConfig:
    """
    Load configuration from a file. If file doesn't exist,
    create with default settings.
    """
    path = config_path or USER_CONFIG_PATH
    
    # If user config doesn't exist, try to load default config
    if not path.exists() and path == USER_CONFIG_PATH:
        if not DEFAULT_CONFIG_PATH.exists():
            save_default_config()
        
        if DEFAULT_CONFIG_PATH.exists():
            path = DEFAULT_CONFIG_PATH
    
    # If config file exists, load it
    if path.exists():
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert dict to RAGConfig
            return RAGConfig(**config_dict)
        except Exception as e:
            print(f"Error loading config from {path}: {e}")
    
    # Return default config if loading failed
    return get_default_rag_config()

def save_config(config: RAGConfig, config_path: Optional[Path] = None) -> bool:
    """Save configuration to a file"""
    path = config_path or USER_CONFIG_PATH
    
    try:
        with open(path, 'w') as f:
            # Convert to dict and save as JSON
            if hasattr(config, "model_dump"):  # Pydantic v2
                json.dump(config.model_dump(), f, indent=2)
            else:  # Pydantic v1
                json.dump(config.dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config to {path}: {e}")
        return False
