from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
import os
from pathlib import Path


class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "DSP AI RAG API"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # JWT Auth settings
    JWT_AUTH_URL: Optional[str] = os.getenv("JWT_AUTH_URL", "http://localhost:8001/auth/validate")
    
    # Storage settings
    UPLOAD_DIR: Path = Path("./storage/uploads")
    VECTOR_DB_DIR: Path = Path("./storage/vectordb")
    DOCUMENT_STORAGE_DIR: Path = Path("./storage/documents")
    
    # Default model settings
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_COMPLETION_MODEL: str = "gpt-3.5-turbo"
    
    # Model provider settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    
    # Vector DB settings
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    WEAVIATE_URL: Optional[str] = os.getenv("WEAVIATE_URL")
    
    # Local models settings
    LOCAL_MODELS_DIR: Path = Path("./storage/models")
    ENABLE_LOCAL_MODELS: bool = True
    
    # Processing settings
    MAX_DOCUMENT_SIZE_MB: int = 50
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure necessary directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)
