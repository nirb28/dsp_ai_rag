from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, List, Any, Optional

from app.models.config_options import RAGConfig, ChunkingStrategy, VectorStoreType, EmbeddingModelType, CompletionModelType
from app.core.security import get_current_user

router = APIRouter()

# Simulated database for configurations
# In production, replace this with a proper database
configurations_db = {}


@router.post("/", response_model=RAGConfig)
async def create_configuration(
    config: RAGConfig,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new configuration for RAG pipeline"""
    user_id = current_user.get("sub")
    
    # Set the user ID in the config
    config.user_id = user_id
    
    # Generate a unique key for this configuration
    config_key = f"{user_id}:{config.config_name}"
    
    # Check if configuration already exists
    if config_key in configurations_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Configuration with name '{config.config_name}' already exists"
        )
    
    # Store the configuration
    configurations_db[config_key] = config.dict()
    
    return config


@router.get("/", response_model=List[RAGConfig])
async def list_configurations(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all configurations for the current user"""
    user_id = current_user.get("sub")
    
    # Filter configurations for this user
    user_configs = [
        RAGConfig(**config_data)
        for key, config_data in configurations_db.items()
        if key.startswith(f"{user_id}:")
    ]
    
    return user_configs


@router.get("/{config_name}", response_model=RAGConfig)
async def get_configuration(
    config_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a specific configuration by name"""
    user_id = current_user.get("sub")
    config_key = f"{user_id}:{config_name}"
    
    if config_key not in configurations_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration '{config_name}' not found"
        )
    
    return RAGConfig(**configurations_db[config_key])


@router.put("/{config_name}", response_model=RAGConfig)
async def update_configuration(
    config_name: str,
    config: RAGConfig,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update an existing configuration"""
    user_id = current_user.get("sub")
    config_key = f"{user_id}:{config_name}"
    
    if config_key not in configurations_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration '{config_name}' not found"
        )
    
    # Update the configuration
    config.user_id = user_id
    config.config_name = config_name
    configurations_db[config_key] = config.dict()
    
    return config


@router.delete("/{config_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_configuration(
    config_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a configuration"""
    user_id = current_user.get("sub")
    config_key = f"{user_id}:{config_name}"
    
    if config_key not in configurations_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration '{config_name}' not found"
        )
    
    del configurations_db[config_key]
    return None


@router.get("/options/chunking", response_model=List[str])
async def get_chunking_strategies():
    """Get available chunking strategies"""
    return [strategy.value for strategy in ChunkingStrategy]


@router.get("/options/vectorstores", response_model=List[str])
async def get_vectorstore_types():
    """Get available vector store types"""
    return [store.value for store in VectorStoreType]


@router.get("/options/embeddings", response_model=List[str])
async def get_embedding_models():
    """Get available embedding model types"""
    return [model.value for model in EmbeddingModelType]


@router.get("/options/completions", response_model=List[str])
async def get_completion_models():
    """Get available completion model types"""
    return [model.value for model in CompletionModelType]


@router.get("/options/local-models", response_model=List[Dict[str, Any]])
async def get_available_local_models(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get available local models"""
    # Here we would scan the local models directory and return model info
    # For now, return a mock response
    return [
        {
            "id": "llama2-7b",
            "name": "Llama 2 7B",
            "type": "completion",
            "path": "models/llama/llama-2-7b",
            "requirements": {
                "gpu": True,
                "min_ram_gb": 16
            }
        },
        {
            "id": "bert-embeddings",
            "name": "BERT Embeddings",
            "type": "embedding",
            "path": "models/bert/bert-base-uncased",
            "requirements": {
                "gpu": False,
                "min_ram_gb": 4
            }
        }
    ]
