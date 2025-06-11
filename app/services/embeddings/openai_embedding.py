from typing import List, Dict, Any, Optional
import os
import httpx
import asyncio

from app.models.config_options import EmbeddingModelConfig
from app.services.embeddings.base import BaseEmbedding
from app.core.config import settings


class OpenAIEmbedding(BaseEmbedding):
    """Embedding model using OpenAI's embedding API"""
    
    async def initialize(self):
        """Initialize the embedding model"""
        # Set up API key
        self.api_key = self.config.openai_api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        
        # Default to text-embedding-ada-002 if not specified
        self.model_name = self.config.model_name or "text-embedding-ada-002"
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }.get(self.model_name, 1536)  # Default to 1536 if model unknown
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not hasattr(self, "api_key"):
            await self.initialize()
        
        # Process in batches to avoid API limits
        batch_size = min(self.config.batch_size, 20)  # OpenAI typically limits batch size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = await self._get_batch_embeddings(batch_texts)
            all_embeddings.extend(embeddings)
            
            # Small delay to avoid rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    async def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts from OpenAI API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "input": texts,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=data,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text}")
                
                response_data = response.json()
                # Extract embeddings in the same order as input texts
                sorted_data = sorted(response_data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
                
        except Exception as e:
            raise RuntimeError(f"Failed to get OpenAI embeddings: {str(e)}")
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if not hasattr(self, "_dimensions"):
            raise ValueError("Model not initialized. Call initialize() first")
        return self._dimensions
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model"""
        if hasattr(self, "model_name"):
            return f"openai:{self.model_name}"
        return "openai:text-embedding-ada-002"
