from typing import List, Dict, Any, Optional
import os

from app.models.config_options import EmbeddingModelConfig
from app.services.embeddings.base import BaseEmbedding
from app.core.config import settings


class SentenceTransformersEmbedding(BaseEmbedding):
    """Embedding model using Sentence Transformers"""
    
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Check if using a local model path or a model name
            if self.config.local_model_path:
                model_path = os.path.join(settings.LOCAL_MODELS_DIR, self.config.local_model_path)
                self._model = SentenceTransformer(model_path, device=self.config.device)
            else:
                self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformers model: {str(e)}")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self._model:
            await self.initialize()
        
        # Process in batches to avoid memory issues
        batch_size = self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # Convert to native Python list
            embeddings = self._model.encode(batch_texts).tolist()
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if not self._model:
            raise ValueError("Model not initialized. Call initialize() first")
        return self._model.get_sentence_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model"""
        if self.config.local_model_path:
            return f"local:{self.config.local_model_path}"
        return self.config.model_name
