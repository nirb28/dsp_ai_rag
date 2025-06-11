from typing import List, Dict, Any, Optional
import os
import torch

from app.models.config_options import EmbeddingModelConfig
from app.services.embeddings.base import BaseEmbedding
from app.core.config import settings


class LocalEmbedding(BaseEmbedding):
    """Custom local embedding model implementation"""
    
    async def initialize(self):
        """Initialize the local embedding model"""
        try:
            # Check if model path is provided
            if not self.config.local_model_path:
                raise ValueError("Local model path is required for local embeddings")
            
            # Construct full model path from LOCAL_MODELS_DIR and the provided path
            model_path = os.path.join(settings.LOCAL_MODELS_DIR, self.config.local_model_path)
            
            # Check if model path exists
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")
            
            # Import necessary libraries
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError:
                raise ImportError("Please install transformers package with: pip install transformers")
            
            # Set device
            self.device = self.config.device
            if self.device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            
            # Set embedding dimension
            self._dimension = self.model.config.hidden_size
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local embedding model: {str(e)}")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using a local model"""
        if not hasattr(self, 'model'):
            await self.initialize()
        
        # Process in batches to avoid memory issues
        batch_size = self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize texts
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Mean Pooling - Take average of all tokens
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Convert to list and append to results
            batch_embeddings = embeddings.cpu().numpy().tolist()
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if not hasattr(self, '_dimension'):
            raise ValueError("Model not initialized. Call initialize() first")
        return self._dimension
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model"""
        return f"local:{self.config.local_model_path}"
