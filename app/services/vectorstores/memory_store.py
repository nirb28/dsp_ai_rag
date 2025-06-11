from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
import json

from app.models.document import DocumentChunk
from app.models.config_options import VectorStoreConfig
from app.models.retrieval import MetadataFilter
from app.services.vectorstores.base import BaseVectorStore
from app.core.config import settings


class InMemoryVectorStore(BaseVectorStore):
    """
    Simple in-memory vector store implementation.
    Useful for testing and small applications.
    """
    
    async def initialize(self):
        """Initialize the in-memory vector store"""
        try:
            self.collection_name = self.config.collection_name
            
            # Create a persist directory for saving data
            self.persist_dir = os.path.join(
                settings.VECTOR_DB_DIR,
                "memory",
                self.collection_name
            )
            os.makedirs(self.persist_dir, exist_ok=True)
            
            self.data_path = os.path.join(self.persist_dir, "vectors.json")
            
            # Initialize data structures
            self.chunks: Dict[str, DocumentChunk] = {}
            self.embeddings: Dict[str, List[float]] = {}
            self.document_ids: Dict[str, List[str]] = {}  # document_id -> list of chunk_ids
            
            # Load data if exists
            await self._load_data()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize in-memory vector store: {str(e)}")
    
    async def _save_data(self):
        """Save data to disk"""
        try:
            # Convert chunks to serializable format
            serializable_chunks = {}
            for chunk_id, chunk in self.chunks.items():
                serializable_chunks[chunk_id] = chunk.dict()
            
            # Prepare data for saving
            data = {
                "chunks": serializable_chunks,
                "embeddings": self.embeddings,
                "document_ids": self.document_ids
            }
            
            with open(self.data_path, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            print(f"Warning: Failed to save in-memory vector store: {str(e)}")
    
    async def _load_data(self):
        """Load data from disk if available"""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                
                # Load chunks
                if "chunks" in data:
                    for chunk_id, chunk_data in data["chunks"].items():
                        self.chunks[chunk_id] = DocumentChunk(**chunk_data)
                
                # Load embeddings
                if "embeddings" in data:
                    self.embeddings = data["embeddings"]
                
                # Load document IDs
                if "document_ids" in data:
                    self.document_ids = data["document_ids"]
                    
        except Exception as e:
            print(f"Warning: Failed to load in-memory vector store: {str(e)}")
            # Start with empty data structures
            self.chunks = {}
            self.embeddings = {}
            self.document_ids = {}
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                return True
            
            for chunk in chunks:
                chunk_id = chunk.chunk_id
                document_id = chunk.document_id
                
                # Store the chunk
                self.chunks[chunk_id] = chunk
                
                # Store the embedding
                if chunk.embedding:
                    self.embeddings[chunk_id] = chunk.embedding
                
                # Update document_id to chunk_id mapping
                if document_id not in self.document_ids:
                    self.document_ids[document_id] = []
                
                self.document_ids[document_id].append(chunk_id)
            
            # Save data to disk
            await self._save_data()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to in-memory store: {str(e)}")
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: Optional[List[MetadataFilter]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for document chunks similar to the query embedding"""
        try:
            if not self.embeddings:
                return []
            
            # Convert query embedding to numpy array
            query_np = np.array(query_embedding)
            
            # Calculate similarities for all embeddings
            similarities = []
            
            for chunk_id, embedding in self.embeddings.items():
                # Skip if chunk doesn't exist
                if chunk_id not in self.chunks:
                    continue
                
                chunk = self.chunks[chunk_id]
                
                # Apply filters if provided
                if filters and not self._apply_filters(chunk, filters):
                    continue
                
                # Calculate cosine similarity
                embedding_np = np.array(embedding)
                similarity = self._cosine_similarity(query_np, embedding_np)
                
                similarities.append((chunk, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            return similarities[:top_k]
            
        except Exception as e:
            raise RuntimeError(f"Failed to search in-memory store: {str(e)}")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Check if either norm is zero to avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _apply_filters(self, chunk: DocumentChunk, filters: List[MetadataFilter]) -> bool:
        """Apply metadata filters to a chunk"""
        if not chunk.metadata:
            return False
        
        for filter_item in filters:
            field = filter_item.field
            operator = filter_item.operator
            value = filter_item.value
            
            # Skip if field not in metadata
            if field not in chunk.metadata:
                return False
            
            field_value = chunk.metadata[field]
            
            # Apply the filter based on the operator
            if operator == "eq" and field_value != value:
                return False
            elif operator == "neq" and field_value == value:
                return False
            elif operator == "gt" and field_value <= value:
                return False
            elif operator == "gte" and field_value < value:
                return False
            elif operator == "lt" and field_value >= value:
                return False
            elif operator == "lte" and field_value > value:
                return False
            elif operator == "in" and field_value not in value:
                return False
            elif operator == "nin" and field_value in value:
                return False
            elif operator == "contains" and value not in str(field_value):
                return False
            elif operator == "starts_with" and not str(field_value).startswith(value):
                return False
            elif operator == "ends_with" and not str(field_value).endswith(value):
                return False
        
        # If all filters pass, return True
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from the vector store"""
        try:
            if document_id not in self.document_ids:
                return True
            
            # Get chunk IDs for this document
            chunk_ids = self.document_ids[document_id]
            
            # Delete chunks and embeddings
            for chunk_id in chunk_ids:
                if chunk_id in self.chunks:
                    del self.chunks[chunk_id]
                
                if chunk_id in self.embeddings:
                    del self.embeddings[chunk_id]
            
            # Remove document ID
            del self.document_ids[document_id]
            
            # Save changes to disk
            await self._save_data()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from in-memory store: {str(e)}")
    
    async def get_document_count(self) -> int:
        """Get the number of unique documents in the vector store"""
        return len(self.document_ids)
    
    async def get_chunk_count(self) -> int:
        """Get the number of chunks in the vector store"""
        return len(self.chunks)
