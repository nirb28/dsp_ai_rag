from typing import List, Dict, Any, Optional, Tuple
import os
import json
import numpy as np
import faiss
import pickle

from app.models.document import DocumentChunk
from app.models.config_options import VectorStoreConfig
from app.models.retrieval import MetadataFilter
from app.services.vectorstores.base import BaseVectorStore
from app.core.config import settings


class FAISSVectorStore(BaseVectorStore):
    """Vector store using FAISS"""
    
    async def initialize(self):
        """Initialize the FAISS vector store"""
        try:
            self.collection_name = self.config.collection_name
            self.persist_dir = os.path.join(
                settings.VECTOR_DB_DIR,
                "faiss",
                self.collection_name
            )
            
            # Create directory if it doesn't exist
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Paths to the index and metadata files
            self.index_path = os.path.join(self.persist_dir, "index.faiss")
            self.metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
            
            # Check if index exists, otherwise create it
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                await self._load_index()
            else:
                # Create empty metadata store and index
                self.metadata = {}
                self.document_ids = set()
                self.index = None
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS vector store: {str(e)}")
    
    async def _load_index(self):
        """Load the FAISS index and metadata from disk"""
        try:
            self.index = faiss.read_index(self.index_path)
            
            with open(self.metadata_path, 'rb') as f:
                stored_data = pickle.load(f)
                self.metadata = stored_data.get('metadata', {})
                self.document_ids = stored_data.get('document_ids', set())
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {str(e)}")
    
    async def _save_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'document_ids': self.document_ids
                }, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS index: {str(e)}")
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                return True
            
            # Extract embeddings and create a matrix
            embeddings = [chunk.embedding for chunk in chunks]
            embedding_dim = len(embeddings[0])
            embedding_matrix = np.array(embeddings).astype('float32')
            
            # Initialize index if not exists
            if self.index is None:
                # Choose the appropriate index type based on config or default
                if self.config.faiss_index_type == "flat":
                    self.index = faiss.IndexFlatL2(embedding_dim)
                elif self.config.faiss_index_type == "ivf":
                    # IVF requires training, so we'll use a flat index here and create IVF later
                    self.index = faiss.IndexFlatL2(embedding_dim)
                else:
                    # Default to a good general-purpose index
                    self.index = faiss.IndexFlatL2(embedding_dim)
            
            # Add the embeddings to the index
            start_idx = self.index.ntotal
            self.index.add(embedding_matrix)
            
            # Store metadata for each chunk
            for i, chunk in enumerate(chunks):
                idx = start_idx + i
                self.metadata[idx] = {
                    'chunk_id': chunk.chunk_id,
                    'document_id': chunk.document_id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.chunk_index
                }
                self.document_ids.add(chunk.document_id)
            
            # Save the updated index and metadata
            await self._save_index()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to FAISS: {str(e)}")
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: Optional[List[MetadataFilter]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for document chunks similar to the query embedding"""
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            # Convert query embedding to numpy array
            query_np = np.array([query_embedding]).astype('float32')
            
            # Search the index - returns distances and indices
            distances, indices = self.index.search(query_np, self.index.ntotal)
            
            # Map indices to document chunks with scores
            results = []
            for i in range(min(len(indices[0]), self.index.ntotal)):
                idx = indices[0][i]
                distance = distances[0][i]
                
                # Skip if index not found in metadata
                if idx not in self.metadata:
                    continue
                
                metadata_entry = self.metadata[idx]
                
                # Apply filters if provided
                if filters and not self._apply_filters(metadata_entry, filters):
                    continue
                
                # Convert distance to similarity score (FAISS returns L2 distance)
                # Smaller distance is better, so we invert it
                # Normalize to 0-1 range by using exp(-distance)
                score = np.exp(-distance).item()
                
                chunk = DocumentChunk(
                    chunk_id=metadata_entry['chunk_id'],
                    document_id=metadata_entry['document_id'],
                    content=metadata_entry['content'],
                    metadata=metadata_entry['metadata'],
                    page_number=metadata_entry['page_number'],
                    chunk_index=metadata_entry['chunk_index']
                )
                
                results.append((chunk, score))
                
                # Stop once we have top_k results
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to search FAISS: {str(e)}")
    
    def _apply_filters(self, metadata_entry: Dict[str, Any], filters: List[MetadataFilter]) -> bool:
        """Apply metadata filters to a search result"""
        item_metadata = metadata_entry.get('metadata', {})
        
        for filter_item in filters:
            field = filter_item.field
            operator = filter_item.operator
            value = filter_item.value
            
            # Skip if field not in metadata
            if field not in item_metadata:
                return False
            
            field_value = item_metadata[field]
            
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
            if self.index is None:
                return True
            
            # Find all chunks for this document
            chunks_to_delete = []
            for idx, metadata_entry in self.metadata.items():
                if metadata_entry['document_id'] == document_id:
                    chunks_to_delete.append(idx)
            
            if not chunks_to_delete:
                return True
            
            # Remove document from document_ids
            if document_id in self.document_ids:
                self.document_ids.remove(document_id)
            
            # Since FAISS doesn't support direct removal, we need to rebuild the index
            # Extract all embeddings except the ones we want to delete
            valid_indices = [i for i in range(self.index.ntotal) if i not in chunks_to_delete]
            
            if not valid_indices:
                # If all documents are deleted, reset the index
                self.index = None
                self.metadata = {}
                await self._save_index()
                return True
            
            # Get dimension from existing index
            dimension = self.index.d
            
            # Create a new index
            new_index = faiss.IndexFlatL2(dimension)
            
            # Search with dummy query to get all vectors
            dummy_query = np.zeros((1, dimension), dtype='float32')
            _, all_indices = self.index.search(dummy_query, self.index.ntotal)
            
            # Extract vectors for valid indices
            valid_vectors = []
            new_metadata = {}
            
            for i, old_idx in enumerate(valid_indices):
                if old_idx in self.metadata:
                    vector = self.index.reconstruct(old_idx).reshape(1, -1)
                    valid_vectors.append(vector)
                    new_metadata[i] = self.metadata[old_idx]
            
            if valid_vectors:
                valid_embedding_matrix = np.vstack(valid_vectors)
                new_index.add(valid_embedding_matrix)
            
            # Replace old index and metadata
            self.index = new_index
            self.metadata = new_metadata
            
            # Save the updated index and metadata
            await self._save_index()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from FAISS: {str(e)}")
    
    async def get_document_count(self) -> int:
        """Get the number of unique documents in the vector store"""
        return len(self.document_ids)
    
    async def get_chunk_count(self) -> int:
        """Get the number of chunks in the vector store"""
        if self.index is None:
            return 0
        return self.index.ntotal
