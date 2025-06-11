from typing import List, Dict, Any, Optional, Tuple
import os
import json
import uuid

from app.models.document import DocumentChunk
from app.models.config_options import VectorStoreConfig
from app.models.retrieval import MetadataFilter
from app.services.vectorstores.base import BaseVectorStore
from app.core.config import settings


class PineconeVectorStore(BaseVectorStore):
    """Vector store using Pinecone"""
    
    async def initialize(self):
        """Initialize the Pinecone vector store"""
        try:
            import pinecone
            
            # Get API key from config or settings
            api_key = self.config.pinecone_api_key or settings.PINECONE_API_KEY
            
            if not api_key:
                raise ValueError("Pinecone API key is required")
            
            # Get environment from config or settings
            environment = self.config.pinecone_environment or settings.PINECONE_ENVIRONMENT
            
            if not environment:
                raise ValueError("Pinecone environment is required")
            
            # Initialize Pinecone
            pinecone.init(api_key=api_key, environment=environment)
            
            # Get or create index
            self.index_name = self.config.collection_name
            self.namespace = self.config.namespace or "default"
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                # Create index if it doesn't exist
                dimension = self.config.dimension or 1536  # Default to OpenAI embedding dimension
                
                # Create the index with cosine similarity metric
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine"
                )
            
            # Connect to the index
            self.index = pinecone.Index(self.index_name)
            
            # Create a mapping to store document IDs and their chunk counts
            self.document_count_path = os.path.join(
                settings.VECTOR_DB_DIR, 
                "pinecone", 
                f"{self.index_name}_doc_counts.json"
            )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.document_count_path), exist_ok=True)
            
            # Load document counts if exists
            if os.path.exists(self.document_count_path):
                with open(self.document_count_path, 'r') as f:
                    self.document_counts = json.load(f)
            else:
                self.document_counts = {}
                
        except ImportError:
            raise ImportError("Failed to import pinecone. Please install with: pip install pinecone-client")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone vector store: {str(e)}")
    
    async def _save_document_counts(self):
        """Save document counts to disk"""
        with open(self.document_count_path, 'w') as f:
            json.dump(self.document_counts, f)
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                return True
            
            # Prepare vectors for upsert
            vectors = []
            
            for chunk in chunks:
                # Get document ID and chunk content
                doc_id = chunk.document_id
                content = chunk.content
                
                # Create metadata
                metadata = {
                    "document_id": doc_id,
                    "content": content,
                    "chunk_index": chunk.chunk_index,
                }
                
                # Add page number if available
                if chunk.page_number is not None:
                    metadata["page_number"] = chunk.page_number
                
                # Add custom metadata - flatten to string values for Pinecone
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            # Convert complex types to string
                            metadata[key] = str(value)
                
                # Create vector
                vector = {
                    "id": chunk.chunk_id,
                    "values": chunk.embedding,
                    "metadata": metadata
                }
                
                vectors.append(vector)
                
                # Update document counts
                if doc_id not in self.document_counts:
                    self.document_counts[doc_id] = 1
                else:
                    self.document_counts[doc_id] += 1
            
            # Upsert vectors in batches (Pinecone has limits)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)
            
            # Save document counts
            await self._save_document_counts()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Pinecone: {str(e)}")
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: Optional[List[MetadataFilter]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for document chunks similar to the query embedding"""
        try:
            # Prepare filter expression
            filter_expr = None
            
            if filters:
                filter_conditions = []
                
                for filter_item in filters:
                    field = filter_item.field
                    operator = filter_item.operator
                    value = filter_item.value
                    
                    # Pinecone supports a subset of operators
                    if operator == "eq":
                        filter_conditions.append({field: value})
                    elif operator == "in" and isinstance(value, list):
                        filter_conditions.append({field: {"$in": value}})
                    # More operators can be added as needed
                
                # Combine conditions
                if filter_conditions:
                    if len(filter_conditions) == 1:
                        filter_expr = filter_conditions[0]
                    else:
                        filter_expr = {"$and": filter_conditions}
            
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                filter=filter_expr,
                include_metadata=True
            )
            
            # Process results
            chunks_with_scores = []
            
            for match in results["matches"]:
                chunk_id = match["id"]
                score = float(match["score"])  # Pinecone returns cosine similarity
                metadata = match["metadata"]
                
                # Extract fields
                document_id = metadata.get("document_id", "")
                content = metadata.get("content", "")
                chunk_index = metadata.get("chunk_index", 0)
                page_number = metadata.get("page_number", None)
                
                # Remove special fields to get user metadata
                user_metadata = {k: v for k, v in metadata.items() 
                               if k not in ["document_id", "content", "chunk_index", "page_number"]}
                
                # Create document chunk
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=content,
                    metadata=user_metadata,
                    page_number=page_number,
                    chunk_index=chunk_index
                )
                
                chunks_with_scores.append((chunk, score))
            
            return chunks_with_scores
            
        except Exception as e:
            raise RuntimeError(f"Failed to search Pinecone: {str(e)}")
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from the vector store"""
        try:
            # Use Pinecone's delete by metadata feature
            self.index.delete(
                filter={"document_id": document_id},
                namespace=self.namespace
            )
            
            # Update document counts
            if document_id in self.document_counts:
                del self.document_counts[document_id]
                await self._save_document_counts()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from Pinecone: {str(e)}")
    
    async def get_document_count(self) -> int:
        """Get the number of unique documents in the vector store"""
        return len(self.document_counts)
    
    async def get_chunk_count(self) -> int:
        """Get the number of chunks in the vector store"""
        try:
            stats = self.index.describe_index_stats()
            return stats["namespaces"].get(self.namespace, {}).get("vector_count", 0)
        except Exception:
            return sum(self.document_counts.values()) if self.document_counts else 0
