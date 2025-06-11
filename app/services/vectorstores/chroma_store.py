from typing import List, Dict, Any, Optional, Tuple
import os
import uuid

from app.models.document import DocumentChunk
from app.models.config_options import VectorStoreConfig
from app.models.retrieval import MetadataFilter
from app.services.vectorstores.base import BaseVectorStore
from app.core.config import settings


class ChromaVectorStore(BaseVectorStore):
    """Vector store using Chroma DB"""
    
    async def initialize(self):
        """Initialize the Chroma vector store"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.collection_name = self.config.collection_name
            self.persist_dir = os.path.join(
                settings.VECTOR_DB_DIR,
                "chroma",
                self.collection_name
            )
            
            # Create directory if it doesn't exist
            os.makedirs(self.persist_dir, exist_ok=True)
            
            # Initialize Chroma client
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Try to get the existing collection or create a new one
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
            
            # Keep an in-memory mapping of document_id to chunk_ids
            self.doc_to_chunks = {}
            
            # Load existing document-chunk mappings
            existing_chunks = self.collection.get()
            if existing_chunks and "metadatas" in existing_chunks and existing_chunks["metadatas"]:
                for i, metadata in enumerate(existing_chunks["metadatas"]):
                    if metadata and "document_id" in metadata:
                        doc_id = metadata["document_id"]
                        chunk_id = existing_chunks["ids"][i]
                        
                        if doc_id not in self.doc_to_chunks:
                            self.doc_to_chunks[doc_id] = []
                        
                        self.doc_to_chunks[doc_id].append(chunk_id)
            
        except ImportError:
            raise ImportError("Failed to import chromadb. Please install it with: pip install chromadb")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma vector store: {str(e)}")
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                return True
            
            # Extract data for Chroma
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                # Combine document metadata and chunk metadata
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                }
                
                # Add page number if available
                if chunk.page_number is not None:
                    metadata["page_number"] = chunk.page_number
                    
                # Add custom metadata
                if chunk.metadata:
                    # Flatten metadata for Chroma
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            # Convert complex types to string
                            metadata[key] = str(value)
                
                metadatas.append(metadata)
                
                # Update in-memory mapping
                if chunk.document_id not in self.doc_to_chunks:
                    self.doc_to_chunks[chunk.document_id] = []
                self.doc_to_chunks[chunk.document_id].append(chunk.chunk_id)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Chroma: {str(e)}")
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: Optional[List[MetadataFilter]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for document chunks similar to the query embedding"""
        try:
            # Prepare where clauses for filtering
            where_clauses = {}
            
            if filters:
                for filter_item in filters:
                    field = filter_item.field
                    operator = filter_item.operator
                    value = filter_item.value
                    
                    # Chroma supports a limited set of operators
                    if operator == "eq":
                        where_clauses[field] = value
                    elif operator == "in" and isinstance(value, list):
                        where_clauses[field] = {"$in": value}
                    # Add more operators as needed
            
            # Search in the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clauses if where_clauses else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            chunks_with_scores = []
            
            if (results and 
                "ids" in results and 
                results["ids"] and 
                "documents" in results and 
                "metadatas" in results and 
                "distances" in results):
                
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {}
                    content = results["documents"][0][i] if i < len(results["documents"][0]) else ""
                    distance = results["distances"][0][i] if i < len(results["distances"][0]) else 1.0
                    
                    # Convert distance to similarity score (Chroma uses cosine distance)
                    # For cosine distance, score = 1 - distance
                    score = 1.0 - distance
                    
                    # Extract metadata fields
                    document_id = metadata.get("document_id", "")
                    chunk_index = metadata.get("chunk_index", 0)
                    page_number = metadata.get("page_number", None)
                    
                    # Remove special metadata fields to get user metadata
                    user_metadata = {k: v for k, v in metadata.items() 
                                  if k not in ["document_id", "chunk_index", "page_number"]}
                    
                    # Create document chunk
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=content,
                        metadata=user_metadata,
                        page_number=page_number,
                        chunk_index=chunk_index
                    )
                    
                    chunks_with_scores.append((chunk, float(score)))
            
            return chunks_with_scores
            
        except Exception as e:
            raise RuntimeError(f"Failed to search Chroma: {str(e)}")
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from the vector store"""
        try:
            # Get chunk IDs for this document
            chunk_ids = self.doc_to_chunks.get(document_id, [])
            
            if chunk_ids:
                # Delete chunks from collection
                self.collection.delete(ids=chunk_ids)
                
                # Remove from in-memory mapping
                if document_id in self.doc_to_chunks:
                    del self.doc_to_chunks[document_id]
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from Chroma: {str(e)}")
    
    async def get_document_count(self) -> int:
        """Get the number of unique documents in the vector store"""
        return len(self.doc_to_chunks)
    
    async def get_chunk_count(self) -> int:
        """Get the number of chunks in the vector store"""
        try:
            return self.collection.count()
        except Exception:
            return 0
