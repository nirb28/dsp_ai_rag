from typing import List, Dict, Any, Optional, Tuple
import os
import uuid

from app.models.document import DocumentChunk
from app.models.config_options import VectorStoreConfig
from app.models.retrieval import MetadataFilter
from app.services.vectorstores.base import BaseVectorStore
from app.core.config import settings


class WeaviateVectorStore(BaseVectorStore):
    """Vector store using Weaviate"""
    
    async def initialize(self):
        """Initialize the Weaviate vector store"""
        try:
            import weaviate
            
            # Get API key and URL from config or settings
            url = self.config.weaviate_url or settings.WEAVIATE_URL
            api_key = self.config.weaviate_api_key or settings.WEAVIATE_API_KEY
            
            if not url:
                raise ValueError("Weaviate URL is required")
            
            # Initialize Weaviate client
            auth_config = weaviate.AuthApiKey(api_key=api_key) if api_key else None
            
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=auth_config
            )
            
            # Define class name for this collection
            self.class_name = self.config.collection_name.capitalize()
            
            # Check if class exists, create if it doesn't
            if not self.client.schema.exists(self.class_name):
                # Create class schema
                schema = {
                    "class": self.class_name,
                    "vectorizer": "none",  # We provide our own vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The content of the document chunk",
                        },
                        {
                            "name": "documentId",
                            "dataType": ["string"],
                            "description": "The ID of the parent document",
                            "indexInverted": True
                        },
                        {
                            "name": "chunkIndex",
                            "dataType": ["int"],
                            "description": "The index of this chunk in the document",
                        },
                        {
                            "name": "pageNumber",
                            "dataType": ["int"],
                            "description": "The page number of this chunk",
                            "indexInverted": True
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Additional metadata for the chunk",
                        }
                    ]
                }
                
                self.client.schema.create_class(schema)
            
            # Track document counts
            self.document_ids = set()
            self._load_document_ids()
                
        except ImportError:
            raise ImportError("Failed to import weaviate-client. Please install with: pip install weaviate-client")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Weaviate vector store: {str(e)}")
    
    def _load_document_ids(self):
        """Load existing document IDs from Weaviate"""
        try:
            # Query to get all unique document IDs
            query = f"""
            {{
              Get {{
                {self.class_name} (limit: 10000) {{
                  documentId
                }}
              }}
            }}
            """
            
            result = self.client.query.raw(query)
            
            if result and "data" in result and "Get" in result["data"]:
                items = result["data"]["Get"].get(self.class_name, [])
                
                for item in items:
                    if "documentId" in item:
                        self.document_ids.add(item["documentId"])
        except Exception:
            # If this fails, we start with an empty set
            pass
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                return True
            
            # Prepare batch for import
            with self.client.batch as batch:
                batch.batch_size = 100  # Set batch size
                
                for chunk in chunks:
                    # Prepare metadata
                    metadata = chunk.metadata if chunk.metadata else {}
                    
                    # Create properties
                    properties = {
                        "content": chunk.content,
                        "documentId": chunk.document_id,
                        "chunkIndex": chunk.chunk_index,
                        "metadata": metadata
                    }
                    
                    # Add page number if available
                    if chunk.page_number is not None:
                        properties["pageNumber"] = chunk.page_number
                    
                    # Add to Weaviate with the embedding
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        uuid=chunk.chunk_id,
                        vector=chunk.embedding
                    )
                    
                    # Track document ID
                    self.document_ids.add(chunk.document_id)
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Weaviate: {str(e)}")
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filters: Optional[List[MetadataFilter]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for document chunks similar to the query embedding"""
        try:
            # Build where filter if needed
            where_filter = None
            
            if filters:
                filter_conditions = []
                
                for filter_item in filters:
                    field = filter_item.field
                    operator = filter_item.operator
                    value = filter_item.value
                    
                    # Map field name to Weaviate property name
                    weaviate_field = "documentId" if field == "document_id" else field
                    weaviate_field = "pageNumber" if field == "page_number" else weaviate_field
                    
                    if field in ["document_id", "page_number"]:
                        # For standard fields
                        if operator == "eq":
                            filter_conditions.append({
                                "path": [weaviate_field],
                                "operator": "Equal",
                                "valueString" if field == "document_id" else "valueInt": value
                            })
                    else:
                        # For metadata fields
                        if operator == "eq":
                            filter_conditions.append({
                                "path": ["metadata", field],
                                "operator": "Equal",
                                "valueText": value if isinstance(value, str) else str(value)
                            })
                
                # Build the where filter
                if filter_conditions:
                    if len(filter_conditions) == 1:
                        where_filter = filter_conditions[0]
                    else:
                        where_filter = {
                            "operator": "And",
                            "operands": filter_conditions
                        }
            
            # Perform the search
            result = self.client.query.get(
                self.class_name,
                ["content", "documentId", "chunkIndex", "pageNumber", "metadata", "_additional { id certainty }"]
            ).with_near_vector({
                "vector": query_embedding
            }).with_limit(top_k)
            
            # Add where filter if needed
            if where_filter:
                result = result.with_where(where_filter)
                
            # Execute query
            response = result.do()
            
            # Process results
            chunks_with_scores = []
            
            if (response and 
                "data" in response and 
                "Get" in response["data"] and 
                self.class_name in response["data"]["Get"]):
                
                items = response["data"]["Get"][self.class_name]
                
                for item in items:
                    # Extract data
                    chunk_id = item["_additional"]["id"]
                    score = item["_additional"]["certainty"]
                    content = item["content"]
                    document_id = item["documentId"]
                    chunk_index = item["chunkIndex"]
                    page_number = item.get("pageNumber")
                    metadata = item.get("metadata", {})
                    
                    # Create document chunk
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        content=content,
                        metadata=metadata,
                        page_number=page_number,
                        chunk_index=chunk_index
                    )
                    
                    chunks_with_scores.append((chunk, float(score)))
            
            return chunks_with_scores
            
        except Exception as e:
            raise RuntimeError(f"Failed to search Weaviate: {str(e)}")
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from the vector store"""
        try:
            # Delete all objects with matching document ID
            where_filter = {
                "path": ["documentId"],
                "operator": "Equal",
                "valueString": document_id
            }
            
            self.client.batch.delete_objects(
                class_name=self.class_name,
                where=where_filter
            )
            
            # Remove from tracked document IDs
            if document_id in self.document_ids:
                self.document_ids.remove(document_id)
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from Weaviate: {str(e)}")
    
    async def get_document_count(self) -> int:
        """Get the number of unique documents in the vector store"""
        return len(self.document_ids)
    
    async def get_chunk_count(self) -> int:
        """Get the number of chunks in the vector store"""
        try:
            result = self.client.query.aggregate(self.class_name).with_meta_count().do()
            
            if (result and 
                "data" in result and 
                "Aggregate" in result["data"] and 
                self.class_name in result["data"]["Aggregate"]):
                
                return result["data"]["Aggregate"][self.class_name]["meta"]["count"]
            
            return 0
        except Exception:
            return 0
