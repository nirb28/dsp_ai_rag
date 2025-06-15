import requests
import json
from typing import Dict, Any, List, Optional
import time

class RAGClient:
    """Client for interacting with the RAG API"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: str = None):
        """
        Initialize the RAG API client
        
        Args:
            base_url: Base URL for the RAG API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
    
    def _handle_response(self, response):
        """Handle API response and errors"""
        try:
            if response.status_code >= 400:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
            return response.json()
        except ValueError:
            return {"error": "Invalid JSON response"}
            
    def _get_headers(self):
        """Get headers with authentication if available"""
        headers = {}
        if hasattr(self, 'api_key') and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self._get_headers())
            return self._handle_response(response)
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the system"""
        try:
            # Fixed endpoint with trailing slash
            response = requests.get(f"{self.base_url}/documents/", headers=self._get_headers())
            result = self._handle_response(response)
            if "error" in result:
                return []
            return result.get("documents", [])
        except Exception as e:
            return []
    
    def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response for a given query
        
        Args:
            request: Dictionary containing generation parameters
                - query: The query string
                - enable_retrieval: Whether to use retrieval
                - temperature: Temperature for generation
                - max_tokens: Maximum tokens to generate
                - retrieval_top_k: Number of chunks to retrieve
                - include_sources: Whether to include sources in the response
                - document_id: Optional document ID to restrict retrieval to
                - model: Optional model name for completion
        
        Returns:
            Dictionary containing the response
        """
        try:
            # Correct endpoint for generations
            url = f"{self.base_url}/generations/"
            
            # Create a proper generation request
            generation_request = {
                "query": request["query"],
                "enable_retrieval": request.get("enable_retrieval", True),
                "temperature": request.get("temperature", 0.7),
                "max_tokens": request.get("max_tokens", None),
                "retrieval_top_k": request.get("retrieval_top_k", 5),
                "include_sources": request.get("include_sources", True),
            }
            
            # Add model if specified
            if "model" in request and request["model"]:
                generation_request["model"] = request["model"]
            
            # Add document ID filter if specified
            if request.get("document_id"):
                generation_request["filters"] = [
                    {
                        "field": "document_id",
                        "operator": "eq",
                        "value": request["document_id"]
                    }
                ]
            
            # Add authentication headers
            headers = self._get_headers()
            headers["Content-Type"] = "application/json"
            
            response = requests.post(url, json=generation_request, headers=headers)
            result = self._handle_response(response)
            
            # Process result to normalize the response format
            if "error" not in result:
                # Convert retrieved_chunks to sources for consistency
                if "retrieved_chunks" in result and result["retrieved_chunks"]:
                    result["sources"] = [
                        {
                            "document_id": chunk["document_id"],
                            "chunk_id": chunk["chunk_id"],
                            "content": chunk.get("content", ""),
                            "metadata": chunk.get("metadata", {}),
                            "score": chunk.get("score", 0)
                        }
                        for chunk in result["retrieved_chunks"]
                    ]
            
            return result
            
        except Exception as e:
            return {"error": f"Generation error: {str(e)}", "response": "", "sources": []}
    
    def upload_document(self, file, metadata: Optional[Dict[str, Any]] = None, chunking_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload a document to the system
        
        Args:
            file: File object to upload
            metadata: Optional metadata for the document
            chunking_config: Optional chunking configuration
                - strategy: Chunking strategy (character, token, sentence, paragraph, recursive)
                - chunk_size: Size of each chunk
                - chunk_overlap: Overlap between chunks
        
        Returns:
            Dictionary containing the document ID and processing status
        """
        try:
            # Corrected endpoint - use root documents endpoint
            url = f"{self.base_url}/documents/"
            
            # Prepare file for upload
            files = {"file": file}
            
            # Prepare form data
            data = {}
            
            # Add title from metadata if available
            if metadata and "title" in metadata:
                data["title"] = metadata["title"]
            
            # Add metadata as JSON string (this is what the backend expects)
            if metadata:
                data["metadata"] = json.dumps(metadata)
                
            # Add config_id if available in chunking_config
            if chunking_config and "config_id" in chunking_config:
                data["config_id"] = chunking_config["config_id"]
            
            # Headers for authentication (if needed)
            headers = {}
            if hasattr(self, 'api_key') and self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make the request with a timeout
            response = requests.post(
                url, 
                files=files, 
                data=data, 
                headers=headers,
                timeout=300  # 5-minute timeout
            )
            result = self._handle_response(response)
            
            return result
        except requests.exceptions.Timeout:
            return {"error": "Upload timed out. The document may be too large or the server is busy."}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error. Please check that the API server is running."}
        except Exception as e:
            return {"error": f"Upload error: {str(e)}"}
