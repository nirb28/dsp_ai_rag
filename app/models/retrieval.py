from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid


class MetadataFilter(BaseModel):
    """Filter for document metadata"""
    field: str
    operator: str  # "eq", "neq", "gt", "gte", "lt", "lte", "in", "nin", "contains", "starts_with", "ends_with"
    value: Any


class RetrievalRequest(BaseModel):
    """Request model for retrieval"""
    query: str
    config_id: Optional[str] = None
    filters: List[MetadataFilter] = Field(default_factory=list)
    top_k: int = 5
    rerank: bool = False
    include_content: bool = True
    include_metadata: bool = True
    include_embeddings: bool = False
    namespace: Optional[str] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """Response model for retrieved document chunk"""
    chunk_id: str
    document_id: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    page_number: Optional[int] = None
    chunk_index: int
    score: float
    embedding: Optional[List[float]] = None


class RetrievalResponse(BaseModel):
    """Response model for retrieval"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    results: List[RetrievedChunk]
    total_chunks_searched: int
    elapsed_time_ms: float
    config_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class GenerationRequest(BaseModel):
    """Request model for generation"""
    query: str
    config_id: Optional[str] = None
    filters: List[MetadataFilter] = Field(default_factory=list)
    retrieval_top_k: int = 5
    rerank: bool = False
    include_sources: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class GenerationSource(BaseModel):
    """Source information for generation response"""
    document_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float


class GenerationResponse(BaseModel):
    """Response model for generation"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    response: str
    sources: List[GenerationSource] = Field(default_factory=list)
    elapsed_time_ms: float
    model_used: str
    config_id: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class StreamGenerationResponse(BaseModel):
    """Model for streaming generation response chunks"""
    request_id: str
    chunk: str
    finish_reason: Optional[str] = None
    is_complete: bool = False
