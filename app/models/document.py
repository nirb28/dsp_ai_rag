from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import uuid


class DocumentStatus(str, Enum):
    """Status of document processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class DocumentType(str, Enum):
    """Types of supported documents"""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    MARKDOWN = "markdown"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    

class DocumentChunk(BaseModel):
    """Chunk of a document"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    page_number: Optional[int] = None
    chunk_index: int
    

class Document(BaseModel):
    """Document model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    user_id: str
    config_id: Optional[str] = None
    title: Optional[str] = None
    document_type: DocumentType
    mime_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None
    size_bytes: int
    chunk_count: Optional[int] = None
    
    class Config:
        orm_mode = True


class DocumentCreate(BaseModel):
    """Request model for document creation"""
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    config_id: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response model for document"""
    id: str
    filename: str
    title: Optional[str] = None
    document_type: str
    status: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    size_bytes: int
    chunk_count: Optional[int] = None


class DocumentList(BaseModel):
    """Response model for list of documents"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
