from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import json
import uuid
import os
import shutil
from datetime import datetime
from pathlib import Path

from app.models.document import (
    Document, DocumentCreate, DocumentResponse, DocumentList,
    DocumentStatus, DocumentType
)
from app.models.config_options import RAGConfig
from app.core.security import get_current_user, get_optional_current_user
from app.core.config import settings

router = APIRouter()

# Simulated database for documents
# In production, replace this with a proper database
documents_db = {}

def determine_document_type(filename: str) -> DocumentType:
    """Determine document type from filename extension"""
    extension = filename.split('.')[-1].lower()
    
    extension_mapping = {
        'pdf': DocumentType.PDF,
        'docx': DocumentType.DOCX,
        'doc': DocumentType.DOCX,
        'pptx': DocumentType.PPTX,
        'ppt': DocumentType.PPTX,
        'txt': DocumentType.TEXT,
        'html': DocumentType.HTML,
        'htm': DocumentType.HTML,
        'csv': DocumentType.CSV,
        'json': DocumentType.JSON,
        'md': DocumentType.MARKDOWN,
        'png': DocumentType.IMAGE,
        'jpg': DocumentType.IMAGE,
        'jpeg': DocumentType.IMAGE,
        'gif': DocumentType.IMAGE,
        'mp3': DocumentType.AUDIO,
        'wav': DocumentType.AUDIO,
        'mp4': DocumentType.VIDEO,
        'mov': DocumentType.VIDEO,
    }
    
    return extension_mapping.get(extension, DocumentType.TEXT)


def determine_mime_type(document_type: DocumentType) -> str:
    """Determine MIME type from document type"""
    mime_mapping = {
        DocumentType.PDF: 'application/pdf',
        DocumentType.DOCX: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        DocumentType.PPTX: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        DocumentType.TEXT: 'text/plain',
        DocumentType.HTML: 'text/html',
        DocumentType.CSV: 'text/csv',
        DocumentType.JSON: 'application/json',
        DocumentType.MARKDOWN: 'text/markdown',
        DocumentType.IMAGE: 'image/*',
        DocumentType.AUDIO: 'audio/*',
        DocumentType.VIDEO: 'video/*',
    }
    
    return mime_mapping.get(document_type, 'application/octet-stream')


@router.post("/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    config_id: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Upload a document for indexing"""
    user_id = current_user.get("sub")
    
    # Validate file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB
    max_size = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
    
    # Create temp file for streaming
    temp_file_path = Path(settings.UPLOAD_DIR) / f"temp_{uuid.uuid4()}"
    try:
        with open(temp_file_path, "wb") as buffer:
            # Read and write file in chunks to avoid memory issues
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > max_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds maximum of {settings.MAX_DOCUMENT_SIZE_MB}MB"
                    )
                buffer.write(chunk)
    except Exception as e:
        if temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )
    
    # Process document metadata
    try:
        doc_metadata = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid metadata JSON format"
        )
    
    # Determine document type and MIME type
    document_type = determine_document_type(file.filename)
    mime_type = determine_mime_type(document_type)
    
    # Create document record
    document_id = str(uuid.uuid4())
    document = Document(
        id=document_id,
        filename=file.filename,
        user_id=user_id,
        config_id=config_id,
        title=title or file.filename,
        document_type=document_type,
        mime_type=mime_type,
        status=DocumentStatus.PENDING,
        metadata=doc_metadata,
        size_bytes=file_size
    )
    
    # Store document in database
    documents_db[document_id] = document.dict()
    
    # Create user document directory
    user_doc_dir = Path(settings.UPLOAD_DIR) / user_id
    user_doc_dir.mkdir(exist_ok=True, parents=True)
    
    # Move file to permanent location
    file_path = user_doc_dir / f"{document_id}_{file.filename}"
    shutil.move(temp_file_path, file_path)
    
    # In a real application, we would trigger an async task to process the document here
    # For simplicity, we'll just mark it as complete
    document.status = DocumentStatus.COMPLETE
    document.updated_at = datetime.now()
    documents_db[document_id] = document.dict()
    
    return DocumentResponse(**document.dict())


@router.get("/", response_model=DocumentList)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """List all documents for the current user"""
    user_id = current_user.get("sub")
    
    # Filter documents for this user
    user_docs = [
        doc for doc in documents_db.values()
        if doc["user_id"] == user_id
    ]
    
    # Apply pagination
    total = len(user_docs)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    
    paginated_docs = user_docs[start_idx:end_idx]
    
    return DocumentList(
        documents=[DocumentResponse(**doc) for doc in paginated_docs],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Get a specific document by ID"""
    user_id = current_user.get("sub")
    
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    document = documents_db[document_id]
    
    # Check that the user has access to this document
    if document["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this document"
        )
    
    return DocumentResponse(**document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Delete a document"""
    user_id = current_user.get("sub")
    
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    document = documents_db[document_id]
    
    # Check that the user has access to this document
    if document["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this document"
        )
    
    # Delete the document file
    filename = document["filename"]
    file_path = Path(settings.UPLOAD_DIR) / user_id / f"{document_id}_{filename}"
    
    if file_path.exists():
        os.remove(file_path)
    
    # Delete the document record
    del documents_db[document_id]
    
    return None


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: str,
    config_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_optional_current_user)
):
    """Reprocess a document with a new configuration"""
    user_id = current_user.get("sub")
    
    if document_id not in documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    
    document = documents_db[document_id]
    
    # Check that the user has access to this document
    if document["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this document"
        )
    
    # Update the document configuration if provided
    if config_id:
        document["config_id"] = config_id
    
    # Set document status to pending for reprocessing
    document["status"] = DocumentStatus.PENDING
    document["updated_at"] = datetime.now().isoformat()
    
    # Update the database
    documents_db[document_id] = document
    
    # In a real application, we would trigger an async task to reprocess the document here
    # For simplicity, we'll just mark it as complete
    document["status"] = DocumentStatus.COMPLETE
    document["updated_at"] = datetime.now().isoformat()
    documents_db[document_id] = document
    
    return DocumentResponse(**document)
