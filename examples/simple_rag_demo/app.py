import os
import sys
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import logging

# Add parent directory to path to import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import RAG components
from app.models.document import Document, DocumentStatus, DocumentType
from app.models.config_options import (
    RAGConfig, 
    ChunkingConfig,
    VectorStoreConfig,
    EmbeddingModelConfig,
    CompletionModelConfig,
    ReRankingConfig,
    ChunkingStrategy,
    VectorStoreType,
    EmbeddingModelType,
    CompletionModelType
)
from app.models.retrieval import (
    GenerationRequest,
    RetrievalRequest,
    MetadataFilter
)
from app.services.processing.document_processor import DocumentProcessor
from app.services.retrieval.retriever import RetrievalService
from app.services.generation.generator import GenerationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Demo API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
retrieval_service = RetrievalService()
generation_service = GenerationService()

# In-memory storage for demo
documents = {}
current_config = None
processing_tasks = {}

# Default configuration
def get_default_config() -> RAGConfig:
    return RAGConfig(
        chunking_config=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=1000,
            chunk_overlap=200
        ),
        vector_store_config=VectorStoreConfig(
            store_type=VectorStoreType.IN_MEMORY,
            collection_name="demo_collection"
        ),
        embedding_model_config=EmbeddingModelConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        ),
        completion_model_config=CompletionModelConfig(
            model_type=CompletionModelType.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=800,
            system_prompt="You are a helpful assistant that provides information based on the provided documents."
        ),
        reranking_config=ReRankingConfig(
            enabled=False,
            method="cross_encoder",
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

# Initialize default configuration
current_config = get_default_config()

# Pydantic models for API
class ConfigUpdateRequest(BaseModel):
    chunking_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model: Optional[str] = None
    completion_model: Optional[str] = None
    temperature: Optional[float] = None
    enable_reranking: Optional[bool] = None

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    enable_retrieval: bool = True
    top_k: int = 5
    include_sources: bool = True

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    error: Optional[str] = None
    processed_at: Optional[datetime] = None
    chunk_count: Optional[int] = None

# Background task for processing documents
async def process_document_task(doc_id: str):
    try:
        if doc_id in documents:
            document = documents[doc_id]
            document = await document_processor.process_document(document, current_config)
            documents[doc_id] = document
            logger.info(f"Document {doc_id} processed successfully")
    except Exception as e:
        if doc_id in documents:
            documents[doc_id].status = DocumentStatus.ERROR
            documents[doc_id].error = str(e)
        logger.error(f"Error processing document {doc_id}: {str(e)}")
    finally:
        if doc_id in processing_tasks:
            del processing_tasks[doc_id]

# Serve static files from the frontend directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "frontend/static")), name="static")

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    with open(os.path.join(os.path.dirname(__file__), "frontend/index.html"), "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: str = Form(default="text")
):
    """Upload a document for processing"""
    try:
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Read file content
        file_content = await file.read()
        
        # Store document file
        file_path = await document_processor.store_document_file(file_content, file.filename)
        
        # Determine document type
        if document_type == "auto":
            # Auto-detect based on file extension
            _, ext = os.path.splitext(file.filename)
            ext = ext.lower()
            if ext in ['.pdf']:
                doc_type = DocumentType.PDF
            elif ext in ['.docx', '.doc']:
                doc_type = DocumentType.DOCX
            elif ext in ['.html', '.htm']:
                doc_type = DocumentType.HTML
            else:
                doc_type = DocumentType.TEXT
        else:
            doc_type = DocumentType(document_type)
        
        # Create document record
        document = Document(
            id=doc_id,
            filename=file.filename,
            file_path=file_path,
            doc_type=doc_type,
            status=DocumentStatus.PENDING,
            upload_time=datetime.utcnow(),
            metadata={"original_filename": file.filename}
        )
        
        # Store in memory
        documents[doc_id] = document
        
        # Process document in background
        background_tasks.add_task(process_document_task, doc_id)
        processing_tasks[doc_id] = True
        
        return {"document_id": doc_id, "status": "processing"}
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """Get a list of all documents"""
    result = []
    for doc_id, doc in documents.items():
        result.append({
            "id": doc.id,
            "filename": doc.filename,
            "status": doc.status.value,
            "error": doc.error,
            "processed_at": doc.processed_at,
            "chunk_count": doc.chunk_count
        })
    return result

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get details for a specific document"""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents[document_id]
    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status.value,
        "error": doc.error,
        "processed_at": doc.processed_at,
        "chunk_count": doc.chunk_count,
        "content": doc.content[:1000] + "..." if len(doc.content or "") > 1000 else doc.content
    }

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from vector store
    try:
        vector_store = await app.vector_store_factory.create_vector_store(current_config.vector_store_config)
        await vector_store.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Error deleting document from vector store: {str(e)}")
    
    # Delete files
    try:
        await document_processor.delete_document_file(documents[document_id])
    except Exception as e:
        logger.warning(f"Error deleting document files: {str(e)}")
    
    # Remove from memory
    del documents[document_id]
    
    return {"status": "deleted"}

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "chunking": {
            "strategy": current_config.chunking_config.strategy.value,
            "chunk_size": current_config.chunking_config.chunk_size,
            "chunk_overlap": current_config.chunking_config.chunk_overlap
        },
        "embedding": {
            "model_type": current_config.embedding_model_config.model_type.value,
            "model_name": current_config.embedding_model_config.model_name,
            "device": current_config.embedding_model_config.device
        },
        "completion": {
            "model_type": current_config.completion_model_config.model_type.value,
            "model_name": current_config.completion_model_config.model_name,
            "temperature": current_config.completion_model_config.temperature,
            "max_tokens": current_config.completion_model_config.max_tokens
        },
        "reranking": {
            "enabled": current_config.reranking_config.enabled,
            "method": current_config.reranking_config.method
        }
    }

@app.post("/api/config")
async def update_config(request: ConfigUpdateRequest):
    """Update configuration"""
    global current_config
    
    # Update chunking config
    if request.chunking_strategy:
        current_config.chunking_config.strategy = ChunkingStrategy(request.chunking_strategy)
    if request.chunk_size:
        current_config.chunking_config.chunk_size = request.chunk_size
    if request.chunk_overlap:
        current_config.chunking_config.chunk_overlap = request.chunk_overlap
    
    # Update embedding config
    if request.embedding_model:
        if request.embedding_model.startswith("local:"):
            current_config.embedding_model_config.model_type = EmbeddingModelType.LOCAL
            current_config.embedding_model_config.local_model_path = request.embedding_model[6:]
        elif request.embedding_model.startswith("openai:"):
            current_config.embedding_model_config.model_type = EmbeddingModelType.OPENAI
            current_config.embedding_model_config.model_name = request.embedding_model[7:]
        else:
            current_config.embedding_model_config.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
            current_config.embedding_model_config.model_name = request.embedding_model
    
    # Update completion config
    if request.completion_model:
        if request.completion_model.startswith("local:"):
            current_config.completion_model_config.model_type = CompletionModelType.LOCAL
            current_config.completion_model_config.local_model_path = request.completion_model[6:]
        else:
            current_config.completion_model_config.model_type = CompletionModelType.OPENAI
            current_config.completion_model_config.model_name = request.completion_model
    
    if request.temperature is not None:
        current_config.completion_model_config.temperature = request.temperature
    
    # Update reranking config
    if request.enable_reranking is not None:
        current_config.reranking_config.enabled = request.enable_reranking
    
    return {"status": "updated"}

@app.post("/api/query")
async def query(request: QueryRequest):
    """Query the RAG system"""
    try:
        # Create generation request
        generation_req = GenerationRequest(
            query=request.query,
            enable_retrieval=request.enable_retrieval,
            include_sources=request.include_sources,
            retrieval_top_k=request.top_k
        )
        
        # Add document filters if specified
        if request.document_ids:
            filters = []
            for doc_id in request.document_ids:
                filters.append(MetadataFilter(
                    field="document_id",
                    operator="eq",
                    value=doc_id
                ))
            generation_req.filters = filters
        
        # Generate response
        response = await generation_service.generate(
            request=generation_req,
            config=current_config,
            user_id="demo_user"
        )
        
        # Format response
        result = {
            "query": response.query,
            "response": response.response,
            "error": response.error,
            "generation_time": response.generation_time
        }
        
        # Include sources if requested
        if request.include_sources and response.retrieved_chunks:
            result["sources"] = []
            for chunk in response.retrieved_chunks:
                doc_id = chunk.document_id
                doc_info = {
                    "document_id": doc_id,
                    "filename": documents[doc_id].filename if doc_id in documents else "Unknown",
                    "content": chunk.content,
                    "score": chunk.score
                }
                if chunk.page_number:
                    doc_info["page_number"] = chunk.page_number
                
                result["sources"].append(doc_info)
        
        return result
    
    except Exception as e:
        logger.error(f"Error querying: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying: {str(e)}")

@app.post("/api/stream")
async def stream_query(request: QueryRequest):
    """Stream query response from the RAG system"""
    async def generate():
        try:
            # Create generation request
            generation_req = GenerationRequest(
                query=request.query,
                enable_retrieval=request.enable_retrieval,
                include_sources=request.include_sources,
                retrieval_top_k=request.top_k
            )
            
            # Add document filters if specified
            if request.document_ids:
                filters = []
                for doc_id in request.document_ids:
                    filters.append(MetadataFilter(
                        field="document_id",
                        operator="eq",
                        value=doc_id
                    ))
                generation_req.filters = filters
            
            # Stream response
            async for chunk in generation_service.generate_stream(
                request=generation_req,
                config=current_config,
                user_id="demo_user"
            ):
                if chunk.sources and request.include_sources:
                    # Format sources
                    sources_data = []
                    for source in chunk.sources:
                        doc_id = source.document_id
                        doc_info = {
                            "document_id": doc_id,
                            "filename": documents[doc_id].filename if doc_id in documents else "Unknown",
                            "content": source.content,
                            "score": source.score
                        }
                        if source.page_number:
                            doc_info["page_number"] = source.page_number
                        
                        sources_data.append(doc_info)
                    
                    yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                    
                # Yield text chunk
                if chunk.chunk:
                    yield f"data: {json.dumps({'text': chunk.chunk})}\n\n"
                
                # Handle errors
                if chunk.error:
                    yield f"data: {json.dumps({'error': chunk.error})}\n\n"
                
                # Handle completion
                if chunk.is_done:
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
                    
        except Exception as e:
            logger.error(f"Error in stream: {str(e)}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
