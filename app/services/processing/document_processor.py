import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import tempfile
import shutil

from app.models.document import Document, DocumentStatus, DocumentChunk, DocumentType
from app.models.config_options import RAGConfig
from app.services.chunking.factory import ChunkerFactory
from app.services.embeddings.factory import EmbeddingFactory
from app.services.vectorstores.factory import VectorStoreFactory
from app.core.config import settings


class DocumentProcessor:
    """
    Service for processing documents: extracting text, chunking, embedding, and indexing.
    """
    
    def __init__(self):
        self.document_dir = settings.DOCUMENT_STORAGE_DIR
        os.makedirs(self.document_dir, exist_ok=True)
    
    async def store_document_file(self, file_content: bytes, filename: str) -> str:
        """
        Store a document file on disk.
        
        Args:
            file_content: Raw bytes of the document file
            filename: Original filename
            
        Returns:
            Path to the stored file
        """
        # Generate a unique ID for this file
        file_id = str(uuid.uuid4())
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        
        # Create document directory
        doc_dir = os.path.join(self.document_dir, file_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Create full path with original extension
        file_path = os.path.join(doc_dir, f"original{ext}")
        
        # Write file
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path
    
    async def get_document_text(self, file_path: str, document_type: DocumentType) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            document_type: Type of document
            
        Returns:
            Extracted text
        """
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        try:
            # Handle text files directly
            if document_type == DocumentType.TEXT or ext in ['.txt', '.md', '.csv']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            
            # Handle PDF
            elif document_type == DocumentType.PDF or ext == '.pdf':
                from pypdf import PdfReader
                
                text = ""
                with open(file_path, 'rb') as f:
                    pdf = PdfReader(f)
                    for page in pdf.pages:
                        text += page.extract_text() + "\n\n"
                    
                    return text
            
            # Handle DOCX files
            elif document_type == DocumentType.DOCX or ext in ['.doc', '.docx']:
                import docx
                
                doc = docx.Document(file_path)
                return "\n\n".join([para.text for para in doc.paragraphs])
            
            # Handle HTML files
            elif document_type == DocumentType.HTML or ext in ['.html', '.htm']:
                from bs4 import BeautifulSoup
                
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Get text
                    text = soup.get_text()
                    
                    # Break into lines and remove leading and trailing space
                    lines = (line.strip() for line in text.splitlines())
                    # Break multi-headlines into a line each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    # Drop blank lines
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    return text
            
            # Default to attempting to read as text
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        return f.read()
                except:
                    return f"Unsupported document type: {document_type}"
                    
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    async def process_document(self, document: Document, config: RAGConfig) -> Document:
        """
        Process a document: extract text, chunk, embed, and index.
        
        Args:
            document: Document to process
            config: RAG configuration to use
            
        Returns:
            Updated document with processing status
        """
        try:
            # Update document status to processing
            document.status = DocumentStatus.PROCESSING
            document.processed_at = None
            
            # Extract text from document
            document.content = await self.get_document_text(document.file_path, document.doc_type)
            
            if not document.content or len(document.content) < 10:
                document.status = DocumentStatus.ERROR
                document.error = "Failed to extract text or document is too short"
                return document
            
            # Create chunker
            chunker = await ChunkerFactory.create_chunker(config.chunking_config)
            
            # Chunk document
            chunks = await chunker.chunk_text(
                document.content,
                document.metadata,
                document.id
            )
            
            # Create embedding model
            embedding_model = await EmbeddingFactory.create_embedding_model(config.embedding_model_config)
            
            # Generate embeddings for chunks
            all_texts = [chunk.content for chunk in chunks]
            all_embeddings = await embedding_model.get_embeddings(all_texts)
            
            # Assign embeddings to chunks
            for i, chunk in enumerate(chunks):
                if i < len(all_embeddings):
                    chunk.embedding = all_embeddings[i]
            
            # Create vector store
            vector_store = await VectorStoreFactory.create_vector_store(config.vector_store_config)
            
            # Add chunks to vector store
            success = await vector_store.add_documents(chunks)
            
            if not success:
                document.status = DocumentStatus.ERROR
                document.error = "Failed to add document to vector store"
                return document
            
            # Update document with chunk information
            document.chunk_count = len(chunks)
            document.status = DocumentStatus.PROCESSED
            document.processed_at = datetime.utcnow()
            
            return document
            
        except Exception as e:
            document.status = DocumentStatus.ERROR
            document.error = f"Processing error: {str(e)}"
            return document
    
    async def reprocess_document(self, document: Document, config: RAGConfig) -> Document:
        """
        Reprocess a document with a new configuration.
        
        Args:
            document: Document to reprocess
            config: New RAG configuration to use
            
        Returns:
            Updated document with new processing status
        """
        try:
            # Update document status to processing
            document.status = DocumentStatus.PROCESSING
            document.processed_at = None
            
            # Delete existing vector store entries if needed
            vector_store = await VectorStoreFactory.create_vector_store(config.vector_store_config)
            await vector_store.delete_document(document.id)
            
            # Process document with new config
            return await self.process_document(document, config)
            
        except Exception as e:
            document.status = DocumentStatus.ERROR
            document.error = f"Reprocessing error: {str(e)}"
            return document
    
    async def delete_document_file(self, document: Document) -> bool:
        """
        Delete a document's files from disk.
        
        Args:
            document: Document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if document.file_path and os.path.exists(document.file_path):
                # Get document directory (parent of file path)
                doc_dir = os.path.dirname(document.file_path)
                
                # Remove the entire directory
                if os.path.exists(doc_dir):
                    shutil.rmtree(doc_dir)
            
            return True
        except Exception:
            return False
