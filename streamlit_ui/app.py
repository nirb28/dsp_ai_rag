import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import time
import tempfile

from config import load_config, get_default_rag_config
from api_client import RAGClient

# Page configuration
st.set_page_config(
    page_title="RAG UI Demo",
    page_icon="🔍",
    layout="wide",
)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_config" not in st.session_state:
    st.session_state.rag_config = get_default_rag_config()
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000/api/v1"
if "selected_document_id" not in st.session_state:
    st.session_state.selected_document_id = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "documents" not in st.session_state:
    st.session_state.documents = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None

# Initialize API client
@st.cache_resource
def get_client():
    return RAGClient(base_url=st.session_state.api_base_url)

client = get_client()

# App title and description
st.title("RAG UI Demo")

# Set the active tab in session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"

# Tab selection buttons
tab_col1, tab_col2, tab_col3 = st.columns(3)
with tab_col1:
    if st.button("Chat", use_container_width=True, type="primary" if st.session_state.active_tab == "Chat" else "secondary"):
        st.session_state.active_tab = "Chat"
        st.rerun()
with tab_col2:
    if st.button("Document Management", use_container_width=True, type="primary" if st.session_state.active_tab == "Document Management" else "secondary"):
        st.session_state.active_tab = "Document Management"
        st.rerun()
with tab_col3:
    if st.button("Configuration", use_container_width=True, type="primary" if st.session_state.active_tab == "Configuration" else "secondary"):
        st.session_state.active_tab = "Configuration"
        st.rerun()

# Sidebar for main configuration
with st.sidebar:
    st.header("Configuration")
    
    # API server settings
    st.subheader("API Connection")
    api_base_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
    if api_base_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_base_url
        st.rerun()
    
    # Get available documents for selection
    try:
        st.session_state.documents = client.list_documents()
    except Exception as e:
        st.error(f"Failed to load documents: {str(e)}")
        st.session_state.documents = []
    
    # Document selection
    st.subheader("Document Selection")
    document_options = [{"label": f"{doc.get('title', 'Untitled')} ({doc['id'][:8]}...)", "value": doc["id"]} 
                       for doc in st.session_state.documents]
    document_options.insert(0, {"label": "All Documents", "value": None})
    
    selected_doc = st.selectbox(
        "Select Document for Chat",
        options=[d["value"] for d in document_options],
        format_func=lambda x: next((d["label"] for d in document_options if d["value"] == x), "All Documents"),
        index=0
    )
    st.session_state.selected_document_id = selected_doc
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat input at top level (only shown when in Chat tab)
show_chat_input = st.session_state.active_tab == "Chat"

# Input for new query (placed at top level as required by Streamlit)
if show_chat_input:
    if query := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get config from session state
        retrieval_enabled = st.session_state.get("retrieval_enabled", True)
        temperature = st.session_state.get("temperature", 0.7)
        top_k = st.session_state.get("top_k", 5)
        model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
        
        # Prepare generation request
        request = {
            "query": query,
            "enable_retrieval": retrieval_enabled,
            "retrieval_top_k": top_k,
            "temperature": temperature,
            "include_sources": True,
            "document_id": st.session_state.selected_document_id,
            "model": model_name
        }
        
        # Store the current query and request for processing after rerun
        st.session_state.current_query = query
        st.session_state.current_request = request
        
        # Rerun to show the message in the UI
        st.rerun()

# Main content based on active tab
if st.session_state.active_tab == "Chat":
    # Chat Interface
    st.header("Chat with your Documents")
    
    # RAG configuration settings for chat
    st.subheader("Chat Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        retrieval_enabled = st.checkbox("Enable Retrieval", value=True)
        st.session_state.retrieval_enabled = retrieval_enabled
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        st.session_state.temperature = temperature
    
    with col2:
        top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
        st.session_state.top_k = top_k
        
        model_name = st.selectbox(
            "Completion Model",
            ["gpt-3.5-turbo", "gpt-4", "claude-2", "local-llama-7b"],
            index=0
        )
        st.session_state.model_name = model_name
        st.session_state.rag_config.completion.model_name = model_name
    
    # Display previous messages
    st.subheader("Conversation")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** - Score: {source['score']:.2f}")
                        st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                        if "metadata" in source and source["metadata"]:
                            with st.expander("Metadata"):
                                st.json(source["metadata"])
                        st.divider()
    
    # Process any pending query
    if "current_query" in st.session_state and "current_request" in st.session_state:
        query = st.session_state.current_query
        request = st.session_state.current_request
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Show spinner while generating
                with st.spinner("Generating response..."):
                    # Get response from API
                    response = client.generate(request)
                
                if "error" in response and response["error"]:
                    message_placeholder.error(f"Error: {response['error']}")
                else:
                    # Display the response
                    message_placeholder.markdown(response["response"])
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": response["response"],
                    }
                    
                    # Add sources if available
                    if "sources" in response and response["sources"]:
                        assistant_message["sources"] = response["sources"]
                        
                        # Display sources in an expander
                        with st.expander("Sources"):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"**Source {i+1}** - Score: {source['score']:.2f}")
                                st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                                if "metadata" in source and source["metadata"]:
                                    with st.expander("Metadata"):
                                        st.json(source["metadata"])
                                st.divider()
                    
                    st.session_state.messages.append(assistant_message)
            
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")
            
            # Clear the current query and request
            del st.session_state.current_query
            del st.session_state.current_request

elif st.session_state.active_tab == "Document Management":
    st.header("Document Management")
    
    # Document Upload Section
    st.subheader("Upload Documents")
    
    # Form for document upload
    with st.form(key="upload_form"):
        # File upload widget
        uploaded_file = st.file_uploader("Choose a document to upload", 
                                         type=["pdf", "txt", "docx", "md", "pptx"], 
                                         help="Supported formats: PDF, TXT, DOCX, MD, PPTX")
        
        # Metadata input
        col1, col2 = st.columns(2)
        with col1:
            doc_title = st.text_input("Document Title", help="A descriptive title for the document")
        with col2:
            doc_source = st.text_input("Source", help="Where this document came from")
        
        # Chunking settings
        st.subheader("Chunking Settings")
        chunking_col1, chunking_col2 = st.columns(2)
        
        with chunking_col1:
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                ["character", "token", "sentence", "paragraph", "recursive"],
                index=0
            )
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000, 100)
        
        with chunking_col2:
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 10)
        
        # Submit button
        submit_button = st.form_submit_button("Upload & Process Document")
    
    # Handle document upload
    if submit_button and uploaded_file is not None:
        try:
            with st.spinner("Uploading and processing document..."):
                # Create a temporary file to save the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    file_path = tmp_file.name
                
                # Prepare metadata
                metadata = {
                    "title": doc_title or uploaded_file.name,
                    "source": doc_source or "UI Upload",
                    "upload_date": datetime.now().isoformat()
                }
                
                # Prepare chunking configuration
                chunking_config = {
                    "chunking_strategy": chunking_strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
                
                # Upload document
                with open(file_path, "rb") as f:
                    result = client.upload_document(f, metadata, chunking_config)
                
                # Clean up the temporary file
                os.unlink(file_path)
                
                # Check result
                if "error" in result and result["error"]:
                    st.error(f"Upload failed: {result['error']}")
                else:
                    st.success(f"Document uploaded successfully! Document ID: {result.get('document_id', 'Unknown')}")
                    st.session_state.upload_status = {
                        "success": True,
                        "message": f"Document uploaded with ID: {result.get('document_id', 'Unknown')}",
                        "document_id": result.get("document_id", "Unknown")
                    }
                    
                    # Refresh document list
                    st.session_state.documents = client.list_documents()
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error during upload: {str(e)}")
            st.session_state.upload_status = {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    # Display existing documents
    st.subheader("Existing Documents")
    
    if st.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = client.list_documents()
            st.rerun()
    
    # Create a table of documents
    if not st.session_state.documents:
        st.info("No documents found. Upload a document to get started!")
    else:
        # Display documents in a dataframe
        doc_data = []
        for doc in st.session_state.documents:
            created_at = doc.get("created_at", "Unknown")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            doc_data.append({
                "ID": doc.get("id", "")[:8] + "...",
                "Title": doc.get("title", "Untitled"),
                "Source": doc.get("source", "Unknown"),
                "Created": created_at,
                "Chunks": doc.get("chunk_count", 0),
            })
        
        if doc_data:
            st.dataframe(doc_data, use_container_width=True)

elif st.session_state.active_tab == "Configuration":
    st.header("RAG Configuration")
    
    # Loading spinner
    with st.spinner("Loading configuration..."):
        # Load current RAG config
        rag_config = st.session_state.rag_config
    
    # Embedding Model Configuration
    st.subheader("Embedding Model")
    
    embedding_col1, embedding_col2 = st.columns(2)
    
    with embedding_col1:
        embedding_model_type = st.selectbox(
            "Embedding Model Type",
            ["sentence_transformers", "openai", "cohere", "local"],
            index=0
        )
        rag_config.embedding.model_type = embedding_model_type
    
    with embedding_col2:
        default_model_name = "sentence-transformers/all-MiniLM-L6-v2" if embedding_model_type == "sentence_transformers" else "text-embedding-ada-002"
        embedding_model_name = st.text_input(
            "Model Name",
            value=rag_config.embedding.model_name or default_model_name
        )
        rag_config.embedding.model_name = embedding_model_name
    
    # Completion Model Configuration
    st.subheader("Completion Model")
    
    completion_col1, completion_col2 = st.columns(2)
    
    with completion_col1:
        completion_model_type = st.selectbox(
            "Completion Model Type",
            ["openai", "anthropic", "local_llama", "local_transformers"],
            index=0
        )
        rag_config.completion.model_type = completion_model_type
    
    with completion_col2:
        default_completion_model = "gpt-3.5-turbo" if completion_model_type == "openai" else "claude-2" if completion_model_type == "anthropic" else "local-model"
        completion_model_name = st.text_input(
            "Model Name",
            value=rag_config.completion.model_name or default_completion_model
        )
        rag_config.completion.model_name = completion_model_name
    
    # Vector Store Configuration
    st.subheader("Vector Store")
    
    vector_col1, vector_col2 = st.columns(2)
    
    with vector_col1:
        vector_store_type = st.selectbox(
            "Vector Store Type",
            ["faiss", "chroma", "pinecone", "weaviate", "in_memory"],
            index=0
        )
        rag_config.vectorstore.store_type = vector_store_type
    
    with vector_col2:
        collection_name = st.text_input(
            "Collection Name",
            value=rag_config.vectorstore.collection_name or "default"
        )
        rag_config.vectorstore.collection_name = collection_name
    
    # Save configuration button
    if st.button("Save Configuration"):
        from config import save_config
        
        try:
            with st.spinner("Saving configuration..."):
                save_result = save_config(rag_config)
                
                if save_result:
                    st.success("Configuration saved successfully!")
                else:
                    st.error("Failed to save configuration")
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")

# Footer
st.markdown("---")
st.caption("Simple RAG UI Demo | Connected to DSP AI RAG API")
