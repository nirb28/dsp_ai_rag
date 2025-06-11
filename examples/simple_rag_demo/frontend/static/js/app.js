// RAG Demo Application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('uploadForm');
    const documentFile = document.getElementById('documentFile');
    const documentType = document.getElementById('documentType');
    const documentList = document.getElementById('documentList');
    const refreshDocumentsBtn = document.getElementById('refreshDocuments');
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const chatMessages = document.getElementById('chatMessages');
    const documentFilters = document.getElementById('documentFilters');
    const saveConfigBtn = document.getElementById('saveConfig');
    const deleteDocumentBtn = document.getElementById('deleteDocument');
    
    // Query options
    const enableRetrieval = document.getElementById('enableRetrieval');
    const includeSources = document.getElementById('includeSources');
    const topK = document.getElementById('topK');
    const topKValue = document.getElementById('topKValue');
    
    // Configuration options
    const chunkingStrategy = document.getElementById('chunkingStrategy');
    const chunkSize = document.getElementById('chunkSize');
    const chunkOverlap = document.getElementById('chunkOverlap');
    const embeddingModel = document.getElementById('embeddingModel');
    const completionModel = document.getElementById('completionModel');
    const enableReranking = document.getElementById('enableReranking');
    const temperature = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperatureValue');
    const maxTokens = document.getElementById('maxTokens');
    
    // State
    let currentDocuments = [];
    let selectedDocuments = [];
    let currentConfig = {};
    let currentDocumentId = null;
    
    // Initialize
    loadConfig();
    loadDocuments();
    
    // Event Listeners
    topK.addEventListener('input', function() {
        topKValue.textContent = this.value;
    });
    
    temperature.addEventListener('input', function() {
        temperatureValue.textContent = this.value;
    });
    
    refreshDocumentsBtn.addEventListener('click', loadDocuments);
    
    // Upload document form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!documentFile.files.length) {
            showError('Please select a file to upload');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', documentFile.files[0]);
        formData.append('document_type', documentType.value);
        
        // Show uploading message
        addSystemMessage('Uploading document...');
        
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            addSystemMessage(`Document uploaded successfully! Processing in background.`);
            documentFile.value = '';
            loadDocuments();
        })
        .catch(error => {
            console.error('Error:', error);
            addSystemMessage(`Error uploading document: ${error.message}`, true);
        });
    });
    
    // Query form submission
    queryForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) {
            return;
        }
        
        // Add user message
        addUserMessage(query);
        
        // Clear input
        queryInput.value = '';
        
        // Add loading indicator
        const loadingId = addLoadingMessage();
        
        // Prepare query options
        const queryOptions = {
            query: query,
            enable_retrieval: enableRetrieval.checked,
            include_sources: includeSources.checked,
            top_k: parseInt(topK.value),
            document_ids: selectedDocuments.length > 0 ? selectedDocuments : null
        };
        
        // Use streaming endpoint for better UX
        streamQuery(queryOptions, loadingId);
    });
    
    // Save configuration
    saveConfigBtn.addEventListener('click', function() {
        const configUpdate = {
            chunking_strategy: chunkingStrategy.value,
            chunk_size: parseInt(chunkSize.value),
            chunk_overlap: parseInt(chunkOverlap.value),
            embedding_model: embeddingModel.value,
            completion_model: completionModel.value,
            temperature: parseFloat(temperature.value),
            enable_reranking: enableReranking.checked
        };
        
        fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(configUpdate)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('configModal'));
            modal.hide();
            
            // Show success message
            addSystemMessage('Configuration updated successfully');
        })
        .catch(error => {
            console.error('Error:', error);
            addSystemMessage(`Error updating configuration: ${error.message}`, true);
        });
    });
    
    // Handle document deletion
    deleteDocumentBtn.addEventListener('click', function() {
        if (!currentDocumentId) return;
        
        fetch(`/api/documents/${currentDocumentId}`, {
            method: 'DELETE'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('documentDetailModal'));
            modal.hide();
            
            // Show success message
            addSystemMessage(`Document deleted successfully`);
            
            // Refresh document list
            loadDocuments();
        })
        .catch(error => {
            console.error('Error:', error);
            addSystemMessage(`Error deleting document: ${error.message}`, true);
        });
    });
}); // End of DOMContentLoaded
