# RAG UI Demo with Streamlit

This is a simple user interface built with Streamlit for interacting with the DSP AI RAG API.

## Features

- Simple chat interface for asking questions
- Document upload and chunking capabilities
- Document management with configurable chunking parameters
- Document selection for targeted retrieval
- Configuration options for model parameters
- Display of retrieved sources with relevance scores
- Easy configuration of API connection

## Prerequisites

- Python 3.8+
- The DSP AI RAG API running and accessible
- Required Python packages (install via `pip install -r requirements.txt`)

## Getting Started

1. Make sure the DSP AI RAG API is running (typically on http://localhost:8000)
2. Run the Streamlit app:

```bash
cd /path/to/dsp_ai_rag
streamlit run streamlit_ui/app.py
```

3. Open your browser to the URL displayed in the terminal (typically http://localhost:8501)

## Configuration

The app comes with default settings, but you can customize:

- API Base URL: Connect to a different API instance
- Completion Model: Choose different LLM models
- Retrieval Settings: Turn retrieval on/off and set the number of chunks
- Document Selection: Filter questions to specific documents
- Temperature: Adjust the randomness of responses

## Directory Structure

```
streamlit_ui/
├── app.py        # Main Streamlit application
├── api_client.py # Client for RAG API interaction
├── config.py     # Configuration handling
└── config/       # Configuration storage
```

## Notes

- The UI will automatically connect to the API at http://localhost:8000/api/v1 by default
- Make sure API keys are correctly configured in the RAG API if using models that require them
- Document upload functionality will be added in a future version
