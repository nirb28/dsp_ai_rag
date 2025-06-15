#!/usr/bin/env python
"""
Entry point script to run the Streamlit UI for DSP AI RAG
"""
import os
import sys
import subprocess

def main():
    """Main function to run the Streamlit UI"""
    # Add the parent directory to the path so we can import from app
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    print("Starting Streamlit UI for DSP AI RAG...")
    print("API URL: http://localhost:8000/api/v1 (default)")
    
    # Run the Streamlit app
    streamlit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    subprocess.run(["streamlit", "run", streamlit_path], check=True)

if __name__ == "__main__":
    main()
