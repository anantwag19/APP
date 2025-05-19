import streamlit as st
import requests
import json
from io import BytesIO
from typing import Optional

# =================================================================
# 1. CONFIGURATION (Update with your ngrok URL)
# =================================================================
API_BASE_URL = "https://8254-27-4-56-158.ngrok-free.app"  # Replace with your ngrok URL
ENDPOINTS = {
    "query": "/query",
    "query_stream": "/query_stream",
    "upload": "/upload",
    "ingestion": "/ingestion"
}

# =================================================================
# 2. API HELPER FUNCTIONS
# =================================================================
def call_api(endpoint: str, payload: Optional[dict] = None, files: Optional[dict] = None) -> dict:
    """Generic API call function with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if files:
            response = requests.post(url, files=files)
        else:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers, stream=endpoint == "query_stream")
        
        response.raise_for_status()
        return response.json() if not endpoint == "query_stream" else response
    
    except requests.exceptions.RequestException as e:
        return {"error": f"API Error: {str(e)}"}

# =================================================================
# 3. STREAMLIT UI COMPONENTS
# =================================================================
def display_chat_message(role: str, content: str):
    """Displays a chat message with role-based formatting."""
    with st.chat_message(role):
        st.markdown(content)

def handle_streaming_response(query: str):
    """Handles real-time streaming from the /query_stream endpoint."""
    display_chat_message("user", query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream_url = f"{API_BASE_URL}{ENDPOINTS['query_stream']}"
            with requests.post(stream_url, json={'query': query}, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            chunk = json.loads(decoded_line[5:].strip())
                            if chunk.get("type") == "chunk":
                                full_response += chunk["text"]
                                message_placeholder.markdown(full_response + "▌")
                            elif chunk.get("type") == "eval":
                                st.divider()
                                st.subheader("Evaluation Metrics")
                                metrics = chunk.get("metrics", {})
                                if metrics:
                                    st.json(metrics)
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Streaming error: {str(e)}")

def handle_regular_query(query: str):
    """Handles non-streaming queries."""
    display_chat_message("user", query)
    
    with st.spinner("Thinking..."):
        response = call_api(ENDPOINTS["query"], payload={'query': query})
        
        if "error" in response:
            st.error(response["error"])
        else:
            display_chat_message("assistant", response.get("answer", "No answer found."))
            
            # Show evaluation metrics if available
            if "evaluation" in response:
                st.divider()
                st.subheader("Evaluation Metrics")
                st.json(response["evaluation"])

def handle_file_upload():
    """Handles PDF upload and ingestion pipeline."""
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Uploading..."):
            files = {'file': (uploaded_file.name, uploaded_file, "application/pdf")}
            upload_response = call_api(ENDPOINTS["upload"], files=files)
        
        if "error" in upload_response:
            st.error(upload_response["error"])
        else:
            st.success("File uploaded successfully!")
            
            # Start ingestion
            with st.spinner("Processing document..."):
                ingestion_response = call_api(
                    ENDPOINTS["ingestion"],
                    payload={'filename': upload_response.get("filename")}
                )
            
            if "error" in ingestion_response:
                st.error(ingestion_response["error"])
            else:
                st.success(f"Processed {ingestion_response.get('chunks', 0)} chunks!")

# =================================================================
# 4. MAIN APP LAYOUT
# =================================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    stream_mode = st.toggle("Streaming Mode", value=True)
    st.divider()
    handle_file_upload()

# Main chat interface
st.title("📚 RAG Chatbot with Evaluation")
st.caption("Ask questions about your uploaded documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if stream_mode:
        handle_streaming_response(prompt)
    else:
        handle_regular_query(prompt)