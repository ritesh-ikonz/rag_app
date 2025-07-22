import streamlit as st
import requests
import json
from typing import List, Dict
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Client RAG Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_clients():
    """Get list of available clients"""
    try:
        response = requests.get(f"{API_BASE_URL}/clients")
        if response.status_code == 200:
            return response.json().get("clients", [])
        return []
    except:
        return []

def upload_document(client_id: str, file):
    """Upload a document for a specific client"""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        data = {"client_id": client_id}
        response = requests.post(f"{API_BASE_URL}/ingest", files=files, data=data)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def ask_question(client_id: str, question: str):
    """Ask a question to the agent"""
    try:
        payload = {"question": question, "client_id": client_id}
        response = requests.post(f"{API_BASE_URL}/ask", json=payload)
        if response.status_code == 200:
            return True, response.json().get("answer", "")
        else:
            return False, response.json().get("detail", "Unknown error")
    except Exception as e:
        return False, str(e)

def delete_client_kb(client_id: str):
    """Delete a client's knowledge base"""
    try:
        response = requests.delete(f"{API_BASE_URL}/clients/{client_id}")
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "current_client" not in st.session_state:
    st.session_state.current_client = None

# Main app
def main():
    st.markdown('<h1 class="main-header">🤖 Multi-Client RAG Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the FastAPI server at http://localhost:8000")
        st.stop()
    
    # Sidebar for navigation and client management
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox("Select Page", ["Document Upload", "Chat Interface", "Client Management"])
        
        st.header("👥 Available Clients")
        clients = get_clients()
        if clients:
            for client in clients:
                st.write(f"• {client}")
        else:
            st.write("No clients available")
    
    # Main content based on selected page
    if page == "Document Upload":
        document_upload_page()
    elif page == "Chat Interface":
        chat_interface_page()
    elif page == "Client Management":
        client_management_page()

def document_upload_page():
    st.header("📄 Document Upload")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Document")
        
        # Client ID input
        client_id = st.text_input("Client ID", placeholder="Enter client identifier (e.g., client1, company_a)")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if st.button("Upload Document", disabled=not (client_id and uploaded_file)):
            if client_id and uploaded_file:
                with st.spinner("Uploading and processing document..."):
                    success, result = upload_document(client_id, uploaded_file)
                    
                if success:
                    st.markdown(f'<div class="success-box">✅ Document uploaded successfully for client: {client_id}</div>', unsafe_allow_html=True)
                    st.success(f"Collection: {result.get('collection_name', 'N/A')}")
                    # Refresh the page to update client list
                    st.rerun()
                else:
                    st.markdown(f'<div class="error-box">❌ Error: {result.get("detail", "Unknown error")}</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Upload Instructions")
        st.info("""
        **How to upload documents:**
        
        1. Enter a unique Client ID (e.g., "client1", "company_a")
        2. Select a PDF file to upload
        3. Click "Upload Document"
        4. The system will process and index the document
        5. You can then chat with the document using the Chat Interface
        
        **Note:** Each client has their own separate knowledge base.
        """)

def chat_interface_page():
    st.header("💬 Chat Interface")
    
    clients = get_clients()
    if not clients:
        st.warning("No clients available. Please upload documents first.")
        return
    
    # Client selection
    selected_client = st.selectbox("Select Client", clients, key="chat_client_select")
    
    if selected_client:
        st.session_state.current_client = selected_client
        
        # Initialize chat history for this client if not exists
        if selected_client not in st.session_state.messages:
            st.session_state.messages[selected_client] = []
        
        # Display chat history
        st.subheader(f"Chat with {selected_client}")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages[selected_client]:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        with st.form(key=f"chat_form_{selected_client}", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input("Ask a question about the documents:", key=f"user_input_{selected_client}")
            with col2:
                submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.messages[selected_client].append({"role": "user", "content": user_input})
            
            # Get response from agent
            with st.spinner("Thinking..."):
                success, response = ask_question(selected_client, user_input)
            
            if success:
                st.session_state.messages[selected_client].append({"role": "assistant", "content": response})
            else:
                st.session_state.messages[selected_client].append({"role": "assistant", "content": f"Error: {response}"})
            
            # Refresh to show new messages
            st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages[selected_client] = []
            st.rerun()

def client_management_page():
    st.header("👥 Client Management")
    
    clients = get_clients()
    
    if clients:
        st.subheader("Current Clients")
        
        for client in clients:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{client}**")
            
            with col2:
                if st.button(f"Chat", key=f"chat_{client}"):
                    st.session_state.current_client = client
                    st.switch_page("Chat Interface")
            
            with col3:
                if st.button(f"Delete", key=f"delete_{client}", type="secondary"):
                    if st.session_state.get(f"confirm_delete_{client}", False):
                        success, result = delete_client_kb(client)
                        if success:
                            st.success(f"Knowledge base for {client} deleted successfully")
                            if client in st.session_state.messages:
                                del st.session_state.messages[client]
                            st.rerun()
                        else:
                            st.error(f"Error deleting {client}: {result.get('detail', 'Unknown error')}")
                    else:
                        st.session_state[f"confirm_delete_{client}"] = True
                        st.warning(f"Click Delete again to confirm deletion of {client}")
                        st.rerun()
    else:
        st.info("No clients available. Upload documents to create clients.")
    
    # Add some statistics
    st.subheader("Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Clients", len(clients))
    
    with col2:
        total_messages = sum(len(messages) for messages in st.session_state.messages.values())
        st.metric("Total Messages", total_messages)

if __name__ == "__main__":
    main()