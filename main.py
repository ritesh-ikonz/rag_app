# main.py

import uvicorn
import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

# LangChain message types
from langchain_core.messages import HumanMessage

# Import agent functions from your agent file
from agent import create_rag_agent, ingest_pdf_for_client, chroma_client

# Load environment variables
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multi-Client RAG Agent API",
    description="An API for ingesting documents and querying a RAG agent against client-specific knowledge bases.",
    version="2.0.0"
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    client_id: str

class QueryResponse(BaseModel):
    answer: str

class IngestResponse(BaseModel):
    message: str
    client_id: str
    collection_name: str

class ClientListResponse(BaseModel):
    clients: List[str]

# --- Load the RAG Agent ---
rag_agent = create_rag_agent()
print("RAG Agent compiled and ready.")

# --- API Endpoints ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "Multi-Client RAG Agent API is running"}

# --- Health check endpoint ---
@app.get("/health", tags=["Status"])
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "message": "API is running successfully"}

# --- Endpoint to list available clients/knowledge bases ---
@app.get("/clients", response_model=ClientListResponse, tags=["Knowledge Base Management"])
def list_clients():
    """Lists all available client knowledge bases."""
    try:
        collections = chroma_client.list_collections()
        # Our convention is collection_name = f"kb_{client_id}"
        client_ids = [col.name.replace("kb_", "", 1) for col in collections if col.name.startswith("kb_")]
        return ClientListResponse(clients=client_ids)
    except Exception as e:
        print(f"Error listing clients: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list clients: {str(e)}")

# --- Endpoint to ingest a PDF for a client ---
@app.post("/ingest", response_model=IngestResponse, tags=["Knowledge Base Management"])
async def ingest_document(
    client_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Uploads a PDF and ingests its content into a specific client's knowledge base.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate client_id
    if not client_id or not client_id.strip():
        raise HTTPException(status_code=400, detail="Client ID cannot be empty")
    
    # Sanitize client_id to prevent path traversal
    client_id = client_id.strip().replace('/', '_').replace('\\', '_')
    
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{client_id}_{file.filename}")

    try:
        # Save the uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call the ingestion logic
        ingest_pdf_for_client(client_id, temp_path)
        
        return IngestResponse(
            message="Successfully ingested document.",
            client_id=client_id,
            collection_name=f"kb_{client_id}"
        )
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        raise HTTPException(status_code=400, detail=f"File not found: {str(e)}")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- Endpoint to ask a question to a specific client's agent ---
@app.post("/ask", response_model=QueryResponse, tags=["Agent"])
async def ask_agent(request: QueryRequest):
    """
    Receives a question and client_id, passes it to the RAG agent, and returns the answer.
    """
    print(f"Received question for client '{request.client_id}': {request.question}")
    
    # Validate inputs
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not request.client_id.strip():
        raise HTTPException(status_code=400, detail="Client ID cannot be empty")
    
    # Check if the client's knowledge base exists
    try:
        collections = [col.name for col in chroma_client.list_collections()]
        if f"kb_{request.client_id}" not in collections:
            raise HTTPException(status_code=404, detail=f"Knowledge base for client '{request.client_id}' not found. Please ingest documents first.")
    except Exception as e:
        print(f"Error checking collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to check knowledge base availability")

    # Prepare the input for the agent, now including the client_id
    messages = [HumanMessage(content=request.question)]
    initial_state = {"messages": messages, "client_id": request.client_id}
    
    try:
        result = rag_agent.invoke(initial_state)
        final_answer = result['messages'][-1].content
        print(f"Agent's final answer: {final_answer}")
        return QueryResponse(answer=final_answer)
    except Exception as e:
        print(f"An error occurred during agent invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process the request: {str(e)}")


# --- Endpoint to delete a client's knowledge base ---
@app.delete("/clients/{client_id}", tags=["Knowledge Base Management"])
async def delete_client_kb(client_id: str):
    """
    Deletes a client's knowledge base collection.
    """
    try:
        collection_name = f"kb_{client_id}"
        chroma_client.delete_collection(collection_name)
        return {"message": f"Knowledge base for client '{client_id}' deleted successfully"}
    except Exception as e:
        print(f"Error deleting client KB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge base: {str(e)}")


# --- Uvicorn Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)