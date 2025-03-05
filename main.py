import os
import sys
import json
import pickle
import streamlit as st
import subprocess
import dotenv

from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings

# Load environment variables
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Force pysqlite3 for ChromaDB (only if needed)
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Directories
DATA_DIR = "data"
VECTORIZE_SCRIPT = "vectorized_documents.py"
VECTOR_DB_DIR = "vectordb"
HISTORY_FILE = "chat_history.pkl"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Function to Load Chat History
def load_chat_history():
    try:
        with open(HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, FileNotFoundError):
        return []

# Function to Save Chat History
def save_chat_history():
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(st.session_state.chat_history, f)

# Cached Vector Store Setup
@st.cache_resource
def setup_vectorstore():
    try:
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None

# Function to List Available Documents
def list_documents():
    return [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]

# Function to Run the Vectorization Script
def update_vector_db():
    with st.spinner("Vectorizing documents... Please wait."):
        result = subprocess.run(["python", VECTORIZE_SCRIPT], capture_output=True, text=True)
        st.success("Vectorization Completed!")
        st.text(result.stdout)
    st.session_state.vectorstore = setup_vectorstore()

# Function to Delete Documents
def delete_documents(file_names):
    for file_name in file_names:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            st.sidebar.success(f"Deleted {file_name}")
    update_vector_db()

# Function to Create Chat Chain
def chat_chain(vectorstore):
    if not vectorstore:
        st.error("Vector store not initialized!")
        return None

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever()

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True,
    )

# Streamlit Page Configuration
st.set_page_config(page_title="Multi-Doc RAG Chatbot", page_icon="📚", layout="wide")

# Sidebar for Document Management
st.sidebar.title("Available Documents")
documents = list_documents()
selected_docs = st.sidebar.multiselect("Select Documents", documents, default=documents)

# Upload New Document
uploaded_file = st.sidebar.file_uploader("Upload a Document", type=["pdf", "json"])
if uploaded_file:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.ge
