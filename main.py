import os
import json
import pickle
import streamlit as st
import subprocess
import sys

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings

# Fix for ChromaDB's SQLite dependency issue
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    st.warning("pysqlite3 not found. Ensure 'pysqlite3-binary' is installed.")

# Directories
DATA_DIR = "data"
VECTORIZE_SCRIPT = "vectorized_documents.py"
VECTOR_DB_DIR = "vectordb"
HISTORY_FILE = "chat_history.pkl"

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Load API Key from config file
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_data = json.load(f)
        os.environ["GROQ_API_KEY"] = config_data.get("GROQ_API_KEY", "")

# Load Chat History
def load_chat_history():
    try:
        with open(HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, FileNotFoundError):
        return []

# Save Chat History
def save_chat_history():
    temp_file = HISTORY_FILE + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(st.session_state.chat_history, f)
    os.replace(temp_file, HISTORY_FILE)

# Cached Vector Store Setup
@st.cache_resource
def setup_vectorstore():
    try:
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None

# List Available Documents
def list_documents():
    return [
        f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))
    ]

# Run the Vectorization Script
def update_vector_db():
    with st.spinner("Vectorizing documents... Please wait."):
        result = subprocess.run(
            ["python", VECTORIZE_SCRIPT], capture_output=True, text=True
        )
        st.success("Vectorization Completed!")
        st.text(result.stdout)
    st.session_state.vectorstore = setup_vectorstore()

# Delete Documents
def delete_documents(file_names):
    for file_name in file_names:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            st.sidebar.success(f"Deleted {file_name}")
        else:
            st.sidebar.error(f"File {file_name} not found.")
    update_vector_db()

# Create Chat Chain
def chat_chain(vectorstore):
    if not vectorstore:
        st.error("Vectorstore is not initialized! Please check your setup.")
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
st.set_page_config(page_title="Multi-Doc RAG Chatbot", layout="wide")

# Sidebar for Document Management
st.sidebar.title("Available Documents")
documents = list_documents()
selected_docs = st.sidebar.multiselect("Select Documents", documents, default=documents)

# Upload New Document
uploaded_file = st.sidebar.file_uploader("Upload a Document", type=["pdf", "json"])
if uploaded_file:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")
    st.rerun()

# Button to Trigger Vectorization
if st.sidebar.button("Update Vector DB"):
    update_vector_db()

# Delete Documents
if documents:
    files_to_delete = st.sidebar.multiselect("Select Documents to Delete", documents)
    if st.sidebar.button("Delete Selected Documents"):
        delete_documents(files_to_delete)
        st.rerun()

# Custom CSS for Styling
st.markdown(
    """
    <style>
    .stChatInput > div > div > textarea:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 5px #4A90E2 !important;
    }
    .title-button {
        display: block;
        width: 100%;
        padding: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: white;
        background: linear-gradient(135deg, #4A90E2, #6B56E2);
        border-radius: 10px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
    }
    .title-button:hover {
        background: linear-gradient(135deg, #6B56E2, #4A90E2);
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Title Button
st.markdown('<button class="title-button">E-LMS RAG CHATBOT</button>', unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state or "vectorstore" in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
user_input = st.chat_input("I am your AI assistant, Ask me anything ...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if st.session_state.conversational_chain:
            response = st.session_state.conversational_chain.invoke(
                {"question": user_input, "chat_history": st.session_state.chat_history}
            )
            assistant_response = response.get("answer", "I couldn't understand that.")
        else:
            assistant_response = "Error: Chatbot is not properly initialized."

        st.markdown(assistant_response)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_response}
        )
        save_chat_history()
