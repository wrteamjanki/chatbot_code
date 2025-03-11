# Override sqlite3 with pysqlite3-binary
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    import sqlite3
    from packaging import version
    if version.parse(sqlite3.sqlite_version) < version.parse("3.35.0"):
        raise RuntimeError(
            "Your system sqlite3 version is too old. Please install pysqlite3-binary."
        )
import os
import json
import pickle
import streamlit as st
import subprocess

# Instead of ChatGroq, we'll use ChatOpenAI from LangChain
from langchain.chat_models import ChatOpenAI  
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from vectorized_documents import embeddings

# --------------------------
# Setup Deepseek API Environment
# --------------------------
# Set the base URL for the OpenRouter (Deepseek) API
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Load your Deepseek API key from secrets.toml (assuming it is stored there)
# If your secrets.toml is in the .streamlit folder, adjust the path accordingly:
import toml
secrets = toml.load(".streamlit/secrets.toml")
# For Deepseek, ensure you have the key stored under a proper name
deepseek_api_key = secrets["general"]["OPENROUTER_API_KEY"]

# Set the API key as an environment variable for OpenAI's client to use
os.environ["OPENAI_API_KEY"] = deepseek_api_key

# --------------------------
# Directories
# --------------------------
DATA_DIR = "data"
VECTOR_DB_DIR = "vectordb"
HISTORY_FILE = "chat_history.pkl"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------
# Cached Vector Store Setup
# --------------------------
@st.cache_resource
def setup_vectorstore():
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# --------------------------
# Function to Create Chat Chain using Deepseek (via ChatOpenAI)
# --------------------------
def chat_chain(vectorstore):
    # Create a ChatOpenAI instance pointing to the Deepseek model
    llm = ChatOpenAI(model_name="deepseek/deepseek-r1-zero:free", temperature=0)
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

# --------------------------
# Streamlit Page Configuration
# --------------------------
st.set_page_config(page_title="AI ASSISTANT", page_icon="💬", layout="centered")

# Custom CSS for Minimalist Design
st.markdown(
    """
    <style>
    .stChatInput > div > div > textarea:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 5px #4A90E2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Title
st.markdown("# AI ASSISTANT")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Do not load previous history on rerun

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display Chat History within the session
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.chat_message("assistant"):
            response = st.session_state.conversational_chain.invoke(
                {"question": user_input, "chat_history": st.session_state.chat_history}
            )
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )
    except Exception as e:
        st.error(f"Error: {e}")
