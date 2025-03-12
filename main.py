import gemini_api
import os
import json
import pickle
import streamlit as st
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings
import google.generativeai as genai

# Directories
DATA_DIR = "data"
VECTOR_DB_DIR = "vectordb"
HISTORY_FILE = "chat_history.pkl"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Cached Vector Store Setup
@st.cache_resource
def setup_vectorstore():
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# Function to clean text and handle Unicode errors
def clean_text(text):
    return text.encode("utf-16", "surrogatepass").decode("utf-16")

# Function to Create Chat Chain
def chat_chain(vectorstore):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    
    model = genai.GenerativeModel('gemini-1.5-flash-002')
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    
    # Custom chain for Gemini API
    def custom_chain(question, chat_history):
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        prompt = f"{history_text}\nUser: {question}"
        
        response = model.generate_content([prompt])
        cleaned_response = clean_text(response.text)
        return {"answer": cleaned_response}
    
    return custom_chain

# Streamlit Page Configuration
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
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display Chat History
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
            response = st.session_state.conversational_chain(user_input, st.session_state.chat_history)
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")