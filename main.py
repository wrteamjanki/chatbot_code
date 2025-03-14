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
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from vectorized_documents import embeddings

# WRTeam Support Details
SUPPORT_NUMBER = "1234567890"
SUPPORT_EMAIL = "support@wrteam.com"

# Directories
DATA_DIR = "data"
VECTOR_DB_DIR = "vectordb"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Cached Vector Store Setup
@st.cache_resource
def setup_vectorstore():
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# Function to clean text
def clean_text(text):
    return text.encode("utf-16", "surrogatepass").decode("utf-16")

# Function to Create Chat Chain
def chat_chain(vectorstore):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    model = genai.GenerativeModel("gemini-1.5-flash-002")
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Custom chain for WRTeam Assistant
    def custom_chain(question, chat_history):
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history]
        )

        system_prompt = (
            "You are WRTeam's official AI assistant. You answer queries on behalf of WRTeam "
            "using the company's knowledge base. If the question is about WRTeam but you don't have an answer, "
            f"then suggest the user contact WRTeam support at {SUPPORT_NUMBER} or email {SUPPORT_EMAIL}. "
            "However, if the question is general and not related to WRTeam, do not redirect to support. "
            "Instead, respond naturally or say you don't have enough information."
        )

        prompt = f"{system_prompt}\n{history_text}\nUser: {question}"
        
        response = model.generate_content([prompt])
        cleaned_response = clean_text(response.text)

        # Detect if the model gives an uncertain response
        if "I don’t know" in cleaned_response or not cleaned_response.strip():
            # Check if the question is about WRTeam
            wrteam_keywords = ["wrteam", "your company", "your product", "support", "customer service"]
            if any(keyword in question.lower() for keyword in wrteam_keywords):
                return {
                    "answer": (
                        f"I'm sorry, but I couldn't find relevant information for your WRTeam-related query. "
                        f"Please contact WRTeam support at **{SUPPORT_NUMBER}** or email **{SUPPORT_EMAIL}** for assistance."
                    )
                }
            else:
                return {"answer": "I'm not sure about that. You might want to check online for more details!"}

        return {"answer": cleaned_response}

    return custom_chain

# Streamlit Page Configuration
st.set_page_config(page_title="WRTeam AI Assistant", page_icon="💬", layout="centered")

# Custom CSS for Dark Theme
st.markdown(
    """
    <style>
    body { background-color: #1E1E1E; color: white; }
    .stChatInput > div > div > textarea:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 5px #4A90E2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display Title
st.markdown("# 🤖 WRTeam AI Assistant")

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
            response = st.session_state.conversational_chain(
                user_input, st.session_state.chat_history
            )
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
