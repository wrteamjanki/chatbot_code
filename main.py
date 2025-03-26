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
from typing import List, Dict, Callable

# Get API key from streamlit secrets.toml
if 'GEMINI_API_KEY' not in st.secrets:
    st.error('GEMINI_API_KEY not found or secrets.toml is missing. Please check your secrets.toml file.')
    st.stop()

# Use the API key from secrets
api_key = st.secrets['GEMINI_API_KEY']

# WRTeam Support Details
SUPPORT_NUMBER: str = "+91 8849493106"
SUPPORT_EMAIL: str = "support@wrteam.in"

# Directories
DATA_DIR: str = "data"
VECTOR_DB_DIR: str = "vectordb"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Cached Vector Store Setup
@st.cache_resource
def setup_vectorstore() -> Chroma:
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# Function to clean text
def clean_text(text: str) -> str:
    return text.encode("utf-16", "surrogatepass").decode("utf-16")

# Function to Create Chat Chain
def chat_chain(vectorstore: Chroma) -> Callable[[str, List[Dict[str, str]]], Dict[str, str]]:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-002")
    except Exception as e:
        st.error(f"API key error: {e}")
        return lambda question, chat_history: {"answer": "API key error. Please check your API key."}

    # Custom chain for WRTeam Assistant
    def custom_chain(question: str, chat_history: List[Dict[str, str]]) -> Dict[str, str]:
        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history]
        )

        retriever = st.session_state.vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)

        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        system_prompt = f"""
            You are an AI assistant for WRTeam's eSchoolSaaS, responsible for answering user queries based on the provided data.
            Use the provided context to answer the questions.
            Prioritize accuracy and provide detailed, relevant, and well-structured responses. Use bullet points, numbered lists, or tables when appropriate.

            If a query is related to eSchoolSaaS and you have relevant data in the context, provide a detailed, accurate, and well-structured response.
            If the query is about eSchoolSaaS but the answer is not in the context, inform the user and suggest contacting WRTeam support at +91 8849491306 or emailing support@wrteam.in.
            If the query is unrelated to WRTeam, do not redirect to support. Instead, respond naturally with as much relevant detail as possible based on general knowledge,
            maintaining a helpful and professional tone, or state that you lack sufficient information.

            When providing answers, please consider the user's role, and give role-specific information.

            Example 1:
            User: 'What are the available user roles?'
            Assistant: 'The available user roles are Super Admin, School Admin, Teacher, Staff, Parent, and Student. This is the detailed info of them...'

            Example 2:
            User: 'How do I reset my password?'
            Assistant: 'I do not have that information. Please contact WRTeam support at +91 8849491306 or email support@wrteam.in.'

            Context:
            {context}
            """
        prompt = f"{system_prompt}\n{history_text}\nUser: {question}"

        response = model.generate_content([prompt])
        cleaned_response = clean_text(response.text)

        # Detect if the model gives an uncertain response
        if "I don't know" in cleaned_response.lower() or not cleaned_response.strip():
            # Check if the question is about WRTeam
            wrteam_keywords = ["wrteam", "eschoolsaas", "your product", "support", "customer service"]
            if any(keyword in question.lower() for keyword in wrteam_keywords):
                return {
                    "answer": (
                        f"I'm sorry, but I couldn't find relevant information for your WRTeam-related query. "
                        f"Please contact WRTeam support at +91 8849491306 or email support@wrteam.in for assistance."
                    )
                }
            else:
                return {"answer": "I'm not sure about that. You might want to check online for more details!"}

        return {"answer": cleaned_response}

    return custom_chain

# Streamlit Page Configuration
st.set_page_config(page_title="WRTeam AI Assistant chatbot", page_icon="💬", layout="centered")

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
st.markdown("# 👻 WRTeam AI Assistant")

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