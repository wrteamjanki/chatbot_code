try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    import sqlite3
    from packaging import version
    if version.parse(sqlite3.sqlite_version) < version.parse("3.35.0"):
        raise RuntimeError("Your system sqlite3 version is too old. Please install pysqlite3-binary.")

import os
import streamlit as st
import gemini_api  # Ensures GEMINI_API_KEY is set
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings

# WRTeam Support Details
SUPPORT_NUMBER = "+91-8849493106"
SUPPORT_EMAIL = "wrteam.priyansh@gmail.com"

# Directories
VECTOR_DB_DIR = "vectordb"

# Callback functions
def request_general_response():
    st.session_state.general_response_requested = True

def decline_general_response():
    st.session_state.general_response_requested = False
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": (
            f"For personalized assistance, contact WRTeam support:\n"
            f"{SUPPORT_NUMBER}\n{SUPPORT_EMAIL}"
        )
    })
    st.session_state.pending_question = ""

# Cached Vector Store Setup
@st.cache_resource
def setup_vectorstore():
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# Function to Create Chat Chain
def chat_chain(vectorstore):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=api_key)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,
    )

# Streamlit Page Configuration
st.set_page_config(page_title="WRTeam AI Assistant", page_icon="💬", layout="centered")
st.markdown("# WRTeam AI Assistant")

# Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "general_response_requested" not in st.session_state:
    st.session_state.general_response_requested = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# Display Chat Messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle pending general response requests
if st.session_state.general_response_requested and st.session_state.pending_question:
    try:
        question = st.session_state.pending_question
        general_response = st.session_state.conversational_chain.llm.invoke(
            f"Provide a clear explanation about {question}. "
            "If you're uncertain, give a general overview."
        )
        
        assistant_response = (
            f"Here's a general explanation:\n\n{general_response.content}\n\n"
            f"For personalized assistance, contact WRTeam support:\n"
            f"{SUPPORT_NUMBER}\n{SUPPORT_EMAIL}"
        )
        
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Sorry, I encountered an error. Please try again."
        })
    
    finally:
        st.session_state.general_response_requested = False
        st.session_state.pending_question = ""
        st.rerun()

# User Input Handling
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = st.session_state.conversational_chain.invoke({"question": user_input})
        assistant_response = response.get("answer", "").strip()
        source_documents = response.get("source_documents", [])

        # Check if response is empty or no documents found
        if not assistant_response or not source_documents:
            st.session_state.pending_question = user_input
            st.warning("I couldn't find specific information. Would you like a general explanation?")
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("Yes", on_click=request_general_response)
            with col2:
                st.button("No", on_click=decline_general_response)
            
            st.stop()  # Pause execution until user responds
        
        # Show found response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Sorry, I encountered an error. Please try again."
        })
