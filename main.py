# Check for required packages before proceeding
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    import sqlite3
    try:
        from packaging import version
    except ImportError:
        raise ImportError("Missing required 'packaging' package. Install with: pip install packaging")
    
    if version.parse(sqlite3.sqlite_version) < version.parse("3.35.0"):
        raise RuntimeError(f"System sqlite3 version {sqlite3.sqlite_version} is too old. Install pysqlite3-binary with: pip install pysqlite3-binary")

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings

# Configuration constants
CONFIG = {
    "SUPPORT_NUMBER": "+91-8849493106",
    "SUPPORT_EMAIL": "wrteam.priyansh@gmail.com",
    "VECTOR_DB_DIR": "vectordb",
    "MODEL_NAME": "gemini-1.5-flash-002",
    "REQUIRED_SQLITE_VERSION": "3.35.0"
}

# Validate environment variables
if "GEMINI_API_KEY" not in os.environ:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set. Ensure it's set in gemini_api.py or your environment.")

# VectorDB initialization with error handling
@st.cache_resource(show_spinner="Initializing knowledge base...")
def setup_vectorstore():
    if not os.path.exists(CONFIG["VECTOR_DB_DIR"]):
        raise FileNotFoundError(f"Vector database directory {CONFIG['VECTOR_DB_DIR']} not found")
    
    try:
        return Chroma(
            persist_directory=CONFIG["VECTOR_DB_DIR"],
            embedding_function=embeddings
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chroma vector store: {str(e)}")

# System initialization
def initialize_system():
    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = setup_vectorstore()
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

    if "conversational_chain" not in st.session_state:
        try:
            llm = ChatGoogleGenerativeAI(
                model=CONFIG["MODEL_NAME"],
                google_api_key=os.environ["GEMINI_API_KEY"],
                temperature=0.3
            )
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            st.session_state.conversational_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            st.error(f"Failed to create conversation chain: {str(e)}")
            st.stop()

    # Initialize session states for conversation flow
    session_defaults = {
        "chat_history": [],
        "pending_question": None,
        "show_general_prompt": False
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

# Streamlit UI configuration
def configure_ui():
    st.set_page_config(
        page_title="WRTeam AI Assistant",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.markdown("# WRTeam AI Assistant")
    st.caption("Powered by Gemini 1.5 Flash and ChromaDB")

# Chat history management
def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Response generation flow
def handle_user_query(user_input: str):
    try:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process query through conversation chain
        with st.spinner("Analyzing your query..."):
            response = st.session_state.conversational_chain.invoke({"question": user_input})
        
        # Handle response based on source documents
        if not response["source_documents"]:
            st.session_state.pending_question = user_input
            st.session_state.show_general_prompt = True
        else:
            display_assistant_response(response["answer"])

    except Exception as e:
        st.error(f"Error processing your request: {str(e)}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Sorry, I encountered an error. Please try again or contact support."
        })

# Response display handlers
def display_assistant_response(response_text: str):
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

def display_support_prompt():
    support_message = (
        "I understand! For personalized assistance, contact our support team:\n\n"
        f"📞 {CONFIG['SUPPORT_NUMBER']}  \n"
        f"📧 {CONFIG['SUPPORT_EMAIL']}\n\n"
        "We're here to help! 😊"
    )
    display_assistant_response(support_message)
    st.session_state.show_general_prompt = False

def handle_general_response_request():
    with st.chat_message("assistant"):
        st.warning("I couldn't find specific documentation. Would you like a general answer?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, please", key="general_yes"):
                try:
                    with st.spinner("Generating general response..."):
                        response = st.session_state.conversational_chain.invoke({
                            "question": f"Provide a general explanation about {st.session_state.pending_question}"
                        })
                    augmented_response = (
                        f"{response['answer']}\n\n"
                        f"💡 For specific implementation details, contact {CONFIG['SUPPORT_EMAIL']}"
                    )
                    display_assistant_response(augmented_response)
                finally:
                    st.session_state.show_general_prompt = False
        
        with col2:
            if st.button("No, thanks", key="general_no"):
                display_support_prompt()

# Main application flow
def main():
    configure_ui()
    initialize_system()
    display_chat_history()

    # Handle pending general response requests first
    if st.session_state.show_general_prompt:
        handle_general_response_request()
        return

    # Process new user input
    if user_input := st.chat_input("Ask me anything about WRTeam products..."):
        handle_user_query(user_input)

if __name__ == "__main__":
    main()
