import os
import streamlit as st
from packaging import version

# Import necessary libraries
try:
    import pysqlite3
    import sqlite3
except ImportError:
    import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents.embeddings import get_embeddings

# Configuration
CONFIG = {
    "SUPPORT_NUMBER": "1234567890",
    "SUPPORT_EMAIL": "support@example.com",
    "VECTOR_DB_DIR": "vector_db",
    "MODEL_NAME": "gemini-pro",
    "REQUIRED_SQLITE_VERSION": "3.35.0",
    "SUPPORT_MESSAGE": "Sorry, I didn't know. You can contact {number} or email {email}."
}

# Ensure SQLite version compatibility
def check_sqlite_version():
    if version.parse(sqlite3.sqlite_version) < version.parse(CONFIG["REQUIRED_SQLITE_VERSION"]):
        raise RuntimeError(f"SQLite version must be >= {CONFIG['REQUIRED_SQLITE_VERSION']}")

# Ensure API key is set
def check_env_variables():
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

# Initialize Vector Database
def setup_vectorstore():
    if not os.path.exists(CONFIG["VECTOR_DB_DIR"]):
        st.error("Vector database directory not found.")
        return None
    return Chroma(persist_directory=CONFIG["VECTOR_DB_DIR"], embedding_function=get_embeddings())

# Initialize System Components
def initialize_system():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore()
    
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("show_general_prompt", False)

    if "conversational_chain" not in st.session_state:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatGoogleGenerativeAI(model=CONFIG["MODEL_NAME"], temperature=0.3)
        retriever = st.session_state.vectorstore.as_retriever()
        st.session_state.conversational_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

# Configure Streamlit UI
def configure_ui():
    st.set_page_config(
        page_title="WRTeam AI Assistant",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.title("💬 WRTeam AI Assistant")
    st.caption("Your smart assistant for all queries.")

# Display Chat History
def display_chat_history():
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

# Handle AI Response
def handle_knowledgebase_response(response):
    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
    st.session_state.pending_question = None

def handle_general_response():
    st.session_state.chat_history.append({"role": "assistant", "content": "Would you like an AI-generated response instead?"})
    st.session_state.show_general_prompt = True

def handle_support_redirect():
    support_msg = CONFIG["SUPPORT_MESSAGE"].format(number=CONFIG["SUPPORT_NUMBER"], email=CONFIG["SUPPORT_EMAIL"])
    st.session_state.chat_history.append({"role": "assistant", "content": support_msg})
    st.session_state.show_general_prompt = False

# Process User Query
def process_user_query(user_input):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    response = st.session_state.conversational_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
    if response["source_documents"]:
        handle_knowledgebase_response(response)
    else:
        handle_general_response()

# Main Function
def main():
    configure_ui()
    initialize_system()
    display_chat_history()
    
    if st.session_state.show_general_prompt:
        if st.button("Yes, generate an AI response"):
            process_user_query(st.session_state.pending_question)
        elif st.button("No, I need support"):
            handle_support_redirect()
    
    user_input = st.chat_input("Ask me anything...")
    if user_input:
        process_user_query(user_input)

# Entry Point
if __name__ == "__main__":
    check_sqlite_version()
    check_env_variables()
    main()
