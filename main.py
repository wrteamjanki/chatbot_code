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
import gemini_api
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
    "SUPPORT_MESSAGE": "\n\nFor further assistance, contact our support team: \nPhone: +91-8849493106\nEmail: wrteam.priyansh@gmail.com"
}

if "GEMINI_API_KEY" not in os.environ:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

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

def initialize_system():
    session_defaults = {
        "chat_history": [],
        "vectorstore": None,
        "conversational_chain": None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.vectorstore is None:
        try:
            st.session_state.vectorstore = setup_vectorstore()
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

    if st.session_state.conversational_chain is None:
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
                return_source_documents=True
            )
        except Exception as e:
            st.error(f"Failed to create conversation chain: {str(e)}")
            st.stop()

def configure_ui():
    st.set_page_config(
        page_title="WRTeam AI Assistant",
        page_icon=None,  # Removed emojis and replaced with None
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.markdown("# WRTeam AI Assistant")
    st.caption("Powered by Gemini 1.5 Flash and ChromaDB")

def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"].encode("utf-8", "ignore").decode("utf-8"))

def handle_response(response, query):
    if response["source_documents"]:
        assistant_response = response['answer']
    else:
        assistant_response = generate_general_response(query)
    
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.rerun()

def generate_general_response(query: str):
    try:
        with st.spinner("Providing general explanation..."):
            general_response = st.session_state.conversational_chain.invoke({
                "question": f"Provide a general explanation about {query}"
            })
        return f"{general_response['answer']}\n{CONFIG['SUPPORT_MESSAGE']}"
    except Exception as e:
        return f"Failed to generate response: {str(e)}\n{CONFIG['SUPPORT_MESSAGE']}"

def process_user_query(user_input: str):
    try:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing your query..."):
            response = st.session_state.conversational_chain.invoke({"question": user_input})
        handle_response(response, user_input)
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        st.rerun()

def main():
    configure_ui()
    initialize_system()
    display_chat_history()
    if user_input := st.chat_input("Ask me anything about WRTeam products..."):
        process_user_query(user_input)

if __name__ == "__main__":
    main()
