import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings
import gemini_api  # Ensure GEMINI_API_KEY is set
from packaging import version

# Support information
SUPPORT_INFO = """
For personalized assistance, contact WRTeam support:
📞 +91-8849493106
📧 wrteam.priyansh@gmail.com
"""

VECTORDB_DIR = "vectordb"

# Ensure pysqlite3 is installed if sqlite3 version is below the required
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    import sqlite3
    if version.parse(sqlite3.sqlite_version) < version.parse("3.35.0"):
        raise RuntimeError("Your system sqlite3 version is too old. Please install pysqlite3-binary.")

# Initialize vector store
@st.experimental_memo
def get_vectorstore():
    return Chroma(persist_directory=VECTORDB_DIR, embedding_function=embeddings)

# Initialize conversation chain
@st.experimental_memo
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

# Page setup
st.set_page_config(page_title="WRTeam AI Assistant", page_icon="💬")
st.title("WRTeam AI Assistant 🚀")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat logic
if user_query := st.chat_input("Ask about WRTeam products/services"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    try:
        # Initialize components
        vectorstore = get_vectorstore()
        qa_chain = get_conversation_chain(vectorstore)
        
        # Get response
        response = qa_chain({"question": user_query})
        answer = response.get("answer", "No answer found.")
        sources = response.get("source_documents", [])
        
        # Handle unknown queries
        if not sources:
            answer = f"I couldn't find specific information in our knowledge base.\n\n{SUPPORT_INFO}"
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
    except Exception as e:
        error_msg = f"Sorry, an error occurred: {str(e)}\n\n{SUPPORT_INFO}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Redisplay all messages
    st.rerun()
