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

# Display Chat Messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # Use conversational chain to get relevant documents
        response = st.session_state.conversational_chain.invoke({"question": user_input})
        assistant_response = response["answer"]
        source_documents = response.get("source_documents", [])

        if not source_documents:
            st.warning("I couldn’t find an exact match in my knowledge base. Would you like a general response?")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, give me a general response"):
                    st.session_state.general_response_requested = True
            with col2:
                if st.button("No, end the conversation"):
                    st.session_state.general_response_requested = False
                    assistant_response = (
                        "I understand! If you need more specific help, feel free to reach out to our support team:\n\n"
                        f" {SUPPORT_NUMBER}  \n {SUPPORT_EMAIL}  \n\n"
                        "Have a great day!"
                    )
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    st.stop()

        if st.session_state.general_response_requested:
            # Generate a general response if requested
            general_response = st.session_state.conversational_chain.llm.invoke(
                f"Provide a general explanation about {user_input}"
            )
            assistant_response = general_response if general_response else "I'm sorry, I don't have information on that."

            # Add customer support info
            assistant_response += (
                f"\n\n If you need personalized assistance, contact WRTeam support:\n {SUPPORT_NUMBER} \n {SUPPORT_EMAIL}"
            )

            # Display the general response
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            st.session_state.general_response_requested = False
        else:
            # If source documents are found, provide the response based on retrieved info
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"An error occurred: {e}")
