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

# Display Chat Messages
for message in st.session_state.get("chat_history", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.chat_history = st.session_state.get("chat_history", [])
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI Response
    try:
        with st.chat_message("assistant"):
            response = st.session_state.conversational_chain.invoke({"question": user_input})
            assistant_response = response["answer"]
            source_documents = response.get("source_documents", [])

            # If no relevant data is found
            if not source_documents:
                st.warning("I couldn’t find any exact match in my knowledge base. Would you like a general response?")
                
                # User Decision Handling
                col1, col2 = st.columns(2)
                with col1:
                    agree = st.button("Yes, give me a general response")
                with col2:
                    decline = st.button("No, end the conversation")

                if agree:
                    assistant_response = (
                        "Sure! While I don’t have exact data on this, here’s what I can share:\n\n"
                        "**[Insert a general response here based on the topic]**"
                    )
                elif decline:
                    assistant_response = (
                        "I understand! If you need more specific help, feel free to reach out to our support team:\n\n"
                        f"📞 {SUPPORT_NUMBER}  \n📧 {SUPPORT_EMAIL}  \n\n"
                        "Have a great day! 😊"
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    st.markdown(assistant_response)
                    st.stop()

                # Include Support Info
                assistant_response += (
                    f"\n\n💡 If you need personalized assistance, you can contact WRTeam support:\n📞 {SUPPORT_NUMBER} \n📧 {SUPPORT_EMAIL}"
                )

            # Display Response
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        st.error(f"An error occurred: {e}")
