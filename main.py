import os
import json
import streamlit as st
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

# Load API Key
config_path = "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_data = json.load(f)
        os.environ["GEMINI_API_KEY"] = config_data.get("GEMINI_API_KEY", "")

# Cached Vector Store Setup
@st.cache_resource
def setup_vectorstore():
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# Function to Create Chat Chain
@st.cache_resource
def chat_chain(vectorstore):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=api_key)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

            # Handle support redirection
            if "I'm sorry" in assistant_response or not assistant_response.strip():
                assistant_response = (
                    f"I'm sorry, but I couldn't find an answer in my knowledge base. "
                    f"Please contact WRTeam support at **{SUPPORT_NUMBER}** or email **{SUPPORT_EMAIL}** for further assistance."
                )

            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    except Exception as e:
        st.error(f"An error occurred: {e}")
