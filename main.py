import os
import json
from rapidfuzz import fuzz
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# File to store unanswered questions
history_dir = os.path.join(working_dir, "chat_history")
os.makedirs(history_dir, exist_ok=True)
history_file = os.path.join(history_dir, "chat_history.json")

def save_unanswered_question(question):
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if question not in data:
        data.append(question)
        with open(history_file, "w") as f:
            json.dump(data, f, indent=4)

# Custom greeting responses
def handle_greetings(user_input):
    greetings = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! How can I help?",
        "good morning": "Good morning! How can I assist you?",
        "good afternoon": "Good afternoon! What can I do for you?",
        "good evening": "Good evening! How can I help?",
        "thank you": "You're welcome! Happy to help!",
        "bye": "Goodbye! Have a great day!",
        "how are you": "I'm fine. How can I help you? :)",
        "who are you": "I'm a bot. I will help you with your questions. Please ask your query.",
        "what your name": "My name is Momo. I will help you with your query.",
        "who made you": "Miss Janki Chauhan made me to help with your queries.",
    }
    for key, response in greetings.items():
        if fuzz.partial_ratio(user_input.lower(), key) > 80:
            return response
    return None

# Check if response is relevant
def is_relevant_response(user_input, assistant_response, threshold=80):
    similarity = fuzz.partial_ratio(user_input.lower(), assistant_response.lower())
    return similarity >= threshold

# Set up vectorstore
def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

# Set up chatbot with updated ConversationalRetrievalChain
def chat_chain(vectorstore):
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = ConversationBufferMemory(
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=False
    )
    return chain

# Streamlit UI
st.set_page_config(page_title="E-LMS Chatbot", page_icon="📚")
st.title("📚 E-LMS Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask AI...")

if user_input:
    if not any(msg["content"] == user_input for msg in st.session_state.chat_history if msg["role"] == "user"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    greeting_response = handle_greetings(user_input)
    if greeting_response:
        assistant_response = greeting_response
    else:
        try:
            response = st.session_state.conversational_chain({"question": user_input})
            assistant_response = response["answer"].strip()
            if not assistant_response or not is_relevant_response(user_input, assistant_response):
                assistant_response = "I currently don't have an answer for that. I'll note this question for future improvements. For now, you can contact our team at 1234567890 or email wrteam.support@gmail.com."
                save_unanswered_question(user_input)
        except Exception:
            assistant_response = "I currently don't have an answer for that. I'll note this question for future improvements. For now, you can contact our team at 1234567890 or email wrteam.support@gmail.com."
            save_unanswered_question(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    
    if not any(msg["content"] == assistant_response for msg in st.session_state.chat_history if msg["role"] == "assistant"):
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
