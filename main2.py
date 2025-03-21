import os
import streamlit as st
import gemini_api
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings

def chat_input_with_persistence():
    """
    Custom chat input box that retains the last entered message
    and stays near the send button with multi-line support.
    """
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""
    
    # Input field with persistent text
    user_input = st.text_area("", st.session_state.pending_question, key="chat_input", height=80)
    
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Send", key="send_button"):
            st.session_state.pending_question = user_input  # Preserve input after sending
            return user_input.strip()
    
    return None

# Configuration constants
CONFIG = {
    "SUPPORT_NUMBER": "+91-8849493106",
    "SUPPORT_EMAIL": "wrteam.priyansh@gmail.com",
    "VECTOR_DB_DIR": "vectordb",
    "MODEL_NAME": "gemini-1.5-flash-002",
    "SUPPORT_MESSAGE": f"\n\nFor further assistance, contact our support team:\n📞 +91-8849493106\n📧 wrteam.priyansh@gmail.com"
}

# Validate environment variables
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
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            st.error(f"Failed to create conversation chain: {str(e)}")
            st.stop()

def configure_ui():
    st.set_page_config(
        page_title="WRTeam AI Assistant",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.markdown("# 👻 WRTeam AI Assistant")
    st.caption("Powered by Gemini 1.5 Flash and ChromaDB")

def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_user_query(user_input: str):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing your query..."):
            response = st.session_state.conversational_chain.invoke({"question": user_input})
        assistant_response = f"{response['answer']}{CONFIG['SUPPORT_MESSAGE']}"
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

def main():
    configure_ui()
    initialize_system()
    display_chat_history()
    
    suggested_questions = [
        "What features does WRTeam's product offer?",
        "How does the subscription model work?",
        "Can I integrate this with my existing LMS?",
        "What are the pricing plans?",
        "Is there a trial version available?"
    ]
    
    st.markdown("### 💡 Suggested Questions")
    for question in suggested_questions:
        if st.button(question, key=f"suggested_{question}"):
            process_user_query(question)
            break
    
    user_input = chat_input_with_persistence()
    if user_input:
        process_user_query(user_input)

if __name__ == "__main__":
    main()
