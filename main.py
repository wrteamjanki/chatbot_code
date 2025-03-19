# Check for required packages
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
        raise RuntimeError(f"System sqlite3 version {sqlite3.sqlite_version} is too old. Install pysqlite3-binary")

import os
import streamlit as st
import gemini_api
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorized_documents import embeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
CONFIG = {
    "SUPPORT_INFO": "\n\nFor further assistance, contact our support team:\n📞 +91-8849493106\n📧 wrteam.priyansh@gmail.com",
    "VECTOR_DB_DIR": "vectordb",
    "MODEL_NAME": "gemini-1.5-flash-002",
    "RETRIEVAL_SETTINGS": {
        "search_kwargs": {
            "k": 5,
            "score_threshold": 0.65
        },
        "validation_prompt": """Verify if this answer properly addresses the question "{question}": 
        {answer}
        Respond ONLY with 'valid' or 'invalid'"""
    }
}

# Environment checks
if "GEMINI_API_KEY" not in os.environ:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

@st.cache_resource(show_spinner="Initializing knowledge base...")
def setup_retriever():
    try:
        vectorstore = Chroma(
            persist_directory=CONFIG["VECTOR_DB_DIR"],
            embedding_function=embeddings
        )
        bm25_retriever = BM25Retriever.from_documents(vectorstore.get()['documents'])
        return EnsembleRetriever(
            retrievers=[vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=CONFIG["RETRIEVAL_SETTINGS"]["search_kwargs"]
            ), bm25_retriever],
            weights=[0.7, 0.3]
        )
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

def initialize_system():
    session_defaults = {
        "chat_history": [],
        "pending_question": None,
        "show_general_prompt": False,
        "validation_attempts": 0,
        "retriever": None,
        "convo_chain": None
    }
    
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)
    
    if not st.session_state.retriever:
        try:
            st.session_state.retriever = setup_retriever()
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()

    if not st.session_state.convo_chain:
        try:
            llm = ChatGoogleGenerativeAI(
                model=CONFIG["MODEL_NAME"],
                temperature=0.3,
                google_api_key=os.environ["GEMINI_API_KEY"]
            )
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            st.session_state.convo_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.retriever,
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            st.error(f"Chain creation failed: {str(e)}")
            st.stop()

def validate_answer(question: str, answer: str) -> bool:
    validation_chain = (
        ChatPromptTemplate.from_template(CONFIG["RETRIEVAL_SETTINGS"]["validation_prompt"])
        | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        | StrOutputParser()
    )
    try:
        return validation_chain.invoke({
            "question": question,
            "answer": answer
        }).strip().lower() == "valid"
    except:
        return False

def handle_query(user_input: str):
    try:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Analyzing query..."):
            response = st.session_state.convo_chain.invoke({"question": user_input})
            
            valid_response = (
                response["source_documents"]
                and validate_answer(user_input, response["answer"])
            )
            
            if valid_response:
                response_text = f"{response['answer']}{CONFIG['SUPPORT_INFO']}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text
                })
            else:
                if st.session_state.validation_attempts < 2:
                    st.session_state.validation_attempts += 1
                    handle_query(f"Explain {user_input} in simple terms")
                else:
                    st.session_state.pending_question = user_input
                    st.session_state.show_general_prompt = True
                    st.session_state.validation_attempts = 0

    except Exception as e:
        error_msg = f"Processing error: {str(e)}{CONFIG['SUPPORT_INFO']}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    finally:
        st.rerun()

def handle_general_response():
    try:
        with st.spinner("Generating general response..."):
            response = st.session_state.convo_chain.invoke({
                "question": st.session_state.pending_question
            })
        response_text = f"{response['answer']}{CONFIG['SUPPORT_INFO']}"
    except Exception as e:
        response_text = f"Response generation failed: {str(e)}{CONFIG['SUPPORT_INFO']}"
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text
    })
    st.session_state.show_general_prompt = False
    st.session_state.pending_question = None
    st.rerun()

def handle_support_redirect():
    support_msg = f"We couldn't find a specific answer.{CONFIG['SUPPORT_INFO']}"
    st.session_state.chat_history.append({"role": "assistant", "content": support_msg})
    st.session_state.show_general_prompt = False
    st.session_state.pending_question = None
    st.rerun()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="WRTeam AI Assistant",
        page_icon="💬",
        layout="centered"
    )
    st.title("WRTeam AI Assistant")
    
    initialize_system()
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle general response prompt
    if st.session_state.show_general_prompt:
        with st.chat_message("assistant"):
            st.warning("I couldn't find specific documentation. Would you like a general answer?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, please", key="general_yes"):
                    handle_general_response()
            with col2:
                if st.button("No, thanks", key="general_no"):
                    handle_support_redirect()
        return
    
    # Process new input
    if user_input := st.chat_input("Ask about WRTeam products..."):
        handle_query(user_input)

if __name__ == "__main__":
    main()
