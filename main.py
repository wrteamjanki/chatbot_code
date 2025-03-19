# Check for required packages
try:
    import pysqlite3
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
import sys
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
from langchain.schema import Document


# Configuration
CONFIG = {
    "SUPPORT_INFO": "\n\nFor further assistance, contact our support team:\n\ud83d\udcde +91-8849493106\n\ud83d\udce7 wrteam.priyansh@gmail.com",
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
        raw_docs = vectorstore.get().get('documents', [])
        
        # Ensure documents have proper structure
        documents = [Document(page_content=doc) if isinstance(doc, str) else doc for doc in raw_docs]
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        return EnsembleRetriever(
            retrievers=[
                vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=CONFIG["RETRIEVAL_SETTINGS"]["search_kwargs"]
                ),
                bm25_retriever
            ],
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
        return validation_chain.invoke({"question": question, "answer": answer}).strip().lower() == "valid"
    except:
        return False

def handle_query(user_input: str):
    try:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Analyzing query..."):
            response = st.session_state.convo_chain.invoke({"question": user_input})
            
            valid_response = (
                response.get("source_documents")
                and validate_answer(user_input, response["answer"])
            )
            
            if valid_response:
                response_text = f"{response['answer']}{CONFIG['SUPPORT_INFO']}"
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
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

def main():
    st.set_page_config(page_title="WRTeam AI Assistant", page_icon="💬", layout="centered")
    st.title("WRTeam AI Assistant")
    initialize_system()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if user_input := st.chat_input("Ask about WRTeam products..."):
        handle_query(user_input)

if __name__ == "__main__":
    main()
