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
        raise ImportError("Missing 'packaging' package. Run: pip install packaging")
    
    if version.parse(sqlite3.sqlite_version) < version.parse("3.35.0"):
        raise RuntimeError(f"Old sqlite3 ({sqlite3.sqlite_version}). Install pysqlite3-binary")

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
from langchain.schema import Document

# Configuration
CONFIG = {
    "SUPPORT_INFO": "\n\nContact support:\n📞 +91-8849493106\n📧 wrteam.priyansh@gmail.com",
    "VECTOR_DB_DIR": "vectordb",
    "MODEL_NAME": "gemini-1.5-flash-002",
    "RETRIEVAL_SETTINGS": {
        "search_kwargs": {
            "k": 3,  # Reduced from 5
            "score_threshold": 0.7  # Increased threshold
        },
        "validation_prompt": """Verify if this answers "{question}": {answer} Respond ONLY 'valid'/'invalid'"""
    }
}

# Environment checks
if "GEMINI_API_KEY" not in os.environ:
    raise EnvironmentError("GEMINI_API_KEY not set")

@st.cache_resource(show_spinner=False)  # Disable default spinner
def setup_retriever():
    @st.cache_data(show_spinner="Loading knowledge base...")
    def load_docs():
        vectorstore = Chroma(
            persist_directory=CONFIG["VECTOR_DB_DIR"],
            embedding_function=embeddings
        )
        return vectorstore.get().get('documents', [])

    try:
        docs = load_docs()
        documents = [Document(page_content=doc) if isinstance(doc, str) else doc for doc in docs]
        
        # Pre-index BM25 for faster retrieval
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 3  # Limit BM25 results
        
        vectorstore = Chroma(
            persist_directory=CONFIG["VECTOR_DB_DIR"],
            embedding_function=embeddings
        )
        
        return EnsembleRetriever(
            retrievers=[
                vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=CONFIG["RETRIEVAL_SETTINGS"]["search_kwargs"]
                ),
                bm25_retriever
            ],
            weights=[0.8, 0.2]  # Favor vector search more
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
    
    for key in session_defaults:
        st.session_state.setdefault(key, session_defaults[key])
    
    if not st.session_state.retriever:
        try:
            st.session_state.retriever = setup_retriever()
        except Exception as e:
            st.error(f"Init error: {str(e)}")
            st.stop()

    if not st.session_state.convo_chain:
        try:
            llm = ChatGoogleGenerativeAI(
                model=CONFIG["MODEL_NAME"],
                temperature=0.3,
                google_api_key=os.environ["GEMINI_API_KEY"],
                max_output_tokens=512  # Limit response size
            )
            st.session_state.convo_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.retriever,
                memory=ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                ),
                return_source_documents=True,
                verbose=False  # Disable verbose logging
            )
        except Exception as e:
            st.error(f"Chain failed: {str(e)}")
            st.stop()

@st.cache_data(ttl=300)  # Cache validation results
def validate_answer(question: str, answer: str) -> bool:
    validation_chain = (
        ChatPromptTemplate.from_template(CONFIG["RETRIEVAL_SETTINGS"]["validation_prompt"])
        | ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0)  # Faster model
        | StrOutputParser()
    )
    try:
        return validation_chain.invoke({
            "question": question[:100],  # Truncate long questions
            "answer": answer[:500]      # Truncate long answers
        }).strip().lower() == "valid"
    except:
        return False

def handle_query(user_input: str):
    try:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show custom progress bar
        progress_bar = st.progress(0, text="Processing your query...")
        
        with st.spinner(""):
            response = st.session_state.convo_chain.invoke({"question": user_input})
            progress_bar.progress(50)
            
            valid_response = response.get("source_documents") and validate_answer(user_input, response["answer"])
            progress_bar.progress(80)
            
            if valid_response:
                response_text = f"{response['answer']}{CONFIG['SUPPORT_INFO']}"
            else:
                if st.session_state.validation_attempts < 1:  # Only 1 retry
                    st.session_state.validation_attempts += 1
                    handle_query(f"Explain briefly: {user_input}")
                else:
                    st.session_state.pending_question = user_input
                    st.session_state.show_general_prompt = True
                    st.session_state.validation_attempts = 0
            
            progress_bar.progress(100)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    except Exception as e:
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"Error: {str(e)}{CONFIG['SUPPORT_INFO']}"
        })
    finally:
        if 'progress_bar' in locals():
            progress_bar.empty()
        st.rerun()

def main():
    st.set_page_config(page_title="WRTeam AI Assistant", page_icon="💬", layout="centered")
    st.title("WRTeam AI Assistant")
    
    initialize_system()
    
    # Chat history with container
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Input and general prompt handling
    if user_input := st.chat_input("Ask about WRTeam products..."):
        handle_query(user_input)
    
    if st.session_state.show_general_prompt:
        with chat_container:
            with st.chat_message("assistant"):
                st.warning("No specific documentation found. Get general answer?")
                cols = st.columns(2)
                with cols[0]:
                    if st.button("Yes", key="yes_btn"):
                        handle_query(st.session_state.pending_question)
                with cols[1]:
                    if st.button("Contact Support", key="support_btn"):
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"Contact support:{CONFIG['SUPPORT_INFO']}"
                        })
                        st.session_state.show_general_prompt = False
                        st.rerun()

if __name__ == "__main__":
    main()
