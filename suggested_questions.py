# suggested_questions.py
import streamlit as st

def display_suggested_questions():
    """
    Displays suggested questions in a transparent and minimalistic message box.
    """
    st.markdown("## Suggested Questions")
    suggested_questions = [
        "What are the features of WRTeam products?",
        "How can I integrate WRTeam API?",
        "What pricing plans are available?",
        "How do I get customer support?",
        "Tell me about WRTeam's latest updates."
    ]
    
    cols = st.columns(len(suggested_questions))
    for i, question in enumerate(suggested_questions):
        if cols[i].button(question, key=f"suggested_{i}"):
            st.session_state["user_input"] = question
            st.rerun()