import streamlit as st
import psycopg2
from pdf_processor import extract_text_from_pdf
from gemini_api import generate_questions, modify_question
from database import store_question  # Import function to store questions in PostgreSQL
from vector_store import retrieve_similar_text  # Retrieve relevant content before question generation

st.title("📄 AI-Generated Questions from PDF")

if "questions" not in st.session_state:
    st.session_state.questions = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    relevant_text = retrieve_similar_text(text)  # Get the most relevant content
    
    st.subheader("Select Question Types")
    mcq_count = st.number_input("MCQs", min_value=0, value=2)
    one_mark_count = st.number_input("1-mark questions", min_value=0, value=1)
    two_mark_count = st.number_input("2-mark questions", min_value=0, value=1)
    three_mark_count = st.number_input("3-mark questions", min_value=0, value=1)
    four_mark_count = st.number_input("4-mark questions", min_value=0, value=1)
    five_mark_count = st.number_input("5-mark questions", min_value=0, value=1)
    
    if st.button("Generate Questions"):
        st.session_state.questions = []  # Reset previous questions
        
        # Generate MCQs first
        if mcq_count > 0:
            mcqs = generate_questions(relevant_text, mcq_count, 0, question_type="mcq")
            for mcq in mcqs:
                mcq_lines = [line.strip() for line in mcq.strip().split("\n") if line.strip()]
                
                if len(mcq_lines) < 2:
                    st.warning(f"Skipping malformed MCQ: {mcq}")
                    continue
                
                question_text = mcq_lines[0]
                options = mcq_lines[1:-1] if len(mcq_lines) > 2 else []
                correct_answer = mcq_lines[-1] if len(mcq_lines) > 1 else ""
                
                if not options or not correct_answer:
                    st.warning(f"Incomplete MCQ: {mcq}")
                    continue
                
                question_data = {
                    "question": question_text,
                    "options": options,
                    "correct_answer": correct_answer,
                    "mark": 1,
                    "type": "mcq"
                }
                st.session_state.questions.append(question_data)
                store_question(question_text, options, 1, "mcq")  # Store in DB
        
        # Generate other questions
        for marks, count in [(1, one_mark_count), (2, two_mark_count), (3, three_mark_count), (4, four_mark_count), (5, five_mark_count)]:
            if count > 0:
                generated_qs = generate_questions(relevant_text, count, marks)
                for q in generated_qs:
                    clean_question = q.strip()
                    
                    if not clean_question or "****" in clean_question:
                        st.warning(f"Skipping malformed question: {q}")
                        continue
                    
                    question_data = {"question": clean_question, "mark": marks, "type": "short_answer"}
                    st.session_state.questions.append(question_data)
                    store_question(clean_question, None, marks, "short_answer")  # Store in DB
    
    if st.session_state.questions:
        st.subheader("Generated Questions:")
        
        for idx, q in enumerate(st.session_state.questions, 1):
            st.write(f"{idx}. {q['question']} ({q['mark']} marks)")
    
    # Submit Questions to Database
    if st.session_state.questions and st.button("Submit Questions to Database"):
        st.success("All generated questions have been stored in the database!")
    
    # Modify Questions
    if st.session_state.questions:
        st.subheader("Modify a Question")
        question_numbers = list(range(1, len(st.session_state.questions) + 1))
        selected_q_num = st.selectbox("Select question number to modify", question_numbers)
        
        modify_option = st.radio("Modify by:", ("AI-generated", "Manual input"))
        
        if modify_option == "Manual input":
            manual_question = st.text_area("Enter your question:")
            if st.button("Update Question") and manual_question.strip():
                st.session_state.questions[selected_q_num - 1]["question"] = manual_question.strip()
                st.success(f"Updated Question {selected_q_num}: {manual_question.strip()}")
        else:
            if st.button("Generate AI-Modified Question"):
                new_question = modify_question(relevant_text, selected_q_num, "short_answer").strip()
                st.session_state.questions[selected_q_num - 1]["question"] = new_question
                st.success(f"Updated Question {selected_q_num}: {new_question}")
