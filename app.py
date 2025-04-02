import streamlit as st
import json
from pdf_processor import extract_text_from_pdf
from gemini_api import generate_questions, modify_question

st.title("📄 AI-Generated Questions from PDF")

if "questions" not in st.session_state:
    st.session_state.questions = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
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
            mcqs = generate_questions(text, mcq_count, 0, question_type="mcq")
            for mcq in mcqs:
                mcq_lines = mcq.strip().split("\n")
                if len(mcq_lines) >= 2:
                    question_text = mcq_lines[0]
                    options = mcq_lines[1:-1]
                    correct_answer = mcq_lines[-1]
                    st.session_state.questions.append({
                        "question": question_text.strip(),
                        "options": [opt.strip() for opt in options],
                        "correct_answer": correct_answer.strip(),
                        "mark": 1,
                        "type": "mcq"
                    })
                else:
                    st.warning(f"Skipping malformed MCQ: {mcq}")
        
        # Generate other questions
        for marks, count in [(1, one_mark_count), (2, two_mark_count), (3, three_mark_count), (4, four_mark_count), (5, five_mark_count)]:
            if count > 0:
                generated_qs = generate_questions(text, count, marks)
                for q in generated_qs:
                    st.session_state.questions.append({"question": q.strip(), "mark": marks, "type": "short_answer"})

    if st.session_state.questions:
        st.subheader("Generated Questions:")
        
        for idx, q in enumerate(st.session_state.questions, 1):
            st.write(f"{idx}. {q['question']}")

        # Submit button to store data in a JSON file
        if st.button("Submit"):
            formatted_data = {"questions": st.session_state.questions}
            with open("stored_questions.json", "w") as json_file:
                json.dump(formatted_data, json_file, indent=4)
            st.success("Questions have been successfully stored!")
    
    # Modify Questions
    if st.session_state.questions:
        st.subheader("Modify a Question")
        question_numbers = list(range(1, len(st.session_state.questions) + 1))
        selected_q_num = st.selectbox("Select question number to modify", question_numbers)
        
        # User choice: AI-generated or manual input
        modify_option = st.radio("Modify by:", ("AI-generated", "Manual input"))
        
        if modify_option == "Manual input":
            manual_question = st.text_area("Enter your question:")
            if st.button("Update Question") and manual_question.strip():
                st.session_state.questions[selected_q_num - 1]["question"] = manual_question.strip()
                st.success(f"Updated Question {selected_q_num}: {manual_question.strip()}")
                # Update stored JSON file
                formatted_data = {"questions": st.session_state.questions}
                with open("stored_questions.json", "w") as json_file:
                    json.dump(formatted_data, json_file, indent=4)
        else:
            if st.button("Generate AI-Modified Question"):
                new_question = modify_question(text, selected_q_num, "short_answer").strip()
                st.session_state.questions[selected_q_num - 1]["question"] = new_question
                st.success(f"Updated Question {selected_q_num}: {new_question}")
                # Update stored JSON file
                formatted_data = {"questions": st.session_state.questions}
                with open("stored_questions.json", "w") as json_file:
                    json.dump(formatted_data, json_file, indent=4)
