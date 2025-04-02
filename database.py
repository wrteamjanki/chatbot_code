import psycopg2

# Function to connect to PostgreSQL
def connect_db():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="wrteam1510",
        host="localhost",
        port="5432"
    )

# Function to create the table if it doesn't exist
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS questions (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        options TEXT[],
        mark INTEGER NOT NULL,
        type TEXT NOT NULL
    );
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Table is ready!")

# Function to store a question
def store_question(question, options, mark, q_type):
    conn = connect_db()
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO questions (question, options, mark, type)
    VALUES (%s, %s, %s, %s);
    """
    
    cursor.execute(insert_query, (question, options, mark, q_type))
    conn.commit()
    
    cursor.close()
    conn.close()
    print("✅ Question stored successfully!")

# Function to fetch all questions
def fetch_questions():
    conn = connect_db()
    cursor = conn.cursor()
    
    fetch_query = "SELECT * FROM questions;"
    cursor.execute(fetch_query)
    questions = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return questions

if __name__ == "__main__":
    # Test storing a question
    test_question = {
        "question": "What is the capital of France?",
        "options": "['Paris', 'Berlin', 'Madrid', 'Rome']",
        "mark": 1,
        "type": "MCQ"
    }
    
    print("📝 Storing test question...")
    store_question(
        question=test_question["question"],
        options=test_question["options"],
        mark=test_question["mark"],
        q_type=test_question["type"]
    )

    # Test fetching questions
    print("📥 Fetching stored questions...")
    questions = fetch_questions()
    for q in questions:
        print(q)

