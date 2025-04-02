import psycopg2

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="question_store",
        user="postgres",
        password="wrteam1510",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Create table
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

    print("✅ Table 'questions' created successfully!")

except psycopg2.Error as e:
    print(f"❌ Error: {e}")

finally:
    if 'conn' in locals() and conn:
        cursor.close()
        conn.close()
        print("🔌 Connection closed.")
