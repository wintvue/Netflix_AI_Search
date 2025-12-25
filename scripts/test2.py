import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DB_NAME")
print(USER, HOST, PORT, DBNAME)
# Connect to the database
try:
    # with psycopg2.connect(
    #     user=USER,
    #     password=PASSWORD,
    #     host=HOST,
    #     port=PORT,
    #     dbname=DBNAME
    # ) as conn:
    #     conn.execute("SET statement_timeout = 0;")
    #     with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
    #         # Pull what we need from your existing movies table
    #         print("Connection successful!")

    # connection = psycopg2.connect(
    #     user=USER,
    #     password=PASSWORD,
    #     host=HOST,
    #     port=PORT,
    #     dbname=DBNAME
    # )
    # print("Connection successful!")
    
    # # Create a cursor to execute SQL queries

    # connection.execute("SET statement_timeout = 0;")
    # print("Statement timeout set to 0:", connection.fetchone())

    # cursor = connection.cursor()
    
    # # Example query
    # cursor.execute("SELECT NOW();")
    # result = cursor.fetchone()
    # print("Current Time:", result)
    # cursor.execute("SELECT 'aaaaa'");
    # result = cursor.fetchall()
    # print("Current Time + 1 hour:", result)

    # # Close the cursor and connection
    # cursor.close()
    # connection.close()
    # print("Connection closed.")

    print("Connecting to database with: ", USER, HOST, PORT, DBNAME)
    conn = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")
    conn.autocommit = False

except Exception as e:
    print(f"Failed to connect: {e}")