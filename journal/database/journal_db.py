import os
import mysql.connector
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()
os.getenv("DBPWD")

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password=os.getenv("DBPWD"),
        database='journal'
    )
    return connection

def close_db_connection(connection):
    if connection.is_connected():
        connection.close()

def add_user(fname, lname, email, password):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user:
            print("User already exists!")
            return
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return
    cursor.execute("INSERT INTO users (fname, lname, email, pwd) VALUES (%s, %s, %s, %s)",
                   (fname, lname, email, generate_password_hash(password)))
    connection.commit()
    close_db_connection(connection)


def check_user_exists(email, password):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    close_db_connection(connection)
    if user and check_password_hash(user[4], password):
        return True
    return False


def addEntry(user_id, entry_text):
    connection = get_db_connection()
    pass

def remove():
    pass
