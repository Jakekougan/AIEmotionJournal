import os
import mysql.connector
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash


load_dotenv()
os.getenv("DBPWD")

def get_db_connection():
    '''Get connection object for MySQL database.

    Parameters:
        None

    Returns:
        A MySQL connection object that will allow us to perform operations'''
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password=os.getenv("DBPWD"),
        database='journal'
    )
    return connection

def close_db_connection(connection):
    '''closes the database connection if open'''
    if connection.is_connected():
        connection.close()

def add_user(fname, lname, email, password):
    '''add user to the database

    Parameters:
        fname (str): first name of user
        lname (str): last name of user
        email (str): email address of user
        password (str): password of user'''
    connection = get_db_connection()
    cursor = connection.cursor()

    #Check if the user already exists in the database
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user:
            print("User already exists!")
            return
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return

    #inserting the user
    cursor.execute("INSERT INTO users (fname, lname, email, pwd) VALUES (%s, %s, %s, %s)",
                   (fname, lname, email, generate_password_hash(password)))
    connection.commit()

    close_db_connection(connection)


def verify_password(stored_password, provided_password):
    '''helper function in user authentication by checking a user's password hash against the submitted one

    Parameters:
        stored_password (str): the user's password hash from the database
        provided_password (str): the plaintext password entered by user'''
    try:
        return check_password_hash(stored_password, provided_password)
    except (ValueError, TypeError):
        return stored_password == provided_password


def check_user_exists(email, password):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user and verify_password(user[4], password):
            return True
        return False
    finally:
        close_db_connection(connection)


def addEntry(user, entry_text, emotion, date):
    connection = get_db_connection()
    cursor = connection.cursor()
    udata = fetchUserData(user)
    try:
        cursor.execute("INSERT INTO entries (user, plaintext, label, date) VALUES (%s, %s, %s, %s)",
                    (udata[0], entry_text, emotion, date))
        connection.commit()
    except mysql.connector.Error as err:
        return err
    finally:
        close_db_connection(connection)

def fetchUserData(user):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (user,))
        user_data = cursor.fetchone()
        if not user_data:
            print("User not found!")
            return
        return user_data
    except mysql.connector.Error as err:
        return err

def changePassword(user, newPWD):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("UPDATE users SET pwd = %s WHERE email = %s", (newPWD, user))
        connection.commit()
        print("Yes")
        return True
    except mysql.connector.Error as err:
        print(f"Error changing password: {err}")
        return False
    finally:
        close_db_connection(connection)


def fetchEntries(user):
    connection = get_db_connection()
    cursor = connection.cursor()
    user_id = fetchUserData(user)[0]
    try:
        cursor.execute("SELECT * FROM entries WHERE user = %s", (user_id,))
        entries = cursor.fetchall()
        if not entries:
            print("No entries found!")
            return
        return entries
    except mysql.connector.Error as err:
        return err
    finally:
        close_db_connection(connection)


def editEntry(user, entry_id, content, emotion):
    connection = get_db_connection()
    cursor = connection.cursor()
    user_id = fetchUserData(user)[0]
    try:
        cursor.execute("UPDATE entries SET plaintext = %s, label = %s WHERE id = %s AND user = %s",
                       (content, emotion, entry_id, user_id))
        connection.commit()
    except mysql.connector.Error as err:
        return err
    finally:
        close_db_connection(connection)

def deleteEntry(user, entry_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    user_id = fetchUserData(user)[0]
    try:
        cursor.execute("DELETE FROM entries WHERE id = %s AND user = %s", (entry_id, user_id))
        connection.commit()

    except mysql.connector.Error as err:
        return err
    finally:
        close_db_connection(connection)
