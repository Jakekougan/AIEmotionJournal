import os
import mysql.connector
from dotenv import load_dotenv

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

