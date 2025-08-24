import requests as req
import sys
from flask import Flask, request, url_for, redirect, flash, session
import requests as req
from flask_cors import CORS
import os
from werkzeug.security import check_password_hash, generate_password_hash
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'database'))
import journal_db as jdb
sys.path.append('../../model')
import model


server = Flask(__name__)

server.config.update(dict(SECRET_KEY='development key'))
CORS(server)


@server.route('/data', methods=['GET'])
def get_data():
    data = request.args.get('data', 'No data provided')
    # Here you can process the data as needed
    return f"Here is your data! {data}"

@server.route('/create_user', methods=['POST'])
def create_user():
    fname, lname = request.form.get('fname'), request.form.get('lname')
    email = request.form.get('email')
    pwd = request.form.get('password')
    cpwd = request.form.get('conf_password')
    if pwd != cpwd:
        return "Passwords do not match!"
    else:
        jdb.add_user(fname, lname, email, pwd)
        return redirect("http://localhost:3000/")


@server.route('/user_auth', methods=['POST'])
def user_auth():
    email = request.form.get('email')
    password = request.form.get('password')
    check = jdb.check_user_exists(email, password)
    print(check)
    if check:
        return "User is authenticated!"
    else:
        return "Authentication failed! Username or password is incorrect."


