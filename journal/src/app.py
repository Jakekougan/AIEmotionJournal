import requests as req
import sys
from flask import Flask, request, url_for, redirect, flash, session
import requests as req
from flask_cors import CORS
import os
import datetime
from werkzeug.security import check_password_hash, generate_password_hash
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'database'))
import journal_db as jdb
sys.path.append('../../model')
import model


server = Flask(__name__)

server.config.update(dict(SECRET_KEY='development key'))
CORS(server)

@server.route('/create_user', methods=['POST'])
def create_user():
    fname, lname = request.form.get('fname'), request.form.get('lname')
    email = request.form.get('email')
    pwd = request.form.get('password')
    cpwd = request.form.get('conf_password')
    if pwd != cpwd:
        return "Passwords do not match!"
    elif len(pwd) < 8:
        return "Password must be at least 8 characters long!"
    elif not jdb.check_user_exists(email):
        return "User already exists!"
    elif not fname or not lname or not email or not pwd or not cpwd:
        return "Please fill out all fields!"
    elif "@" not in email or "." not in email:
        return "Please enter a valid email address!"
    else:
        jdb.add_user(fname, lname, email, pwd)
        return redirect("http://localhost:3000/")


@server.route('/user_auth', methods=['POST'])
def user_auth():
    email = request.form.get('email')
    password = request.form.get('password')
    check = jdb.check_user_exists(email, password)
    if check:
        session['logged_in'] = True
        session['user'] = email
        return "User is authenticated!"
    else:
        return "Authentication failed! Username or password is incorrect."





