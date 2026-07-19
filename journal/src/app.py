import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'database'))
sys.path.append('../../model')

import requests as req
from flask import Flask, jsonify, request, url_for, redirect, flash, session, Response
from flask_cors import CORS
import datetime
from werkzeug.security import check_password_hash, generate_password_hash
import journalDB as jdb
from model import inference, txtEmotionModel, tokenizer
import datetime
import pandas as pd
import time
import json
import math


server = Flask(__name__)
# allow requests from the React dev origin and allow cookies
CORS(server, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
server.config['SESSION_COOKIE_SAMESITE'] = 'None'   # allow cross-site cookie
server.config['SESSION_COOKIE_SECURE'] = False     # set to True in production with HTTPS
server.config['CORS_HEADERS'] = 'Content-Type'

server.config.update(dict(SECRET_KEY='development key'))
CORS(server, supports_credentials=True)

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
    elif jdb.check_user_exists(email, pwd):
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
        print(session)
        return "User is authenticated!"
    else:
        return "Authentication failed! Username or password is incorrect."


@server.route('/add_entry', methods=['POST'])
def add_entry():
    check = checkSession()
    if not check:
        return "You are not logged in!"

    if not checkTime():
        return "It is not time to journal yet. Come back in "


    user = session.get('user')
    content = request.form.get('entry')
    emotion = inference(txtEmotionModel, content, tokenizer)
    print(user, content, emotion)
    jdb.addEntry(user, content, emotion, datetime.datetime.now())
    if checkContent(content):
        return "Entry contains sensitive content."
    return "Entry added successfully!"


@server.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect("http://localhost:3000/")

@server.route('/fetch_entries', methods=['POST'])
def fetch_entries():
    loggedIn = checkSession()
    if not loggedIn:
        return "You are not logged in!"
    user = session.get('user')
    entries = jdb.fetchEntries(user)
    if not entries:
        return "No entries found!"
    for i in range(len(entries)):
        entries[i] = list(entries[i])
        entries[i][3] = txtEmotionModel.getMap()[entries[i][3]]
        entries[i][4] = entries[i][4].strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(entries)

@server.route('/delete_entry', methods=['POST'])
def delete_entry():
    loggedIn = checkSession()
    if not loggedIn:
        return "You are not logged in!"
    user = session.get('user')
    entry_id = request.form.get('entryID')
    if not entry_id:
        return "No entry selected!"
    jdb.deleteEntry(user, entry_id)
    return "Entry deleted successfully!"

@server.route('/edit_entry', methods=['POST'])
def edit_entry():
    loggedIn = checkSession()
    if not loggedIn:
        return "You are not logged in!"
    user = session.get('user')
    entry_id = int(request.form.get('entryID'))
    content = request.form.get('content')
    emotion = inference(txtEmotionModel, content, tokenizer)
    jdb.editEntry(user, entry_id, content, emotion)
    return "Entry edited successfully!"


def checkContent(content):
    keywords = ['suicide', 'end my life', 'ending my life', 'kill myself', 'self harm']
    if any(keyword in content.lower() for keyword in keywords):
        return True
    return False

def checkSession():
    if not session.get('logged_in'):
        return False
    return True

@server.route('/checkTime', methods=["POST"])
def checkTime():
    '''Checks the time of the most recent journal entry from the database.
    Users are only allowed to submit an entry every 24 hours. If a user tries to do so before 24 hours since the last, the entry
    is rejected.'''

    currentTime = datetime.datetime.now()
    try:
        user = session.get('user')
        entries = jdb.fetchEntries(user)
        print(type(entries))

        lastDate = entries[-1][-1]

        timeDelta = currentTime - lastDate

        hoursDif = timeDelta.total_seconds() / 3600

        print(hoursDif)




        if hoursDif < 24:
            displayTime = hrsRemainingToHMS(hoursDif)
            return jsonify({"result": "False", "hours": hoursDif})
        else:
            return jsonify({"result": "True", "hours": hoursDif})




    except Exception as e:
        print("no go partner", e)

def changePWD(username, newPWD):
    hashed = generate_password_hash(newPWD)
    value = jdb.changePassword(username, hashed)
    print(value)

    if not value:
        print("Password could not be changed")

    else:
        print("Password successfully changed!")


def hrsRemainingToHMS(hours):
    remaining = 24 - int(hours)

    totalSeconds = remaining * 3600

    hrs = math.floor(totalSeconds / 3600)
    mins = math.floor((totalSeconds % 3600) / 60)
    secs =  totalSeconds% 60

    return f"{hrs}:{mins}:{secs}"

def fmtSeconstoHMS(seconds):
    seconds = max(0, int(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def checkTimeLeft(user):
    '''Checks the time of the most recent journal entry from the database.
    Users are only allowed to submit an entry every 24 hours. If a user tries to do so before 24 hours since the last, the entry
    is rejected.'''

    currentTime = datetime.datetime.now()
    try:
        if not user:
            return "False",0

        entries = jdb.fetchEntries(user)
        if not entries:
            return "True", 0

        lastDate = entries[-1][-1]
        timeDelta = currentTime - lastDate
        elapsedSeconds = timeDelta.total_seconds()
        totalWindow = 24 * 3600
        remaining = max(0, totalWindow - elapsedSeconds)

        allowed = remaining == 0
        return "True" if allowed else "False", int(remaining)

    except Exception as e:
        print("no go partner", e)
        return "False", 0



@server.route("/stream")
def stream():
    user = session.get('user')
    status, timeSince = checkTimeLeft(user)
    def streamTime():

        try:
            rem = int(timeSince)
            while rem > 0:
                payload = {
                    "value": fmtSeconstoHMS(rem),
                    "remaining":rem,
                    "allowed": status == "True"
                }
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1)
                rem -= 1



            payload = {"value": "00:00:00", "remaining": 0, "allowed": True}
            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            print("No")


    return Response(streamTime(), mimetype="text/event-stream")








def fetchLastEntry():
    pass

def stats():
    pass


#changePWD('jkougan@iwu.edu', "Gates108")