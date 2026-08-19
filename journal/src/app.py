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
    '''Request handler to handle creating user accounts.

    Takes the first and last names, email, password and confirmation of password from the front end, adds a new
    entry into the users database for that new user, returns a redirect to the sign in page.'''

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
    '''Request handler for validating a user trying to sign in has an account. Will help determine if a user is signed to allow for navigation
    to the landing page.

    takes the email and the password from the frontend, checks if the user exists in the users table.
    If the user exists, return message that user is authenticated. If not send a failure message as a string.  '''
    email = request.form.get('email')
    password = request.form.get('password')
    check = jdb.check_user_exists(email, password)
    if check:
        session['logged_in'] = True
        session['user'] = email
        return "User is authenticated!"
    else:
        return "Authentication failed! Username or password is incorrect."


@server.route('/add_entry', methods=['POST'])
def add_entry():
    '''Request handler to add a journal entry into the entries table of the database.

    takes the user, and entry content from the frontend, passes the content into the model for inference, takes the current
    date and time, and passes it all into the entries table of the database. Finally return response as a string.

    If entry was successfully added return success message. If entry could not be return error message.

    Will also check for any language that could indicate self-harm or suicidal ideation.'''

    #check if user is signed in
    check = checkSession()
    if not check:
        return "You are not logged in!"

    #Check if it has been 24 hours since the last submitted journal entry. If not return error message.
    if not checkTime():
        return "It is not time to journal yet. Come back in "


    user = session.get('user')
    content = request.form.get('entry')
    emotion = inference(txtEmotionModel, content, tokenizer)
    jdb.addEntry(user, content, emotion, datetime.datetime.now())

    #Check for any content that could indicate thoughts of self harm or suicide
    if checkContent(content):
        return "Entry contains sensitive content."
    return "Entry added successfully!"


@server.route('/logout', methods=['GET'])
def logout():
    '''request handler, called when the user logsout, clears sessions and returns user to login page.'''
    session.clear()
    return redirect("http://localhost:3000/")

@server.route('/fetch_entries', methods=['POST'])
def fetch_entries():
    '''request handler to grab all of a users entries from the entries table. Returns them in json format to the front end.

    Used when a user navigats to the view entires page of the app. '''

    #Check if the user is logged in
    loggedIn = checkSession()
    if not loggedIn:
        return "You are not logged in!"

    #fetch the signed in user's entries
    user = session.get('user')
    entries = jdb.fetchEntries(user)

    if not entries:
        return "No entries found!"

    #iterate through the fetched entries and seperate the individual items in the sublist by entry, emotion, and datetime
    for i in range(len(entries)):
        entries[i] = list(entries[i])
        entries[i][3] = txtEmotionModel.getMap()[entries[i][3]]
        entries[i][4] = entries[i][4].strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(entries)

@server.route('/delete_entry', methods=['POST'])
def delete_entry():
    '''request handler called when a user wants to delete an entry.

    If the entry does not exist, return error message. If entry does exist, return success message.'''

    #Check if user is logged in
    loggedIn = checkSession()
    if not loggedIn:
        return "You are not logged in!"

    user = session.get('user')
    entry_id = request.form.get('entryID')

    #check if there is actually an entry to delete
    if not entry_id:
        return "No entry selected!"

    jdb.deleteEntry(user, entry_id)
    return "Entry deleted successfully!"


def checkContent(content):
    '''Helper function that checks journal entries for any words that indicate self harm or suicidal ideation.

    Parameters:
        Content (str): the text of the submitted journal entry

    Returns:
        True if one of the keywords is found in the entry
        False if no keywords are found in the entry'''
    keywords = ['suicide', 'end my life', 'ending my life', 'kill myself', 'self harm']
    if any(keyword in content.lower() for keyword in keywords):
        return True
    return False

def checkSession():
    '''Helper function to check if the user is logged in, returning a boolean value

    Parameters:
        None:

    Returns:
        True if user is signed in
        False if user is not logged in'''
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

        #grab the date of the most recent entry
        lastDate = entries[-1][-1]

        timeDelta = currentTime - lastDate

        #convert the timeDelta value from seconds to hours
        hoursDif = timeDelta.total_seconds() / 3600


        if hoursDif < 24:
            displayTime = hrsRemainingToHMS(hoursDif)
            return jsonify({"result": "False", "hours": hoursDif})
        else:
            return jsonify({"result": "True", "hours": hoursDif})


    except Exception as e:
        print("no go partner", e)

def changePWD(username, newPWD):
    '''helper function to change a user's password

    Parameters:
        username (str): the user whose password will be changed
        newPWD (str): the new password for the user

    Returns:
        None
    '''
    hashed = generate_password_hash(newPWD)
    value = jdb.changePassword(username, hashed)
    print(value)

    if not value:
        print("Password could not be changed")

    else:
        print("Password successfully changed!")


def hrsRemainingToHMS(hours):
    '''converts total hours to a hrs:mins:secs format

    Parameters:
        hours (string): An amount of hours

    Returns:
        a string in the format hrs:mins:secs'''
    remaining = 24 - int(hours)

    totalSeconds = remaining * 3600

    hrs = math.floor(totalSeconds / 3600)
    mins = math.floor((totalSeconds % 3600) / 60)
    secs =  totalSeconds% 60

    return f"{hrs}:{mins}:{secs}"

def fmtSeconstoHMS(seconds):
    '''converts total seconds to hrs:mins:secs format'''
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
    '''server route used to handle the live countdown on the landing page of how much time is left before the user can submit another entry.

    Returns:
        A response object with the streamTime() function inside to pass to the front end.'''
    user = session.get('user')
    status, timeSince = checkTimeLeft(user)
    def streamTime():
        '''live countdown of how much time is left until the user can submit another entry, yielding the result each second, before repeating.

        Parameters:
            None

        Returns:
            Yields the remaining time left inside a json payload'''

        try:
            rem = int(timeSince)

            while rem > 0:
                #Creating a payload to send data to the front end
                payload = {
                    "value": fmtSeconstoHMS(rem),
                    "remaining":rem,
                    "allowed": status == "True"
                }
                yield f"data: {json.dumps(payload)}\n\n"

                #Only decrement the timer ever second
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