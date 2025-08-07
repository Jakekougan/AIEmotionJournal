import requests as req

import torch
from flask import Flask, request, url_for, redirect
import requests as req
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return "I am a simple Flask app!"


@app.route('/data', methods=['GET'])
def get_data():
    data = request.args.get('data', 'No data provided')
    # Here you can process the data as needed
    return f"Here is your data! {data}"