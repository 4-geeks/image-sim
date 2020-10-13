"""
Created on Sun Oct  4 14:41:40 2020

@author: homayoun
"""
from sys import stdout
import logging
from flask import Flask
from flask_socketio import SocketIO

import torchvision.models as models
import torch
from utils import base64_to_pil_image
from search import search

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

PORT = 5001
HOST = "localhost"
NAME_SPACE = "/asset"
MESSAGE_FROM_CLIENT = "book_rec"
MESSAGE_TO_CLIENT = "book_rec_results"
socketio = SocketIO(app,cors_allowed_origins="*")

class simnet():
    def __init__(self):
        pass
    def load(self):
        self.model = torch.nn.Sequential(*list(models.vgg16(pretrained=True).features.modules())[1:14])
        
simNet = simnet()
@socketio.on(MESSAGE_FROM_CLIENT, namespace=NAME_SPACE)
def process_frame(input):
    img = base64_to_pil_image(input)
    name, score = search(img,simNet.model,db='index.json')
    message = {'Name':name,'Score':score}
    print(message)
    socketio.emit(MESSAGE_TO_CLIENT,message,namespace=NAME_SPACE)

@socketio.on('connect', namespace=NAME_SPACE)
def connect():
    simNet.load()
    app.logger.info("client connected")


if __name__ == '__main__':
    socketio.run(app)