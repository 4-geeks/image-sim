import socketio
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time
import cv2
import os
import datetime
import time
import json

PORT = 8000
HOST = "34.91.63.13"
NAME_SPACE = "/asset"
MESSAGE_FROM_CLIENT = "book_rec"
MESSAGE_TO_CLIENT = "book_rec_results"
SERVR =  'http://%s:%s' % (HOST,PORT)
class dataHandler():
    def __init__(self):
        pass
    
    def convertToBase64(self,image):
            frame = cv2.imencode('.jpg',image)[1].tobytes()
            base64str = base64.b64encode(frame).decode('utf-8')
            return base64str
        
    def convertFromBase64(self,base64str):
        pil_image = Image.open(BytesIO(base64.b64decode(base64str)))
        return np.array(pil_image)

sio = socketio.Client()
handler = dataHandler()
@sio.event
def connect():
    print('connection established')

@sio.on(MESSAGE_TO_CLIENT,namespace=NAME_SPACE)
def get_json(message):
    print(message.keys())
    with open('message.json', 'w') as outfile:
        json.dump(message, outfile)
        
@sio.on('info',namespace=NAME_SPACE)
def get_info(message):
    print(message)
    
@sio.event
def disconnect():
    print('disconnected from server')

if __name__ == "__main__":
    sio.connect(SERVR ,namespaces=NAME_SPACE)
    img = cv2.imread('images/book_cover_14.jpg')
    base64str = handler.convertToBase64(img)
    while True:
        print('message start emit..')
        sio.emit(MESSAGE_FROM_CLIENT,base64str,namespace='/asset')
        print('emitted!')
        time.sleep(5)
    
