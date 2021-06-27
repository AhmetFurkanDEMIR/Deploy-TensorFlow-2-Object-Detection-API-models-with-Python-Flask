from sys import stdout
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import os

# The processed image is created by interfering with the numpy array from the outside.
from modelConfig import main

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = os.urandom(16)
app.config['DEBUG'] = True
socketio = SocketIO(app)


@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    image_data = input

    img_data = base64.b64decode(image_data)
    nparr = np.frombuffer(img_data,np.uint8) 
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # In the "main" function, the image is processed.
    img_np = main(img_np)

    retval, buffer = cv2.imencode('.jpg', img_np)
    pic_str = base64.b64encode(buffer)
    pic_str = pic_str.decode()

    image_data = "data:image/jpeg;base64," + pic_str

    emit('out-image-event', {'image_data': image_data}, namespace='/test')


@app.route('/')
def index():

    return render_template('index.html')



if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=443, ssl_context=('cert.pem', 'key.pem'))
