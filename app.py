import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from keras.models import load_model

app = Flask(__name__)
socketio = SocketIO(app)

frameWidth = 640         
frameHeight = 480
brightness = 180
threshold = 0.75       
font = cv2.FONT_HERSHEY_SIMPLEX

model = load_model("model1.h5")  

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    class_names = {
        0: 'Alto',
        1: 'Reductor velocidad',
        2: 'Cruce',
        3: 'Direccion prohibida',
        4: 'Girar derecha',
        5: 'Girar izquierda',
        6: 'Incorporacion',
        7: 'No estacionarse',
        8: 'Paso peatonal',
        9: 'Velocidad maxima 10km/h',
        10: 'Zona escolar'
    }
    return class_names.get(classNo, "Unknown")

@socketio.on('image')
def handle_image(image):
    try:
        # Procesar imagen
        nparr = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Guardar la imagen para verificar
        cv2.imwrite('received_image.jpg', img)

        img = cv2.resize(img, (64, 64))
        img = preprocessing(img)
        img = img.reshape(1, 64, 64, 1)
        
        # Predecir imagen
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)
        className = getClassName(classIndex)
        
        emit('classification', {'class': className, 'probability': round(probabilityValue * 100, 2)})
    except Exception as e:
        print("Error:", e)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
