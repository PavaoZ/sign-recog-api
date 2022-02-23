import cv2
from flask import (Flask, jsonify, request)
from matplotlib.pyplot import imshow
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import PIL
from PIL import ImageFilter
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from base64 import b64decode
import cv2

# We change here the origin
model = load_model(
    'C:/Users/User/Desktop/Fakultet/Fifth year/First semester/MPVI/Project/sign-recog-api/model.h5')

# net = Detector(
#     bytes("yolov3.cfg ", encoding=" utf-8 "
#           ),
#     bytes(" yolov3_30000.weights ",
#           encoding=" utf-8 "),
#     0,
#     bytes("obj.data ", encoding=" utf-8 "))

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)
api = Api(app)


@app.route("/process-image", methods=["POST"])
def process_image():
    bytesOfImage = request.get_data()
    with open('image.jpeg', 'wb') as out:
        out.write(bytesOfImage)

    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3_30000.weights'

    labelsPath = 'letters.names'
    labels = open(labelsPath).read().strip().split('\n')
    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    image = cv2.imread('image.jpeg')
    (H, W) = image.shape[:2]

    # Determine output layer names
    layerName = net.getLayerNames()

    layerName = [layerName[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(
        boxes, confidences, confidenceThreshold, NMSThreshold)

    outputs = {}
    if(len(detectionNMS) > 0):
        outputs['detections'] = {}
        outputs['detections']['labels'] = []
        for i in detectionNMS.flatten():
            detection = {}
            detection['Label'] = labels[classIDs[i]]
            detection['confidence'] = confidences[i]
            outputs['detections']['labels'].append(detection)
    else:
        outputs['detections'] = {}
        outputs['detections']['labels'] = []
        detection = {}
        detection['Label'] = 'No object detected'
        detection['confidence'] = '0'
        outputs['detections']['labels'].append(detection)

    return {
        "result": outputs
    }


@app.route('/')
def index_page():
    return 'This is the font recognition API!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
