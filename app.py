# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from flask import Flask, flash, request, Response, redirect, url_for, render_template
from werkzeug.utils import secure_filename
# from werkzeug import secure_filename
import jsonpickle
#import binascii
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image
import pyrebase


config = {

    "apiKey": "AIzaSyCzEM-a7piep6yfNwaZaRSA5NiDzYDRRO8",
    "authDomain": "object-detection-yolo-cdf10.firebaseapp.com",
    "databaseURL": "https://object-detection-yolo-cdf10.firebaseio.com",
    "projectId": "object-detection-yolo-cdf10",
    "storageBucket": "object-detection-yolo-cdf10.appspot.com",
    "messagingSenderId": "940165302494",
    "appId": "1:940165302494:web:edffa29d72752bbd1d1a4f",
    "measurementId": "G-DY7RH14R5P"


}

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()

# construct the argument parse and parse the arguments

confthres = 0.5
nmsthres = 0.4
yolo_path = './'

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath



def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def detect_it(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            print(boxes)
            print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image

labelsPath="yolo_v3/coco.names"
cfgpath="yolo_v3/yolov3.cfg"
wpath="yolo_v3/yolov3.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)

# Initialize the Flask application
app = Flask(__name__)
app.config["IMG_UPLOADS"] = "/c/Users/hp/Documents/6thSemester/MinorProject/flask_docker"
app.config["ALLOWED_EXTENSIONS"] = ["png", "jpg", "jpeg"]

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# route http posts to this method
@app.route("/api", methods=["GET", "POST"])
def main():
    
    if request.method == "POST":
        file = request.files["file_input"]
      

        if "file_input" not in request.files:
            flash("No file part")
            return redirect(request.url)

        if file.filename == " ":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img = request.files["file_input"].read()
            # img.save(os.path.join(app.config["IMG_UPLOADS"], filename))
            img = Image.open(io.BytesIO(img))
            npimg=np.array(img)
            image=npimg.copy()
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            res=detect_it(image,nets,Lables,Colors)    
            
            image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
            np_img=Image.fromarray(image)
            img_encoded=image_to_byte_array(np_img)
            mimetype="image/jpeg"
            # image.save(os.path.join(app.config["IMAGE_UPLOADS"], image))
            storage.child("images/object_detected.jpg").put(img_encoded)
            
            # print(storage.child("banner.jpg").get_url(None))
            # path = 'static/images'
            # cv2.imwrite(os.path.join(path , "object.jpg"), res)
            # cv2.imwrite("object.jpg", res)            
            # img_url = os.path.join(app.config["IMG_UPLOADS"], "object.jpg")
            # return render_template("result.html", img_url=img_url)
            links = storage.child("images/object_detected.jpg").get_url(None)
            return render_template("result.html", l = links)
            # imageF = Image.open(io.BytesIO(img_encoded))
            
            # return render_template("result.html", img_encoded=imageF)
    return render_template("index.html")
    # return Response(response=img_encoded, status=200,mimetype="image/jpeg")

    # start flask app
if __name__ == '__main__':
    app.secret_key = "key_key"
    app.run(debug=True, host='0.0.0.0')



