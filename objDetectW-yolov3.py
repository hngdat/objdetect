import cv2 as cv
import numpy as np


# =============================================
# Yolov Test

# Videos Test
cap = cv.VideoCapture('R295.mp4')
# Images Test
img = cv.imread("car.jpg")

# img = cv.resize(img, (1000, 720)) # Just for testing, nothing special

# Width and Heigth size for blob image
whT = 320
# Confidence Threshold: Save objects with higher confidence score than this value
cfdsThreshold = 0.5
# Non-maximum Suppression Threshold: In order to illiminate boxes of the same objects base on the range of this value
# Suggestion for nmsThreshold value > 0.2 and < 0.8
nmsThreshold = 0.3

classesPath = 'coco.names'
className = []

# Input all classes from coco file to className parameter
with open(classesPath, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')


#  Function to find all objects and draw bounding boxes
def findObject(outputs, img):

    wT, hT, cT = img.shape  # Necessary value to calculate X,Y coordinate and Width, Height
    boundingBox = []
    objClasses = []
    confidences = []

    # For each row
    for output in outputs:
        # For each column
        for detection in output:
            # Get confidence scores of all classes
            # 5 First value of detection are center-X, center-Y, Width, Height
            # and Confidence Score (whether there is an Objects inside the boxes, not exactly classes,
            # basically is the highest confidence score of the rest)
            # The rest are the Confidence Scores for a particular class
            # In order to detect if there is a object, first 5 value is unnecessary
            scores = detection[5:]

            # Get object class (index) with the highest confidence score
            objClass = np.argmax(scores)

            # Get the Confidence Score
            confidence = scores[objClass]

            # Compare Confidence Score with Confidence Threshold value
            # If the Confidence Score is below Confidence Threshold value, then pass this object
            # Else, calculate width, height, top left X and Y coordinate
            # Make sure x,y,w,h are INT value type and follow the pixel value
            if confidence > cfdsThreshold:

                # Width and Height calculate
                w, h = int(detection[2] * hT), int(detection[3] * wT)
                # Top Left X and Y coordinate calculate
                x, y = int((detection[0]*hT)-w/2), int((detection[1]*wT)-h/2)

                # Save value
                boundingBox.append([x, y, w, h])
                # Save Object's Classes
                objClasses.append(objClass)
                # Save Object's Confidence Score
                confidences.append(float(confidence))

    # Illiminate duplicate boxes using Non-maximum Suppression Algorithm
    indices = cv.dnn.NMSBoxes(boundingBox, confidences,
                              cfdsThreshold, nmsThreshold)

    # Show totals Object Founded
    cv.putText(img,
               text=f"Object found: {len(indices)}",
               org=(30, 40),
               fontFace=cv.FONT_HERSHEY_PLAIN,
               fontScale=1.5,
               color=(0, 0, 255),
               thickness=1,
               lineType=cv.LINE_AA,)

    # Draw objects bounding boxes
    drawBoxes(boundingBox, indices, objClasses, confidences, img)


# Draw bounding boxes for objects FUNCTION
def drawBoxes(boundingBoxes, indices, objClasses, confidences, img):

    # indices contain index of the bounding boxes remain after illiminate duplicate boxes
    # Loop throw all the index
    for i in indices:

        # If Object's Class is a vehicle then draw a bounding box for it
        # Else do nothing
        if objClasses[i] > 0 & objClasses[i] < 10:

            # Get box's x,y,w,h
            box = boundingBoxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Draw bounding boxes
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Write Object's Class
            cv.putText(img,
                       text=f"{className[objClasses[i]].upper()} {int(confidences[i]*100)}%",
                       org=(x, y-10),
                       fontFace=cv.FONT_HERSHEY_PLAIN,
                       fontScale=1.5,
                       color=(255, 0, 0),
                       thickness=1,
                       lineType=cv.LINE_AA,)


# Read Yolo model file
modelConfig = 'yolov3.cfg'
modelweights = 'yolov3.weights'
# Setup model, backend and target
net = cv.dnn.readNetFromDarknet(modelConfig, modelweights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)  # Using CPU as main proccessor

# blobImg = cv.dnn.blobFromImage(
#     img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
# net.setInput(blobImg)
# layerNames = net.getLayerNames()
# outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
# outputs = net.forward(outputNames)
# # print(outputs[0][0])
# # print(layerNames)
# # print(len(outputs))
# # print(type(outputs))
# # print(type(outputs[0]))
# # print(outputs[0].shape)
# # print(outputs[1].shape)
# # print(outputs[2].shape)
# # print(outputNames)
# # print(net.getUnconnectedOutLayers())
# findObject(outputs, img)

# cv.imshow("Image", img)
# cv.waitKey(0)

while True:
    success, imgCap = cap.read()
    # In order to help Darknet read image, convert img to blob img (Binary Large Object)
    blobImg = cv.dnn.blobFromImage(
        imgCap, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    # Set input image for Darknet
    net.setInput(blobImg)
    # Get all image Layer Names
    layerNames = net.getLayerNames()
    # Get 3 output layers from layerNames and forward it as output images for Darknet
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    # outputs value can be present like a table
    # Each Row define a single object with its position, width, height and confidence score values
    # Each Column define objects value such as x, y coordinate, width and height,
    # the rest are confidence score for each classes
    # For example:
    #  X  Y  W  H  Object confidence     1   ...   80
    #  |  |  |  |        |               |   ...   |
    # [1][2][3][4]     [0.5]           [0.5] ... [0.0]
    # [5][6][7][8]     [0.9]           [0.5] ... [0.9]
    #                       ...

    # Find ojects and draw bounding boxes
    findObject(outputs, imgCap)

    cv.imshow("Images", imgCap)
    cv.waitKey(1)
