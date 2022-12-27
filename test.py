import random
import cv2 as cv
import numpy as np

# Images import
# img = cv.imread("images/image2.jpg")
# cv.putText(
#     img=img,
#     text=f"opencv version: {cv.__version__}",
#     org=(30, 40),
#     fontFace=cv.FONT_HERSHEY_PLAIN,
#     fontScale=1.5,
#     color=(0, 255, 0),
#     thickness=1,
#     lineType=cv.LINE_AA,
# )
# cv.imshow("Title", img)
# cv.waitKey(0),
# cv.destroyAllWindows()


# ====================================
# Video import
# cap = cv.VideoCapture("videos/sample_01.mp4")
# while True:
#     success, img = cap.read()
#     cv.imshow("Video", img)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break


# ====================================
# Webcam import
# cap = cv.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 100)
# while True:
#     success, img = cap.read()
#     cv.imshow("Video", img)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# ====================================
#   Img Contribution
# img = cv.imread("images/image.jpg")
# kernel = np.ones((5, 5), np.uint8)
# imgBlur = cv.GaussianBlur(img, (7, 7), 0)
# imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# imgCanny = cv.Canny(img, 150, 200)
# imgDialation = cv.dilate(imgCanny, kernel, iterations=1)
# imgEroded = cv.erode(imgDialation, kernel, iterations=1)
# imgResize = cv.resize(img, (1000, 700))
# imgCropped = imgResize[350:550, 350:550]
# print(img.shape)
# cv.imshow("Img", img)
# cv.imshow("Img Blur", imgBlur)
# cv.imshow("Img Gray", imgGray)
# cv.imshow("Img Canny", imgCanny)
# cv.imshow("Img Dialation", imgDialation)
# cv.imshow("Img Eroded", imgEroded)
# cv.imshow("Img Resize", imgResize)
# cv.imshow("Img Cropped", imgCropped)
# cv.waitKey(0)

# ======================================
# Shape, Line, Drawing
# img = np.zeros((512, 512, 3), np.uint8)
# pts = (random.randint(0, 512), random.randint(0, 512))
# while (True):
#     nextpts = (random.randint(0, 512), random.randint(0, 512))
#     cv.line(img, pts, nextpts, (random.randint(0, 255),
#             random.randint(0, 255), random.randint(0, 255)), 3)
#     cv.imshow("Img", img)
#     pts = nextpts

#     cv.waitKey(1000)

# ======================================
# Face Detecting
# faceCascade = cv.CascadeClassifier("lib/haarcascade_frontalface_default.xml")
# img = cv.imread("images/image.jpg")
# imgResize = cv.resize(img, (1000, 700))
# imgGray = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)
# cap = cv.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 100)
# faces = faceCascade.detectMultiScale(imgGray, 1.1, 1)
# for (x, y, w, h) in faces:
#     cv.rectangle(imgResize, (x, y), (x+w, y+h), (0, 255, 0), 2)
# # cv.imshow("Img", imgResize)
# # cv.waitKey(0)
# while True:
#     success, imgcap = cap.read()
#     capGray = cv.cvtColor(imgcap, cv.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(capGray, 1.1, 4)
#     for (x, y, w, h) in faces:
#         cv.rectangle(imgcap, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     cv.imshow("CAM", imgcap)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# ======================================================
# Car Detection
# carCascade = cv.CascadeClassifier("lib/cars.xml")

# img = cv.imread("images/car.jpg")
# imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# cars = carCascade.detectMultiScale(imgGray, 1.1, 1)
# count = 0
# for (x, y, w, h) in cars:
#     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     count = count + 1

# cv.putText(img, , 0,
#            cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)

# cv.imshow("IMG", img)
# cv.waitKey(0)

# cap = cv.VideoCapture("videos/sample_01.mp4")

# while True:
#     sucess, img = cap.read()
#     if (type(img) == type(None)):
#         break
#     imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     cars = carCascade.detectMultiScale(imgGray, 1.2, 1)

#     for (x, y, w, h) in cars:
#         cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     cv.imshow("CAM", img)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# =============================================
# Yolov Test

cap = cv.VideoCapture('R295.mp4')
img = cv.imread("images/image.jpg")
# img = cv.resize(img, (1000, 720))
whT = 320
cfdsThreshold = 0.5
nmsThreshold = 0.3

classesPath = 'coco.names'
className = []

with open(classesPath, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')


def findObject(outputs, img):

    wT, hT, cT = img.shape
    boundingBox = []
    objClasses = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            objClass = np.argmax(scores)
            confidence = scores[objClass]
            if confidence > cfdsThreshold:
                w, h = int(detection[2] * hT), int(detection[3] * wT)
                x, y = int((detection[0]*hT)-w/2), int((detection[1]*wT)-h/2)
                boundingBox.append([x, y, w, h])
                objClasses.append(objClass)
                confidences.append(float(confidence))

    indices = cv.dnn.NMSBoxes(boundingBox, confidences,
                              cfdsThreshold, nmsThreshold)

    cv.putText(img,
               text=f"Object count: {len(indices)}",
               org=(30, 40),
               fontFace=cv.FONT_HERSHEY_PLAIN,
               fontScale=1.5,
               color=(0, 0, 255),
               thickness=1,
               lineType=cv.LINE_AA,)

    # for x, y, w, h in boundingBox:
    #     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for i in indices:
        if objClasses[i] > 0 & objClasses[i] < 10:
            box = boundingBox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.putText(img,
                       text=f"{className[objClasses[i]].upper()} {int(confidences[i]*100)}%",
                       org=(x, y-10),
                       fontFace=cv.FONT_HERSHEY_PLAIN,
                       fontScale=1.5,
                       color=(255, 0, 0),
                       thickness=1,
                       lineType=cv.LINE_AA,)

# print(className)
# print(len(className))


modelConfig = 'yolov3.cfg'
modelweights = 'yolov3.weights'

net = cv.dnn.readNetFromDarknet(modelConfig, modelweights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

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
    blobImg = cv.dnn.blobFromImage(
        imgCap, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blobImg)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObject(outputs, imgCap)

    cv.imshow("Images", imgCap)
    cv.waitKey(1)
