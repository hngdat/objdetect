import cv2 as cv
import numpy as np

# ======================================
# Face Detecting
faceCascade = cv.CascadeClassifier("lib/haarcascade_frontalface_default.xml")
img = cv.imread("images/image.jpg")
imgResize = cv.resize(img, (1000, 700))
imgGray = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
faces = faceCascade.detectMultiScale(imgGray, 1.1, 1)
for (x, y, w, h) in faces:
    cv.rectangle(imgResize, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv.imshow("Img", imgResize)
# cv.waitKey(0)
while True:
    success, imgcap = cap.read()
    capGray = cv.cvtColor(imgcap, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(capGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(imgcap, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow("CAM", imgcap)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
