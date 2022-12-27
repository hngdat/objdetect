import cv2 as cv
import numpy as np

# ======================================================
# Car Detection
carCascade = cv.CascadeClassifier("lib/cars.xml")

img = cv.imread("images/car.jpg")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cars = carCascade.detectMultiScale(imgGray, 1.1, 1)
count = 0
for (x, y, w, h) in cars:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    count = count + 1

cv.putText(img, '', 0,
           cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)

cv.imshow("IMG", img)
cv.waitKey(0)

cap = cv.VideoCapture("videos/sample_01.mp4")

while True:
    sucess, img = cap.read()
    if (type(img) == type(None)):
        break
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cars = carCascade.detectMultiScale(imgGray, 1.2, 1)

    for (x, y, w, h) in cars:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("CAM", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
