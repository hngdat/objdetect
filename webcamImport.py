import cv2 as cv
import numpy as np

# ====================================
# Webcam import
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
while True:
    success, img = cap.read()
    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
