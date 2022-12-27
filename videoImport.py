import cv2 as cv
import numpy as np

# ====================================
# Video import
cap = cv.VideoCapture("videos/sample_01.mp4")
while True:
    success, img = cap.read()
    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
