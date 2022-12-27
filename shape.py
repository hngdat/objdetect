import random
import cv2 as cv
import numpy as np

# ======================================
# Shape, Line, Drawing
img = np.zeros((512, 512, 3), np.uint8)
pts = (random.randint(0, 512), random.randint(0, 512))
while (True):
    nextpts = (random.randint(0, 512), random.randint(0, 512))
    cv.line(img, pts, nextpts, (random.randint(0, 255),
            random.randint(0, 255), random.randint(0, 255)), 3)
    cv.imshow("Img", img)
    pts = nextpts

    cv.waitKey(1)
