import cv2 as cv
import numpy as np

# ====================================
#   Img Contribution
img = cv.imread("images/image.jpg")
kernel = np.ones((5, 5), np.uint8)
imgBlur = cv.GaussianBlur(img, (7, 7), 0)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgCanny = cv.Canny(img, 150, 200)
imgDialation = cv.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv.erode(imgDialation, kernel, iterations=1)
imgResize = cv.resize(img, (1000, 700))
imgCropped = imgResize[350:550, 350:550]
print(img.shape)
cv.imshow("Img", img)
cv.imshow("Img Blur", imgBlur)
cv.imshow("Img Gray", imgGray)
cv.imshow("Img Canny", imgCanny)
cv.imshow("Img Dialation", imgDialation)
cv.imshow("Img Eroded", imgEroded)
cv.imshow("Img Resize", imgResize)
cv.imshow("Img Cropped", imgCropped)
cv.waitKey(0)
