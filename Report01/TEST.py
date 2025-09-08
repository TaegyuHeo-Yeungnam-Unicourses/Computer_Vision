import numpy as np
import cv2 as cv

imgfile = 'window.jpg'
img = cv.imread(imgfile, cv.IMREAD_COLOR)

cv.imshow('img', img)
cv.waitKey(0)