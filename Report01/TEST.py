import numpy as np
import cv2 as cv
import matplotlib as plt

imgfile = 'window.jpg'
img = cv.imread(imgfile, cv.IMREAD_COLOR)

import cv2 as cv
from matplotlib import pyplot as plt

imgfile = 'window.jpg'
img = cv.imread(imgfile, cv.IMREAD_COLOR)

if img is None:
    print("이미지를 불러올 수 없습니다.")
else:
    # OpenCV는 BGR, matplotlib은 RGB 사용 → 변환 필요
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# cv.imshow('img', img)
# cv.waitKey(0)