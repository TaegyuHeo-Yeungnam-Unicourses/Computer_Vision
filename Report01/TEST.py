import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

imgfile = 'window.jpg'

# OpenCV로 이미지를 읽어옵니다.
img = cv.imread(imgfile, cv.IMREAD_COLOR)

# 이미지를 성공적으로 읽었는지 확인합니다.
if img is None:
    print("이미지 파일을 읽을 수 없습니다. 파일 경로를 확인하세요.")
else:
    # OpenCV는 BGR(Blue, Green, Red) 순서로 이미지를 읽습니다.
    # Matplotlib은 RGB(Red, Green, Blue) 순서를 사용하므로 변환이 필요합니다.
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Matplotlib를 사용하여 이미지를 표시합니다.
    plt.imshow(img_rgb)
    
    # 이미지 창에 제목을 추가합니다.
    plt.title('Image using Matplotlib')
    
    # 이미지를 파일로 저장합니다.
    plt.savefig('output_image.png') # <--- 이 부분을 추가합니다.
    
    print("이미지가 'output_image.png' 파일로 저장되었습니다.")


