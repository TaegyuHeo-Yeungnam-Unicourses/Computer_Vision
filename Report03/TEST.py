import numpy as np
import cv2 as cv
import time
from pathlib import Path
import os
import platform
import math

"""
코드 설명
1. 특정 이미지 파일을 imread로 읽어와서 BGR 형식의 3채널 이미지로 저장한다.
2. opencv와 수작업 변환시 각각의 시간차를 측정하기 위해 test_for_report2 wrapper 함수에 각각의 함수와 이미지를 전달한다.
3. 반환된 결과값(시간, Y709, 복원된 이미지)을 각각 변수에 저장한다.
4. 반환된 결과값을 바탕으로 오차율(PSNR)을 계산하여 출력한다.
"""

#시간 측정용 함수(for report2)
def test_for_report2(func, img):
    start_time = time.time()
    y, img_restored = func(img)
    end_time = time.time()
    return (end_time - start_time, y, img_restored)

#시간 측정용 함수(for report3)
def test_for_report3(func, img1, img2):
    start_time = time.time()
    psnr_value = func(img1, img2)
    end_time = time.time()
    return (end_time - start_time, psnr_value)

#opencv 변환용 함수
def opencv_function(img):
    img_YCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    Y709 = img_YCbCr[:, :, 0]
    img_restored = cv.cvtColor(img_YCbCr, cv.COLOR_YCrCb2BGR)
    return (Y709, img_restored)

#수작업 변환용 함수
def mannual_function(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    Y709 = 0.183 * R + 0.614 * G + 0.062 * B + 16
    Cb = - 0.101 * R - 0.338 * G + 0.439 * B + 128
    Cr = 0.439 * R - 0.399 * G - 0.040 * B + 128

    return_Y709 = np.clip(Y709, 0, 255).astype(np.uint8)

    B = 1.164 * (Y709 - 16) + 2.115 * (Cb - 128)
    G = 1.164 * (Y709 - 16) - 0.534 * (Cr - 128) - 0.213 * (Cb - 128)
    R = 1.164 * (Y709 - 16) + 1.793 * (Cr - 128)

    B = np.clip(B, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    R = np.clip(R, 0, 255).astype(np.uint8)

    img_restored= cv.merge((B, G, R))
    return (return_Y709, img_restored)


    return bool(os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP') or 'microsoft' in platform.release().lower())

#오차율 측정 함수(opencv)
def psnr_opencv(img1, img2):
    psnr_value = cv.PSNR(img1, img2)
    return psnr_value

#오차율 측정 함수(수작업)
def psnr_mannual(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

#이미지 읽어오기 (스크립트 위치 기준으로 안전하게 로드)
imgfile = Path(__file__).resolve().parent / 'window.jpg'
img = cv.imread(str(imgfile), cv.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError(f"Failed to read image: {imgfile}")

#변환 및 복원
opencv_time, opencv_Y709, opencv_img_restored = test_for_report2(opencv_function, img)
mannual_time, mannual_Y709, mannual_img_restored = test_for_report2(mannual_function, img)

#오차율 계산 및 시간 측정
opencv_time, psnr_opencv_value = test_for_report3(psnr_opencv, img, opencv_img_restored)
mannual_time, psnr_mannual_value = test_for_report3(psnr_mannual, img, mannual_img_restored)

#값 출력
print(f"PSNR (OpenCV): {psnr_opencv_value:.4f}")
print(f"PSNR (Manual): {psnr_mannual_value:.4f}")

print(f"Time (OpenCV): {opencv_time:.4f}")
print(f"Time (Manual): {mannual_time:.4f}")
