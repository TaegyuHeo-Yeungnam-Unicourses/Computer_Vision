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
2. 기존의 report2에서 작성한 test_for_report2 함수를 변형하여, 추가로 histogram equalization을 수행하도록 한다.
3. 반환된 결과값(시간, 평활화된 이미지)을 각각 변수에 저장한다.
4. 반환된 결과값을 바탕으로 오차율(PSNR)을 계산한다.
5. 결과값을 출력한다.
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

#시간 측정용 함수(for report2) **report2를 변형함**
def test_for_report4(func_restore, func_equalize, img):
    start_time = time.time()
    y, img_restored = func_restore(img)
    y_equalized = func_equalize(y)
    img_equalized = restoring_by_equalization(y_equalized, img)
    end_time = time.time()
    return (end_time - start_time, img_restored, img_equalized)

#복원용 함수
def restoring_by_equalization(y_equalized, img):
    # cv.equalizeHist() 결과는 8비트 1채널이어야 하므로, 타입을 보장한다.
    y_equalized = np.clip(y_equalized, 0, 255).astype(np.uint8)

    # 원본(BGR)에서 YCrCb로 변환 후 Y(0번 채널)만 교체하고 다시 BGR로 복원
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    img_ycrcb[:, :, 0] = y_equalized
    img_equalized = cv.cvtColor(img_ycrcb, cv.COLOR_YCrCb2BGR)
    return img_equalized

#opencv 변환용 함수
def opencv_function_restoring(img):
    img_YCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    Y709 = img_YCbCr[:, :, 0]
    img_restored = cv.cvtColor(img_YCbCr, cv.COLOR_YCrCb2BGR)
    return (Y709, img_restored)

#수작업 변환용 함수
def mannual_function_restoring(img):
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

#opencv histogram equalization 함수
def opencv_equalize_histogram(y):
    equalized_y = cv.equalizeHist(y)
    return equalized_y

#수작업 histogram equalization 함수
def mannual_equalize_histogram(y):
    total_pixels = y.size
    maximum_pixel_value = 255
    histogram, _ = np.histogram(y.flatten(), bins=256, range=(0, 255))
    cdf = histogram.cumsum()
    method_1_result = ((maximum_pixel_value + 1) / total_pixels) * cdf - 1
    equlized_map = np.round(method_1_result).astype(np.uint8)
    equlized_map = np.clip(equlized_map, 0, maximum_pixel_value).astype(np.uint8)
    y_equalized = equlized_map[y]
    return y_equalized

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

def show_imgs_by_grid(imgs, titles, window_size=(420, 300), start_pos=(0, 0)):
    slots = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    w, h = window_size
    sx, sy = start_pos

    for (row, col), img, title in zip(slots, imgs, titles):
        cv.namedWindow(title, cv.WINDOW_NORMAL)
        cv.imshow(title, img)

        cv.resizeWindow(title, w, h)
        cv.moveWindow(title, sx + (col * w), sy + (row * h))

        cv.waitKey(1)

    cv.waitKey(0)
    cv.destroyAllWindows()

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def show_imgs_by_grid_single_window(imgs,titles,tile_size=(420, 300),window_title='grid'):
    slots = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    cols, rows = 3, 2
    tile_w, tile_h = tile_size

    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for (row, col), img, title in zip(slots, imgs, titles):
        tile = _to_bgr(img)
        tile = cv.resize(tile, (tile_w, tile_h), interpolation=cv.INTER_AREA)
        x0, y0 = col * tile_w, row * tile_h
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        cv.putText(
            canvas,
            str(title),
            (x0 + 10, y0 + 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    cv.namedWindow(window_title, cv.WINDOW_NORMAL)
    cv.imshow(window_title, canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()

def _is_wsl() -> bool:
    return bool(os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP') or 'microsoft' in platform.release().lower())

#이미지 읽어오기 (스크립트 위치 기준으로 안전하게 로드)
imgfile = Path(__file__).resolve().parent / 'window.jpg'
img = cv.imread(str(imgfile), cv.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError(f"Failed to read image: {imgfile}")

#변환 및 복원
opencv_time, opencv_img_restored, opencv_img_equalized = test_for_report4(opencv_function_restoring, opencv_equalize_histogram, img)
mannual_time, mannual_img_restored, mannual_img_equalized = test_for_report4(mannual_function_restoring, mannual_equalize_histogram, img)

#오차율 계산 및 시간 측정
_, psnr_opencv_value = test_for_report3(psnr_opencv, img, opencv_img_restored)
_, psnr_opencv_value2 = test_for_report3(psnr_opencv, img, opencv_img_equalized)
_, psnr_mannual_value = test_for_report3(psnr_opencv, img, mannual_img_equalized)

#이미지 출력
show_imgs_by_grid_single_window(
    [img, opencv_img_restored, opencv_img_equalized, mannual_img_restored, mannual_img_equalized],
    ['Original', 'Restored (OpenCV)', 'Equalized (OpenCV)', 'Restored (Manual)', 'Equalized (Manual)']
)

#값 출력
print(f"PSNR(calculated by opencv) (restored(opencv)): {psnr_opencv_value:.4f}")
print(f"PSNR(calculated by opencv) (restored(opencv) & equalized(opencv)): {psnr_opencv_value2:.4f}")
print(f"PSNR(calculated by opencv) (restored(manual) & equalized(manual)): {psnr_mannual_value:.4f}")
print(f"Time (OpenCV): {opencv_time:.4f}")
print(f"Time (Manual): {mannual_time:.4f}")
