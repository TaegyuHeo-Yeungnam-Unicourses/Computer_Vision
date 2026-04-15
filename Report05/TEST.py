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
2. RGB 영상을 YCrCb로 변환한 뒤 Y 정보에만 3x3 averaging filter를 적용한다.
3. 1번, 5번, 10번 반복 적용한 Y 영상을 각각 원래의 Cb, Cr 정보와 합쳐 RGB 영상으로 복원한다.
4. 원본 RGB 영상과 복원된 RGB 영상의 PSNR을 계산한다.
5. 결과값을 출력하고 이미지를 비교한다.
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

# #시간 측정용 함수(for report4) **report2를 변형함**
# def test_for_report4(func_restore, func_equalize, img):
#     start_time = time.time()
#     y, img_restored = func_restore(img)
#     y_equalized = func_equalize(y)
#     img_equalized = restoring_by_equalization(y_equalized, img)
#     end_time = time.time()
#     return (end_time - start_time, img_restored, img_equalized)

#시간 측정용 함수(for report6) **report4를 변형함**
def test_for_report6(func_restore, func_filter, img, repeat_count):
    start_time = time.time()
    y, img_restored = func_restore(img)
    y_filtered = repeat_averaging_filter(func_filter, y, repeat_count)
    img_filtered = restoring_by_filtering(y_filtered, img)
    end_time = time.time()
    return (end_time - start_time, img_restored, img_filtered)

#평균 필터 함수
def averaging_filter(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    filtered_img = cv.filter2D(img, -1, kernel)
    return filtered_img

#평균 필터를 여러 번 적용하는 함수
def repeat_averaging_filter(func_filter, img, repeat_count, kernel_size=3):
    filtered_img = img.copy()
    for _ in range(repeat_count):
        filtered_img = func_filter(filtered_img, kernel_size)
    return filtered_img

#복원용 함수
def restoring_by_filtering(y_filtered, img):
    y_filtered = np.clip(y_filtered, 0, 255).astype(np.uint8)

    # 원본(BGR)에서 YCrCb로 변환 후 Y(0번 채널)만 교체하고 다시 BGR로 복원
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    img_ycrcb[:, :, 0] = y_filtered
    img_filtered = cv.cvtColor(img_ycrcb, cv.COLOR_YCrCb2BGR)
    return img_filtered

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

def show_imgs_by_grid_single_window(imgs, titles, tile_size=(420, 300), window_title='grid', grid=(2, 2)):
    cols, rows = grid
    tile_w, tile_h = tile_size

    slots = [(r, c) for r in range(rows) for c in range(cols)]
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

repeat_counts = [1, 5, 10]
results = []

#변환, filtering, 복원
for repeat_count in repeat_counts:
    elapsed_time, img_restored, img_filtered = test_for_report6(opencv_function_restoring, averaging_filter, img, repeat_count)
    _, psnr_value = test_for_report3(psnr_opencv, img, img_filtered)
    results.append((repeat_count, elapsed_time, img_restored, img_filtered, psnr_value))

#값 출력
for repeat_count, elapsed_time, _, _, psnr_value in results:
    print(f"PSNR implrement_{repeat_count} : {psnr_value:.4f}")
    #print(f"Time ({repeat_count} times filtering): {elapsed_time:.4f}")

#이미지 출력
show_imgs_by_grid_single_window(
    [img] + [result[3] for result in results],
    ['Original', '1 time', '5 times', '10 times'],
    window_title='report06_result'
)