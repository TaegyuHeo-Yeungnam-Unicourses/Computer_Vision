import numpy as np
import cv2 as cv
import time
from pathlib import Path
import os
import platform

"""
코드 설명
1. 특정 이미지 파일을 imread로 읽어와서 BGR 형식의 3채널 이미지로 저장한다.
2. opencv와 수작업 변환시 각각의 시간차를 측정하기 위해 test wrapper 함수에 각각의 함수와 이미지를 전달한다.
3. 반환된 결과값(시간, Y709, 복원된 이미지)을 각각 변수에 저장한다.
4. 측정된 시간값을 출력한다.
5. 변환된 이미지들을 격자의 형태로 적절하게 출력한다.
"""

#시간 측정용 함수
def test(func, img):
    start_time = time.time()
    y, img_restored = func(img)
    end_time = time.time()
    return (end_time - start_time, y, img_restored)

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

#변환된 이미지들을 적절하게 보여주는 함수
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

#변환 및 시간 측정
opencv_time, opencv_Y709, opencv_img_restored = test(opencv_function, img)
mannual_time, mannual_Y709, mannual_img_restored = test(mannual_function, img)

#시간 출력
print(f"opencv_time : {opencv_time:.4f}")
print(f"manual_time : {mannual_time:.4f}")

#변환 이미지 출력
imgs = [img, opencv_Y709, opencv_img_restored, mannual_Y709, mannual_img_restored]
titles = ['original', 'grayscale_opencv', 'OpenCV Restored', 'grayscale_manual', 'Manual Restored']

# WSL에서는 창 위치 배치가 OS에 따라 달라질 수 있어, 단일 창 그리드로 표시
if _is_wsl():
    show_imgs_by_grid_single_window(imgs, titles, tile_size=(420, 300), window_title='grid')
else:
    show_imgs_by_grid(imgs, titles)