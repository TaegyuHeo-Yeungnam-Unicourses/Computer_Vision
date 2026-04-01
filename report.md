# Report04/TEST.py 오류 분석 보고서 (2차)

## 1. 현재 발생한 오류
스크립트 실행 시 다음과 같은 OpenCV 오류가 발생하였습니다:
```
cv2.error: OpenCV(4.13.0) ... error: (-215:Assertion failed) _src1.type() == _src2.type() in function 'PSNR'
```

## 2. 원인 분석
오류는 `psnr_opencv` 함수 호출 시 발생하였습니다:
```python
_, psnr_opencv_value2 = test_for_report3(psnr_opencv, img, opencv_img_equalized)
```
- **`img`**: `cv.imread`로 읽어온 **3채널 BGR 컬러 이미지**입니다.
- **`opencv_img_equalized`**: `test_for_report4`에서 `opencv_equalize_histogram(y)`를 통해 생성된 **1채널 그레이스케일 이미지**입니다.

`cv.PSNR()` 함수는 두 입력 이미지의 타입(채널 수 및 데이터 타입)이 동일해야 합니다. 하지만 원본 컬러 이미지(3채널)와 평활화된 이미지(1채널)를 직접 비교하려 했기 때문에 `_src1.type() == _src2.type()` 단언문 실패가 발생한 것입니다.

## 3. 수정 제안
PSNR을 올바르게 계산하려면 비교 대상을 일치시켜야 합니다:
- 원본 이미지의 Y 채널(휘도)과 평활화된 Y 채널을 비교하거나,
- 평활화된 1채널 이미지를 다시 3채널(BGR)로 복원한 후 원본과 비교해야 합니다.

## 4. 실제 수정 내용
[Report04/TEST.py](Report04/TEST.py)에서 아래 내용을 수정하여 문제를 해결하였습니다.

- `restoring_by_equalization(y_equalized, img)`를 추가/수정하여, 원본 이미지 `img`에서 색상 채널(Cr, Cb)은 유지하고 **Y 채널만 `y_equalized`로 교체**한 뒤 BGR로 복원하도록 변경
	- 올바른 흐름: `BGR → YCrCb → (Y 채널 교체) → BGR`
	- `y_equalized`는 `cv.equalizeHist()` 결과이므로 `uint8`(0~255)로 클리핑/형변환하여 타입 문제를 방지
- `mannual_function_restoring` 내부에 있던 도달 불가능한(실행되지 않는) 코드 조각을 제거하여 혼동을 줄임

## 5. 재실행 결과
수정 후 스크립트는 끝까지 실행되었고 PSNR/시간이 정상적으로 출력되었습니다.

추가로 실행 중 다음과 같은 경고가 출력될 수 있으나( Qt 폰트 디렉터리 관련 ), 계산 로직에는 영향을 주지 않는 **비치명적 경고**입니다:
```
QFontDatabase: Cannot find font directory .../cv2/qt/fonts.
```
