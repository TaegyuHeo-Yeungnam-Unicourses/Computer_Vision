# Report04/TEST.py 오류 분석 보고서 (2차)

## 1. 현재 발생한 오류
스크립트 실행 시 다음과 같은 OpenCV 오류가 발생하였습니다:
```
cv2.error: OpenCV(4.13.0) ... error: (-215:Assertion failed) _src1.type() == _src2.type() in function 'PSNR'
```

## 2. 원인 분석
오류는 `psnr_opencv` 함수 호출 시 발생하였습니다:
```python
# [TEST.py](TEST.py#L173)
_, psnr_opencv_value2 = test_for_report3(psnr_opencv, img, opencv_img_equalized)
```
- **`img`**: `cv.imread`로 읽어온 **3채널 BGR 컬러 이미지**입니다.
- **`opencv_img_equalized`**: `test_for_report4`에서 `opencv_equalize_histogram(y)`를 통해 생성된 **1채널 그레이스케일 이미지**입니다.

`cv.PSNR()` 함수는 두 입력 이미지의 타입(채널 수 및 데이터 타입)이 동일해야 합니다. 하지만 원본 컬러 이미지(3채널)와 평활화된 이미지(1채널)를 직접 비교하려 했기 때문에 `_src1.type() == _src2.type()` 단언문 실패가 발생한 것입니다.

## 3. 수정 제안
PSNR을 올바르게 계산하려면 비교 대상을 일치시켜야 합니다:
- 원본 이미지의 Y 채널(휘도)과 평활화된 Y 채널을 비교하거나,
- 평활화된 1채널 이미지를 다시 3채널(BGR)로 복원한 후 원본과 비교해야 합니다.
