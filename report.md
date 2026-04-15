
# Report05 평균(averaging) 필터 적용 검증 보고서

## 1. 목적
본 보고서는 [Report05/TEST.py](Report05/TEST.py)에서 **평균 필터(averaging filter)**가 의도대로 적용되는지(적용 위치/반복 적용/복원 과정 포함) 코드 흐름을 따라가며 확인한 결과를 정리한다.

## 2. 코드 흐름 요약(필터 적용 지점)

### 2.1 입력 이미지 로드
- `img = cv.imread(..., cv.IMREAD_COLOR)`로 BGR 3채널 이미지를 읽는다.
- `img is None` 체크로 파일 로드 실패를 방지한다.

### 2.2 Y 채널 추출(필터링 대상)
- `opencv_function_restoring(img)`에서
	- `img_YCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)`
	- `Y709 = img_YCbCr[:, :, 0]`
	- 즉, **YCrCb의 Y(휘도) 채널만** 분리하여 반환한다.

### 2.3 평균 필터 정의
- `averaging_filter(img, kernel_size=3)`는
	- `kernel = np.ones((k,k), float32) / (k*k)` 형태의 **정규화된 박스(평균) 커널**을 만들고
	- `cv.filter2D(img, -1, kernel)`로 컨볼루션을 수행한다.
- 커널 합이 1이므로(정규화) **밝기(DC 성분)를 유지하는 평균 필터**가 맞다.

### 2.4 반복 평균 필터 적용(핵심)
- `repeat_averaging_filter(func_filter, img, repeat_count, kernel_size=3)`에서
	- `for _ in range(repeat_count): filtered_img = func_filter(filtered_img, kernel_size)`
	- 즉, **동일한 평균 필터를 repeat_count번 연속 적용**한다.

### 2.5 필터링 결과로 복원
- `restoring_by_filtering(y_filtered, img)`에서
	- `y_filtered`를 `uint8`로 클리핑 후
	- 원본 `img`를 다시 YCrCb로 변환한 뒤 `Y` 채널만 `y_filtered`로 교체
	- `cv.cvtColor(..., cv.COLOR_YCrCb2BGR)`로 BGR 복원
- 따라서 필터는 **색상(Cr, Cb)은 유지**하고 **휘도(Y)만 평활화**하는 구조다.

## 3. “평균 필터가 잘 적용되는가?”에 대한 결론

### 3.1 적용 대상이 올바른가?
예. 평균 필터는 BGR 전체가 아니라 `Y` 채널(`Y709`)에만 적용된다. 일반적으로 노이즈/블러 분석에서 “밝기 성분만” 처리하는 접근은 타당하다(색 왜곡을 줄임).

### 3.2 평균 커널이 올바른가?
예. `k×k` 커널을 모두 `1/(k*k)`로 설정하여 **표준적인 박스 평균 필터**가 맞다.

### 3.3 반복 적용이 올바른가?
예. `repeat_count`만큼 동일 필터를 누적 적용한다. 반복 적용은 블러가 점점 강해지는 효과가 있으며, 반복 횟수가 증가할수록 결과가 더 부드러워지는 것이 정상 동작이다.

### 3.4 데이터 타입/정확도 관점
- 현재 `Y709`는 `uint8`이고, `cv.filter2D(..., ddepth=-1)`도 결과를 `uint8`로 유지한다.
- 이 경우 각 반복 단계마다 정수 양자화/클리핑이 들어가므로, 엄밀한 의미의 “실수 연산 누적”과는 조금 다를 수 있다.
- 하지만 “평균 필터가 적용되었다/반복 적용되었다”는 기능적 관점에서는 **문제 없이 올바르게 적용**된다.

## 4. PSNR 계산 흐름 점검(부가 확인)
- `img_filtered`는 필터링된 Y를 넣어 복원한 BGR 이미지이다.
- PSNR은 `psnr_opencv(img, img_filtered)`로 **원본 대비 필터링 결과**를 평가한다.
- 필터 반복 횟수가 늘수록 블러가 증가하여 일반적으로 PSNR이 낮아질 가능성이 높다(이미지/내용에 따라 예외 가능).

## 5. 개선 제안(선택 사항)
기능은 올바르지만, 수치적 안정성/일관성을 위해 아래 개선을 고려할 수 있다.

1) 반복 필터링 시 `float32`로 누적
- 예: 첫 입력 `Y`를 `float32`로 변환하고 `filter2D` 출력도 `CV_32F`로 유지한 뒤 마지막에만 `uint8`로 변환하면 반복 과정의 양자화 영향을 줄일 수 있다.

2) 경계 처리 명시
- `cv.filter2D`는 기본 border 정책을 사용한다. 실험 재현성을 위해 `borderType`을 명시하는 것도 좋다.

## 6. 최종 요약
- [Report05/TEST.py](Report05/TEST.py)는 **Y 채널에 3×3 평균 필터를 정의하고, 이를 1/5/10회 반복 적용한 뒤 BGR로 복원**하는 파이프라인을 정확히 구현하고 있다.
- 따라서 “average filter가 잘 적용되고 있는지” 관점에서 **구현은 정상/타당**하다.

