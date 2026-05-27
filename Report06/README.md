# YCbCr Y 기반 Canny + Hough 직선/원 검출

이 프로그램은 입력 이미지 또는 동영상에서 YCbCr의 Y 성분을 추출하고, Canny Edge Detection을 적용한 뒤 Hough Transform으로 직선 또는 원을 검출한다.

비전공자가 파이썬을 짧게 배운 뒤에도 읽을 수 있도록 타입 힌트, dataclass, 데코레이터를 사용하지 않았다. 대신 평범한 함수, 클래스, `if`, `for`, 리스트, NumPy, OpenCV 함수 위주로 작성했다.

## 1. 파일 역할

| 파일 | 역할 |
|---|---|
| `main.py` | 프로그램 진입점이다. 설정을 만들고 애플리케이션을 실행한다. |
| `settings.py` | 기본 실행값, 입력, Canny, Hough, 출력 설정을 객체로 나누어 저장한다. |
| `form.py` | 명령행 인자와 선택적인 터미널 입력을 읽어 실행 설정 객체를 만든다. |
| `source.py` | 이미지, 동영상 파일, 웹캠 입력을 같은 방식으로 읽는다. |
| `pipeline.py` | 프레임 하나를 Y 성분 추출, Canny, 도형 검출 순서로 처리한다. |
| `detectors.py` | 직선/원 후보 객체, 점수화, Hough 검출, 결과 그리기를 담당한다. |
| `app.py` | 입력, 처리, 출력을 조립하고 전체 실행 루프를 관리한다. |
| `ui.py` | Canny 결과와 검출 결과를 좌우로 붙이고, 창 표시 또는 파일 저장을 담당한다. |

## 2. 중요 설정 위치

개발자가 자주 바꿀 수 있는 값은 `settings.py`의 설정 객체에 모아 두었다.

- `ProgramSettings.defaults`: 기본 검출 모드, 기본 입력 파일, 웹캠 번호, 출력 폴더, 기본 필터 사용 여부
- `InputSettings`: 상대 입력 경로 기준 폴더, 이미지 확장자, 기본 FPS
- `EdgeSettings`: Gaussian Blur 커널 크기와 Canny 임계값
- `LineSettings`: Hough 직선 검출 값, 표시할 직선 개수, 차선 각도 범위, 아래쪽 직선 가중치
- `CircleSettings`: Hough 원 검출 값, 표시할 원 개수, 원 둘레 에지 지지도 계산 값, 우측 상단 위치 가중치
- `DisplaySettings`: OpenCV 창 대기 시간, 이미지 저장 주기, headless 웹캠 프레임 제한

## 3. 실행 방법

필요 패키지는 다음과 같다.

```bash
python3 -m pip install opencv-python numpy
```

Report06 폴더에서 기본 영상(`test_video.mp4`)으로 실행한다.

```bash
python3 main.py
```

다른 파일을 입력으로 사용할 수 있다.

```bash
python3 main.py --input-type file --source ./road.mp4
python3 main.py --input-type file --source ./road.jpg
```

웹캠을 사용할 때는 다음처럼 실행한다.

```bash
python3 main.py --input-type webcam --source 0
```

원 검출 모드는 다음처럼 선택한다.

```bash
python3 main.py --mode circle --source ./circle_test.mp4
```

터미널에서 직접 값을 입력하려면 `--form`을 붙인다.

```bash
python3 main.py --form
```

## 4. 명령행 옵션

| 옵션 | 설명 |
|---|---|
| `--mode`, `--shape` | `straight` 또는 `circle`을 선택한다. |
| `--input-type` | `file` 또는 `webcam`을 선택한다. |
| `--source` | 파일 경로 또는 웹캠 번호를 지정한다. |
| `--lane-filter` | 직선 모드에서 차선 각도 필터를 `on/off`로 지정한다. |
| `--circle-priority` | 원 모드에서 우측 상단 위치 우선을 `on/off`로 지정한다. |
| `--max-frames` | 처리할 최대 프레임 수를 지정한다. `0`은 제한 없음이다. |
| `--form` | 실행 중 터미널 입력으로 설정을 다시 선택한다. |

## 5. 처리 흐름

1. `main.py`가 `ProgramSettings`와 `UserInputForm`으로 실행 설정을 만든다.
2. `app.py`의 `ShapeDetectionApplication`이 입력, 프레임 처리기, 출력 관리자를 조립한다.
3. `source.py`의 `FrameSource`가 이미지, 동영상, 웹캠 중 하나에서 프레임을 읽는다.
4. `pipeline.py`의 `YChannelEdgeDetector`가 BGR 프레임을 YCrCb로 바꾸고 Y 밝기 성분에 Canny를 적용한다.
5. 직선 모드에서는 `detectors.py`의 `HoughLineDetector`가 `cv2.HoughLinesP()` 결과를 점수화해 상위 직선을 그린다.
6. 원 모드에서는 `detectors.py`의 `HoughCircleDetector`가 `cv2.HoughCircles()` 결과를 에지 지지도 기준으로 점수화해 상위 원을 그린다.
7. `ui.py`의 `ResultDisplayManager`가 Canny 영상과 검출 결과를 좌우로 붙여 표시하거나 `output/` 폴더에 저장한다.

## 6. 코드 작성 기준

- 함수 인자에 `value: str` 같은 타입 힌트를 쓰지 않았다.
- `@dataclass`, `@staticmethod`, `@classmethod`, `@property` 같은 데코레이터를 쓰지 않았다.
- 설정과 후보 정보는 `__init__`이 있는 기본 클래스에 저장했다.
- `main.py`에는 값과 처리 로직을 두지 않고 진입점 책임만 남겼다.
- 입력, 처리, 검출, 출력은 서로 다른 클래스와 파일로 나누었다.
- 후보 목록을 만들 때 어려운 한 줄 문법 대신 `for`문으로 하나씩 추가했다.
- 직선 후보와 원 후보는 `score` 값을 기준으로 정렬한다.

## 7. 출력

GUI를 사용할 수 있는 환경에서는 OpenCV 창으로 결과를 확인할 수 있다. Linux/WSL처럼 GUI가 확실하지 않은 환경에서는 창을 띄우지 않고 `output/` 폴더에 결과 PNG와 MP4를 저장한다.

Linux/WSL에서 GUI 사용을 직접 허용하려면 다음처럼 실행한다.

```bash
OPENCV_GUI=1 python3 main.py
```

실행 중 창이 열려 있으면 `q` 또는 `Esc`를 눌러 종료할 수 있다.

## 8. 참고 자료

- OpenCV Canny Edge Detection: <https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html>
- OpenCV Hough Line Transform: <https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html>
- OpenCV Hough Circle Transform: <https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html>
