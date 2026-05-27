"""
main.py
YCbCr의 Y 성분에 Canny를 적용하고 Hough Transform으로 직선 또는 원을 검출한다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

from form import ConfigDefaults, UserInputForm
from ui import EnvironmentInfo, ResultDisplayManager, draw_label


# ============================================================
# 기본 설정 영역
# 개발자가 자주 바꿀 값은 이곳에 모아 둔다.
# ============================================================
DEFAULT_MODE = "straight"                 # "straight" 또는 "circle"
DEFAULT_INPUT_TYPE = "file"               # "file" 또는 "webcam"
INPUT_SOURCE = "test_video.mp4"           # 기본 입력 파일
WEBCAM_SOURCE = "0"                       # 웹캠 기본 번호
OUTPUT_DIR = "output"

TOP_STRAIGHT_LINES = 10                   # 표시할 상위 직선 개수
TOP_CIRCLES = 3                           # 표시할 상위 원 개수
LOWER_HALF_LINE_WEIGHT = 1.5              # 화면 아래쪽 직선에 주는 점수 가중치

LANE_FILTER_ENABLED = True                # 차선 각도 필터 사용 여부
CIRCLE_TOP_RIGHT_PRIORITY_ENABLED = False # 원 검출에서 우측 상단 우선 여부
MAX_FRAMES = 0                            # 0이면 끝까지 처리

# Canny 설정
CANNY_LOW_THRESHOLD = 80
CANNY_HIGH_THRESHOLD = 160
GAUSSIAN_KERNEL_SIZE = (5, 5)

# Hough Line 설정
LINE_RHO = 1.0
LINE_THETA = math.pi / 180.0
LINE_THRESHOLD = 60
LINE_MIN_LENGTH = 60
LINE_MAX_GAP = 20
LANE_MIN_ABS_ANGLE = 20.0
LANE_MAX_ABS_ANGLE = 75.0

# Hough Circle 설정
CIRCLE_DP = 1.2
CIRCLE_MIN_DIST_RATIO = 0.08
CIRCLE_PARAM1 = 120
CIRCLE_PARAM2 = 30
CIRCLE_MIN_RADIUS = 8
CIRCLE_MAX_RADIUS = 0
CIRCLE_SAMPLE_COUNT = 180
CIRCLE_EDGE_PROBE_RADIUS = 2
CIRCLE_TOP_RIGHT_WEIGHT = 0.35

# 출력 설정
WAIT_DELAY_MS = 1
SAVE_EVERY_N_FRAMES = 30
HEADLESS_WEBCAM_FRAME_LIMIT = 120

IMAGE_EXTENSIONS = [".bmp", ".dib", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"]


class LineCandidate:
    """검출된 직선 후보를 저장하는 클래스이다."""

    def __init__(self, x1, y1, x2, y2, score):
        """직선의 양 끝점과 점수를 저장한다."""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score

    def start(self):
        """직선의 시작 좌표를 반환한다."""
        return self.x1, self.y1

    def end(self):
        """직선의 끝 좌표를 반환한다."""
        return self.x2, self.y2


class CircleCandidate:
    """검출된 원 후보를 저장하는 클래스이다."""

    def __init__(self, x, y, radius, score):
        """원의 중심, 반지름, 점수를 저장한다."""
        self.x = x
        self.y = y
        self.radius = radius
        self.score = score

    def center(self):
        """원의 중심 좌표를 반환한다."""
        return self.x, self.y


class FrameSource:
    """이미지, 동영상 파일, 웹캠에서 프레임을 읽는 클래스이다."""

    def __init__(self, input_type, source):
        """입력 종류와 입력 소스를 저장한다."""
        self.input_type = input_type
        self.source = source
        self.capture = None
        self.image = None
        self.image_was_read = False

    def is_webcam(self):
        """입력이 웹캠이면 True를 반환한다."""
        return self.input_type == "webcam"

    def open(self):
        """입력 소스를 연다."""
        if self.is_webcam():
            self.open_webcam()
        else:
            self.open_file()

    def read(self):
        """다음 프레임을 읽는다."""
        if self.image is not None:
            if self.image_was_read:
                return False, None
            self.image_was_read = True
            return True, self.image.copy()

        if self.capture is None:
            return False, None

        ok, frame = self.capture.read()
        if ok:
            return True, frame
        return False, None

    def fps(self):
        """영상 FPS를 반환한다. 알 수 없으면 30을 반환한다."""
        if self.capture is None:
            return 30.0

        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if fps > 1.0 and not math.isnan(fps):
            return fps
        return 30.0

    def release(self):
        """입력 자원을 정리한다."""
        if self.capture is not None:
            self.capture.release()

    def open_webcam(self):
        """웹캠 번호를 이용해 카메라를 연다."""
        if not self.source.isdigit():
            raise RuntimeError("ERROR: webcam source must be a camera number, for example 0.")

        self.capture = cv2.VideoCapture(int(self.source))
        if not self.capture.isOpened():
            raise RuntimeError("ERROR: webcam could not be opened: " + self.source)

    def open_file(self):
        """이미지 또는 동영상 파일을 연다."""
        path = Path(self.source)
        
        if not path.is_absolute(): # 상대 경로인 경우, main.py 기준으로 절대 경로를 만든다
            base_dir = Path(__file__).resolve().parent
            path = (base_dir / path).resolve()
        
        if not path.exists(): # 경로안에 파일이 없으면 에러 밷는다.vv
            raise RuntimeError("ERROR: input file does not exist: " + str(path))

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if self.image is None:
                raise RuntimeError("ERROR: image file could not be read: " + str(path))
            return

        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError("ERROR: video file could not be opened: " + str(path))


def make_default_config():
    """main.py 위쪽의 기본 설정값을 ConfigDefaults 객체로 만든다."""
    return ConfigDefaults(
        DEFAULT_MODE,
        DEFAULT_INPUT_TYPE,
        INPUT_SOURCE,
        WEBCAM_SOURCE,
        OUTPUT_DIR,
        LANE_FILTER_ENABLED,
        CIRCLE_TOP_RIGHT_PRIORITY_ENABLED,
        MAX_FRAMES,
        WAIT_DELAY_MS,
        SAVE_EVERY_N_FRAMES,
        TOP_STRAIGHT_LINES,
        TOP_CIRCLES,
    )


def extract_y_channel(bgr_frame):
    """BGR 영상을 YCrCb로 바꾸고 Y 밝기 성분만 꺼낸다."""
    ycrcb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
    return ycrcb_frame[:, :, 0]


def detect_edges(y_channel):
    """Y 성분 영상에 Blur와 Canny Edge Detection을 적용한다."""
    blurred = cv2.GaussianBlur(y_channel, GAUSSIAN_KERNEL_SIZE, 0)
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    return edges


def line_length(x1, y1, x2, y2):
    """직선의 길이를 계산한다."""
    return math.hypot(x2 - x1, y2 - y1)


def line_abs_angle(x1, y1, x2, y2):
    """직선 각도를 0도에서 90도 사이 값으로 계산한다."""
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def is_lane_angle(x1, y1, x2, y2):
    """직선 각도가 차선처럼 보이는 대각선 범위인지 확인한다."""
    angle = line_abs_angle(x1, y1, x2, y2)
    if angle >= LANE_MIN_ABS_ANGLE and angle <= LANE_MAX_ABS_ANGLE:
        return True
    return False


def line_mid_y(x1, y1, x2, y2):
    """직선 중점의 y 좌표를 계산한다."""
    return (y1 + y2) / 2.0


def make_line_candidate(raw_line, image_height, lane_filter_enabled):
    """HoughLinesP 결과 하나를 LineCandidate 객체로 바꾼다."""
    x1 = int(raw_line[0])
    y1 = int(raw_line[1])
    x2 = int(raw_line[2])
    y2 = int(raw_line[3])

    if lane_filter_enabled:
        if not is_lane_angle(x1, y1, x2, y2):
            return None

    score = line_length(x1, y1, x2, y2)
    if lane_filter_enabled:
        if line_mid_y(x1, y1, x2, y2) >= image_height / 2.0:
            score = score * LOWER_HALF_LINE_WEIGHT

    return LineCandidate(x1, y1, x2, y2, score)


def upper_right_score(x, y, frame_shape):
    """우측 상단에 가까운 원일수록 높은 위치 점수를 준다."""
    height = frame_shape[0]
    width = frame_shape[1]
    distance = math.hypot(width - x, y)
    max_distance = math.hypot(width, height)
    score = 1.0 - distance / max_distance
    if score < 0.0:
        score = 0.0
    return score


def has_nearby_edge(edges, x, y):
    """한 좌표 주변에 Canny 에지가 있는지 확인한다."""
    height = edges.shape[0]
    width = edges.shape[1]
    radius = CIRCLE_EDGE_PROBE_RADIUS

    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)

    if np.any(edges[y1:y2, x1:x2] > 0):
        return True
    return False


def circle_support_score(edges, center_x, center_y, radius):
    """원 둘레 샘플 지점에 에지가 얼마나 많이 있는지 계산한다."""
    height = edges.shape[0]
    width = edges.shape[1]
    hit_count = 0
    sample_count = 0

    angles = np.linspace(0.0, 2.0 * math.pi, CIRCLE_SAMPLE_COUNT, endpoint=False)
    for angle in angles:
        x = int(round(center_x + radius * math.cos(angle)))
        y = int(round(center_y + radius * math.sin(angle)))

        if x >= 0 and x < width and y >= 0 and y < height:
            sample_count = sample_count + 1
            if has_nearby_edge(edges, x, y):
                hit_count = hit_count + 1

    if sample_count == 0:
        return 0.0
    return hit_count / sample_count


def make_circle_candidate(x, y, radius, edges, frame_shape, config):
    """HoughCircles 결과 하나를 CircleCandidate 객체로 바꾼다."""
    support = circle_support_score(edges, x, y, radius)
    score = support

    if config.circle_top_right_priority_enabled:
        position = upper_right_score(x, y, frame_shape)
        score = (1.0 - CIRCLE_TOP_RIGHT_WEIGHT) * support + CIRCLE_TOP_RIGHT_WEIGHT * position

    return CircleCandidate(x, y, radius, score)


class HoughLineDetector:
    """Canny 에지 영상에서 Hough 직선을 찾는 클래스이다."""

    def __init__(self, config):
        """직선 검출에 필요한 설정을 저장한다."""
        self.config = config

    def detect(self, edges, frame_shape):
        """점수가 높은 직선 후보들을 찾는다."""
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=LINE_RHO,
            theta=LINE_THETA,
            threshold=LINE_THRESHOLD,
            minLineLength=LINE_MIN_LENGTH,
            maxLineGap=LINE_MAX_GAP,
        )

        if raw_lines is None:
            return []

        image_height = frame_shape[0]
        candidates = []
        for raw_line in raw_lines:
            line_data = raw_line[0]
            candidate = make_line_candidate(line_data, image_height, self.config.lane_filter_enabled)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=line_score, reverse=True)
        return candidates[:self.config.top_straight_lines]

    def draw(self, frame, lines):
        """선택된 직선을 프레임 위에 그린다."""
        output = frame.copy()

        rank = 1
        for line in lines:
            cv2.line(output, line.start(), line.end(), (0, 0, 255), 3)
            cv2.putText(
                output,
                "#" + str(rank) + " " + str(round(line.score)),
                line.start(),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            rank = rank + 1

        if self.config.lane_filter_enabled:
            lane_text = "ON"
        else:
            lane_text = "OFF"

        label = "Top " + str(len(lines)) + "/" + str(self.config.top_straight_lines) + " Lines | lane filter: " + lane_text
        draw_label(output, label)
        return output


class HoughCircleDetector:
    """밝기 영상과 Canny 에지 영상에서 Hough 원을 찾는 클래스이다."""

    def __init__(self, config):
        """원 검출에 필요한 설정을 저장한다."""
        self.config = config

    def detect(self, y_channel, edges, frame_shape):
        """점수가 높은 원 후보들을 찾는다."""
        height = frame_shape[0]
        width = frame_shape[1]
        min_distance = max(10, int(min(height, width) * CIRCLE_MIN_DIST_RATIO))
        blurred = cv2.medianBlur(y_channel, 5)

        raw_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=CIRCLE_DP,
            minDist=min_distance,
            param1=CIRCLE_PARAM1,
            param2=CIRCLE_PARAM2,
            minRadius=CIRCLE_MIN_RADIUS,
            maxRadius=CIRCLE_MAX_RADIUS,
        )

        if raw_circles is None:
            return []

        rounded = np.round(raw_circles[0]).astype(int)
        candidates = []
        for circle in rounded:
            x = int(circle[0])
            y = int(circle[1])
            radius = int(circle[2])
            if radius > 0:
                candidate = make_circle_candidate(x, y, radius, edges, frame_shape, self.config)
                candidates.append(candidate)

        candidates.sort(key=circle_score, reverse=True)
        return candidates[:TOP_CIRCLES]

    def draw(self, frame, circles):
        """선택된 원을 프레임 위에 그린다."""
        output = frame.copy()

        rank = 1
        for circle in circles:
            cv2.circle(output, circle.center(), circle.radius, (0, 255, 0), 3)
            cv2.circle(output, circle.center(), 3, (0, 0, 255), -1)
            cv2.putText(
                output,
                "#" + str(rank) + " " + str(round(circle.score, 2)),
                (circle.x + 10, circle.y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            rank = rank + 1

        if self.config.circle_top_right_priority_enabled:
            score_text = "upper-right"
        else:
            score_text = "edge support"

        label = "Top " + str(len(circles)) + "/" + str(TOP_CIRCLES) + " Circles | score: " + score_text
        draw_label(output, label)
        return output


def line_score(line):
    """직선 후보 정렬에 사용할 점수를 반환한다."""
    return line.score


def circle_score(circle):
    """원 후보 정렬에 사용할 점수를 반환한다."""
    return circle.score


def process_frame(frame, config):
    """프레임 하나를 처리해 Canny 결과와 검출 결과를 반환한다."""
    y_channel = extract_y_channel(frame)
    edges = detect_edges(y_channel)

    if config.mode == "straight":
        detector = HoughLineDetector(config)
        lines = detector.detect(edges, frame.shape)
        overlay = detector.draw(frame, lines)
    else:
        detector = HoughCircleDetector(config)
        circles = detector.detect(y_channel, edges, frame.shape)
        overlay = detector.draw(frame, circles)

    return edges, overlay


class ShapeDetectionApplication:
    """입력, 처리, 출력을 순서대로 실행하는 클래스이다."""

    def __init__(self, config, environment):
        """설정과 환경 정보를 저장하고 입력 객체를 만든다."""
        self.config = config
        self.environment = environment
        self.source = FrameSource(config.input_type, config.input_source)

    def run(self):
        """프레임을 반복해서 읽고 처리한다."""
        self.source.open()
        display = ResultDisplayManager(self.config, self.environment, self.source.fps())
        max_frames = self.effective_max_frames()
        frame_index = 0

        self.print_start_message()
        try:
            while True:
                ok, frame = self.source.read()
                if not ok:
                    break
                if frame is None:
                    break

                edges, overlay = process_frame(frame, self.config)
                should_continue = display.handle_frame(edges, overlay, frame_index)

                frame_index = frame_index + 1
                if not should_continue:
                    break
                if max_frames > 0 and frame_index >= max_frames:
                    print("INFO: reached max frame limit: " + str(max_frames))
                    break
        finally:
            display.close()
            self.source.release()

        self.print_finish_message(display, frame_index)

    def effective_max_frames(self):
        """실제로 사용할 최대 프레임 수를 정한다."""
        if self.config.max_frames > 0:
            return self.config.max_frames
        if self.source.is_webcam() and not self.environment.can_show_opencv_window():
            return HEADLESS_WEBCAM_FRAME_LIMIT
        return 0

    def print_start_message(self):
        """프로그램 시작 정보를 출력한다."""
        print("INFO: mode = " + self.config.mode)
        print("INFO: input type = " + self.config.input_type)
        print("INFO: input source = " + self.config.input_source)
        print("INFO: environment = " + self.environment.label())

    def print_finish_message(self, display, processed_frames):
        """프로그램 종료 정보를 출력한다."""
        print("DONE: processed frames = " + str(processed_frames))
        if display.last_image_path is not None:
            print("DONE: last result image = " + str(display.last_image_path))
        if display.video_path is not None:
            print("DONE: result video = " + str(display.video_path))


def build_config(argv=None):
    """form.py의 UserInputForm을 사용해 실행 설정을 만든다."""
    form = UserInputForm(make_default_config(), argv)
    return form.collect_config()


def main(argv=None):
    """프로그램을 시작하는 함수이다."""
    try:
        config = build_config(argv)
        environment = EnvironmentInfo()
        application = ShapeDetectionApplication(config, environment)
        application.run()
    except (ValueError, RuntimeError) as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
