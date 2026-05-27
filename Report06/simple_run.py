"""
simple_run.py
같은 폴더의 test_video.mp4에서 차선처럼 보이는 직선 상위 10개를 검출한다.

결과는 output 폴더에 두 개의 동영상으로만 저장한다.
1. canny_edge_detection.mp4
2. hough_line_detection.mp4
"""

import math
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "test_video.mp4"
OUTPUT_DIR = BASE_DIR / "output"
CANNY_OUTPUT_PATH = OUTPUT_DIR / "canny_edge_detection.mp4"
HOUGH_OUTPUT_PATH = OUTPUT_DIR / "hough_line_detection.mp4"

TOP_LINE_COUNT = 4
LOWER_HALF_LINE_WEIGHT = 10.0

CANNY_LOW_THRESHOLD = 80
CANNY_HIGH_THRESHOLD = 160
GAUSSIAN_KERNEL_SIZE = (5, 5)

LINE_RHO = 1.0
LINE_THETA = math.pi / 180.0
LINE_THRESHOLD = 60
LINE_MIN_LENGTH = 60
LINE_MAX_GAP = 20
LANE_MIN_ABS_ANGLE = 20.0
LANE_MAX_ABS_ANGLE = 75.0


def main():
    """입력 영상을 읽고 Canny 영상과 Hough 직선 검출 영상을 저장한다."""
    if not INPUT_PATH.exists():
        raise RuntimeError("ERROR: input video does not exist: " + str(INPUT_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(INPUT_PATH))
    if not capture.isOpened():
        raise RuntimeError("ERROR: input video could not be opened: " + str(INPUT_PATH))

    fps = get_video_fps(capture)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    canny_writer = create_video_writer(CANNY_OUTPUT_PATH, fps, width, height)
    hough_writer = create_video_writer(HOUGH_OUTPUT_PATH, fps, width, height)

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            edges = detect_edges(frame)
            line_frame = draw_top_lane_lines(frame, edges)

            canny_writer.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
            hough_writer.write(line_frame)
            frame_count = frame_count + 1
    finally:
        capture.release()
        canny_writer.release()
        hough_writer.release()

    print("DONE: processed frames = " + str(frame_count))
    print("DONE: Canny Edge Detection video = " + str(CANNY_OUTPUT_PATH))
    print("DONE: Hough Line Detection video = " + str(HOUGH_OUTPUT_PATH))


def get_video_fps(capture):
    """영상 FPS를 읽는다. 알 수 없으면 30을 사용한다."""
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps > 1.0 and not math.isnan(fps):
        return fps
    return 30.0


def create_video_writer(path, fps, width, height):
    """MP4 동영상 저장기를 만든다."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("ERROR: output video writer could not be opened: " + str(path))
    return writer


def detect_edges(frame):
    """YCbCr의 Y 밝기 성분에 Gaussian Blur와 Canny를 적용한다."""
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_frame[:, :, 0]
    blurred = cv2.GaussianBlur(y_channel, GAUSSIAN_KERNEL_SIZE, 0)
    return cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)


def draw_top_lane_lines(frame, edges):
    """차선 각도에 맞는 상위 n개 직선을 원본 프레임 위에 그린다."""
    output = frame.copy()
    lines = find_top_lane_lines(edges, frame.shape[0], frame.shape[1])

    rank = 1
    for line in lines:
        score, x1, y1, x2, y2 = line
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            output,
            "#" + str(rank), #+ " " + str(round(score),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        rank = rank + 1

    cv2.putText(
        output,
        "Top " + str(len(lines)) + "/" + str(TOP_LINE_COUNT) + " Lane Lines",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    return output


def find_top_lane_lines(edges, image_height, image_width):
    """HoughLinesP 결과 중 차선처럼 보이는 직선 상위 n개를 반환한다."""
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

    lane_lines = []
    for raw_line in raw_lines:
        x1, y1, x2, y2 = raw_line[0]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if is_lane_angle(x1, y1, x2, y2):
            score = line_score(x1, y1, x2, y2, image_height, image_width)
            if score > 0:  # 양수 score만 추가
                lane_lines.append((score, x1, y1, x2, y2))

    lane_lines.sort(reverse=True)
    return lane_lines[:TOP_LINE_COUNT]


def is_lane_angle(x1, y1, x2, y2):
    """직선 각도가 차선처럼 보이는 대각선 범위인지 확인한다."""
 
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle >= LANE_MIN_ABS_ANGLE and angle <= LANE_MAX_ABS_ANGLE


def line_score(x1, y1, x2, y2, image_height, image_width):
    """화면 아래쪽 직선과 화면 중앙에 가까운 직선에 높은 점수를 준다."""
    score = math.hypot(x2 - x1, y2 - y1)
    # middle_y = (y1 + y2) / 2.0
    # # if middle_y >= image_height / 2.0:
    # #     score = score * LOWER_HALF_LINE_WEIGHT
    # if middle_y < image_height / 2.0:
    #     score = 0.0;
    center_x = (x1 + x2) / 2.0
    if (center_x >= image_width / 4.0) and (center_x <= image_width * 3.0 / 4.0):
        score = score * (1.0 + (1.0 - abs(center_x - image_width / 2.0) / (image_width / 2.0)))
    if y2 < y1:
        y1, y2 = y2, y1
    if y1 < image_height / 2.0:
        score = 0.0
    return score


if __name__ == "__main__":
    main()
