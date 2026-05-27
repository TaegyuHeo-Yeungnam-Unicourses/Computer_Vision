# 이 프로그램의 전체 흐름은 다음과 같다.
# 1. main.py 상단의 기본 설정, 환경변수, 명령행 인자, form.py 사용자 입력을 읽는다.
# 2. shape 값은 대소문자를 구분하지 않고 straight 또는 circle만 허용한다.
# 3. 입력 영상은 카메라 번호, 동영상 파일, 또는 이미지 파일로 열 수 있다.
# 4. 각 프레임의 RGB 값을 YCbCr 방식으로 변환하고 Y 성분만 추출한다.
# 5. 추출한 Y 성분에 Gaussian Blur와 Canny Edge Detection을 적용한다.
# 6. straight 모드에서는 Canny 결과에 Hough Line Detection을 적용한다.
# 7. 직선 검출 모드에서는 차선 후보 각도와 하단 영역 필터를 켜거나 끌 수 있다.
# 8. circle 모드에서는 Y 성분에 Hough Circle Detection을 적용한다.
# 9. 원 검출 모드에서는 원형 에지 일치도 또는 우측 상단 우선 옵션으로 최종 원을 고른다.
# 10. ui.py는 Windows, WSL, Debian/Linux 환경에 맞게 창 표시 또는 파일 저장을 수행한다.

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from form import ConfigDefaults, RuntimeConfig, UserInputForm
from ui import EnvironmentInfo, ResultDisplayManager, draw_label


# ==============================
# main.py 기본 설정 영역
# ==============================
# shape 환경변수 또는 --shape 인자가 없을 때 사용하는 기본 검출 모드이다.
shape = "straight"

# 입력 소스 기본값이다. "0"은 카메라 0번이며, 동영상/이미지 경로로 바꿀 수 있다.
INPUT_SOURCE = "0"

# 직선 모드에서 차선형 각도/하단 영역 필터를 적용할지 결정한다.
LANE_FILTER_ENABLED = True

# 원 모드에서 원형 정확도 외에 우측 상단 위치를 우선할지 결정한다.
CIRCLE_TOP_RIGHT_PRIORITY_ENABLED = False

# 처리할 최대 프레임 수이다. 0이면 제한 없음이다.
MAX_FRAMES = 0

# 결과 이미지와 동영상을 저장할 폴더이다.
OUTPUT_DIR = "output"


def is_camera_source(source_text: str) -> bool:
    """입력 소스 문자열이 카메라 번호인지 판단한다."""
    return source_text.strip().isdigit()


def compute_ycbcr_y(bgr_frame: np.ndarray) -> np.ndarray:
    """BGR 프레임을 RGB로 바꾼 뒤 YCbCr의 Y 밝기 성분만 계산한다."""
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    red_channel = rgb_frame[:, :, 0]
    green_channel = rgb_frame[:, :, 1]
    blue_channel = rgb_frame[:, :, 2]

    # YCbCr의 Y 공식이다. 색차 성분 Cb, Cr은 과제 조건에 따라 사용하지 않는다.
    y_channel = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
    return np.clip(y_channel, 0, 255).astype(np.uint8)


def detect_canny_edges(y_channel: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Y 성분 영상에 Gaussian Blur와 Canny Edge Detection을 적용한다."""
    blurred_y = cv2.GaussianBlur(y_channel, (5, 5), 0)
    return cv2.Canny(blurred_y, low_threshold, high_threshold)


def line_angle_degrees(x1: int, y1: int, x2: int, y2: int) -> float:
    """직선 성분의 각도를 도 단위로 계산한다."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def is_lane_like_line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_shape: tuple[int, int, int],
    config: RuntimeConfig,
) -> bool:
    """직선 성분이 차선처럼 보이는 각도와 위치를 가지는지 판정한다."""
    angle = line_angle_degrees(x1, y1, x2, y2)
    abs_angle = abs(angle)
    midpoint_y = (y1 + y2) / 2.0
    image_height = frame_shape[0]

    # 차선은 보통 영상 하단부에 있고, 수평/수직보다 대각선 성분에 가깝다.
    angle_is_lane_like = config.lane_min_abs_angle <= abs_angle <= config.lane_max_abs_angle
    midpoint_is_low_enough = midpoint_y >= image_height * config.lane_min_y_ratio
    return angle_is_lane_like and midpoint_is_low_enough


def upper_right_score(x: int, y: int, frame_shape: tuple[int, int, int]) -> float:
    """점이 영상의 우측 상단에 가까울수록 높은 점수를 반환한다."""
    image_height, image_width = frame_shape[:2]
    distance = math.hypot(image_width - x, y)
    max_distance = math.hypot(image_width, image_height)
    return max(0.0, 1.0 - distance / max_distance)


def make_default_config() -> ConfigDefaults:
    """main.py 상단 설정값을 ConfigDefaults 객체로 묶어 반환한다."""
    return ConfigDefaults(
        shape=shape,
        input_source=INPUT_SOURCE,
        output_dir=OUTPUT_DIR,
        lane_filter_enabled=LANE_FILTER_ENABLED,
        circle_top_right_priority_enabled=CIRCLE_TOP_RIGHT_PRIORITY_ENABLED,
        max_frames=MAX_FRAMES,
    )


class FrameSource:
    """카메라, 동영상 파일, 이미지 파일에서 프레임을 하나씩 제공하는 클래스이다."""

    IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

    def __init__(self, source_text: str) -> None:
        """입력 소스 문자열을 저장하고 내부 상태를 초기화한다."""
        self.source_text = source_text
        self.capture: Optional[cv2.VideoCapture] = None
        self.single_image: Optional[np.ndarray] = None
        self.single_image_already_read = False
        self.is_camera = is_camera_source(source_text)

    def open(self) -> None:
        """입력 소스를 열고 실패하면 RuntimeError를 발생시킨다."""
        source_path = Path(self.source_text)
        if source_path.exists() and source_path.suffix.lower() in self.IMAGE_EXTENSIONS:
            self.single_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if self.single_image is None:
                raise RuntimeError(f"ERROR: image file could not be read: {source_path}")
            return

        capture_source: Union[int, str] = int(self.source_text) if self.is_camera else self.source_text
        self.capture = cv2.VideoCapture(capture_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"ERROR: video source could not be opened: {self.source_text}")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """다음 프레임을 읽고 성공 여부와 프레임을 반환한다."""
        if self.single_image is not None:
            if self.single_image_already_read:
                return False, None
            self.single_image_already_read = True
            return True, self.single_image.copy()

        if self.capture is None:
            return False, None
        success, frame = self.capture.read()
        return success, frame if success else None

    def fps(self) -> float:
        """입력 FPS를 반환하고, 알 수 없으면 30 FPS를 기본값으로 사용한다."""
        if self.capture is None:
            return 30.0
        fps_value = float(self.capture.get(cv2.CAP_PROP_FPS))
        if fps_value <= 1.0 or math.isnan(fps_value):
            return 30.0
        return fps_value

    def release(self) -> None:
        """OpenCV 캡처 객체가 존재하면 해제한다."""
        if self.capture is not None:
            self.capture.release()


class YCbCrYExtractor:
    """YCbCr 색상 모델의 Y 밝기 성분만 추출하는 클래스이다."""

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        """BGR 프레임에서 YCbCr의 Y 성분을 추출한다."""
        return compute_ycbcr_y(bgr_frame)


class CannyEdgeDetector:
    """Y 성분 영상에 Canny Edge Detection을 적용하는 클래스이다."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Canny 임계값을 설정 객체에서 가져온다."""
        self.low_threshold = config.canny_low_threshold
        self.high_threshold = config.canny_high_threshold

    def detect(self, y_channel: np.ndarray) -> np.ndarray:
        """Y 성분 영상에서 이진 에지 영상을 생성한다."""
        return detect_canny_edges(y_channel, self.low_threshold, self.high_threshold)


class BaseShapeDetector:
    """직선 검출기와 원 검출기가 공유하는 공통 인터페이스이다."""

    def detect_and_draw(self, bgr_frame: np.ndarray, y_channel: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """도형을 검출하고 원본 프레임 복사본 위에 결과를 그린다."""
        raise NotImplementedError("자식 클래스에서 detect_and_draw를 구현해야 한다.")


class HoughLineDetector(BaseShapeDetector):
    """확률적 Hough Line Transform으로 직선 성분을 검출하는 클래스이다."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Hough 직선 파라미터와 차선 필터 설정을 저장한다."""
        self.config = config

    def detect_and_draw(self, bgr_frame: np.ndarray, y_channel: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Canny 에지 영상에서 직선을 찾고 원본 프레임 위에 겹쳐 그린다."""
        output_frame = bgr_frame.copy()
        lines = cv2.HoughLinesP(
            edges,
            rho=self.config.line_rho,
            theta=self.config.line_theta,
            threshold=self.config.line_threshold,
            minLineLength=self.config.line_min_length,
            maxLineGap=self.config.line_max_gap,
        )

        kept_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if not self._keep_line(x1, y1, x2, y2, bgr_frame.shape):
                    continue
                kept_count += 1
                cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        filter_text = "ON" if self.config.lane_filter_enabled else "OFF"
        draw_label(output_frame, f"Hough Line Detection | lane filter: {filter_text} | lines: {kept_count}")
        return output_frame

    def _keep_line(self, x1: int, y1: int, x2: int, y2: int, frame_shape: tuple[int, int, int]) -> bool:
        """차선 필터 설정에 따라 직선 유지 여부를 결정한다."""
        if not self.config.lane_filter_enabled:
            return True
        return is_lane_like_line(x1, y1, x2, y2, frame_shape, self.config)


@dataclass(frozen=True)
class CircleCandidate:
    """검출된 원의 중심, 반지름, 평가 점수를 저장하는 데이터 클래스이다."""

    x: int
    y: int
    radius: int
    support_score: float = 0.0
    ranking_score: float = 0.0

    def center(self) -> tuple[int, int]:
        """원의 중심 좌표를 (x, y) 튜플로 반환한다."""
        return self.x, self.y


class HoughCircleDetector(BaseShapeDetector):
    """Hough Circle Transform으로 원을 검출하고 최적 후보를 선택하는 클래스이다."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Hough 원 파라미터와 원 후보 선택 옵션을 저장한다."""
        self.config = config

    def detect_and_draw(self, bgr_frame: np.ndarray, y_channel: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Y 성분 영상에서 원 후보를 찾고 가장 좋은 원을 원본 프레임에 표시한다."""
        output_frame = bgr_frame.copy()
        candidates = self._detect_candidates(y_channel, edges, bgr_frame.shape)

        if candidates:
            best_circle = max(candidates, key=lambda candidate: candidate.ranking_score)
            cv2.circle(output_frame, best_circle.center(), best_circle.radius, (0, 255, 0), 3)
            cv2.circle(output_frame, best_circle.center(), 3, (0, 0, 255), -1)
            score_text = f"support: {best_circle.support_score:.2f}, rank: {best_circle.ranking_score:.2f}"
            cv2.putText(
                output_frame,
                score_text,
                (best_circle.x + 10, best_circle.y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        priority_text = "upper-right" if self.config.circle_top_right_priority_enabled else "roundness"
        draw_label(output_frame, f"Hough Circle Detection | priority: {priority_text} | candidates: {len(candidates)}")
        return output_frame

    def _detect_candidates(
        self,
        y_channel: np.ndarray,
        edges: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> list[CircleCandidate]:
        """HoughCircles 결과를 CircleCandidate 목록으로 변환하고 점수를 계산한다."""
        image_height, image_width = frame_shape[:2]
        min_distance = max(10, int(min(image_height, image_width) * self.config.circle_min_dist_ratio))
        blurred_y = cv2.medianBlur(y_channel, 5)
        raw_circles = cv2.HoughCircles(
            blurred_y,
            cv2.HOUGH_GRADIENT,
            dp=self.config.circle_dp,
            minDist=min_distance,
            param1=self.config.circle_param1,
            param2=self.config.circle_param2,
            minRadius=self.config.circle_min_radius,
            maxRadius=self.config.circle_max_radius,
        )

        if raw_circles is None:
            return []

        circles = np.round(raw_circles[0, :]).astype(int)
        return [self._make_candidate(int(x), int(y), int(radius), edges, frame_shape) for x, y, radius in circles if radius > 0]

    def _make_candidate(
        self,
        x: int,
        y: int,
        radius: int,
        edges: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> CircleCandidate:
        """검출 원 하나에 대해 에지 지지도와 최종 순위 점수를 계산한다."""
        support_score = self._edge_support_score(x, y, radius, edges)
        ranking_score = self._ranking_score(x, y, support_score, frame_shape)
        return CircleCandidate(x=x, y=y, radius=radius, support_score=support_score, ranking_score=ranking_score)

    def _edge_support_score(self, center_x: int, center_y: int, radius: int, edges: np.ndarray) -> float:
        """원 둘레 샘플 지점 주변에 Canny 에지가 존재하는 비율을 계산한다."""
        image_height, image_width = edges.shape[:2]
        angles = np.linspace(0.0, 2.0 * math.pi, self.config.circle_sample_count, endpoint=False)
        xs = np.rint(center_x + radius * np.cos(angles)).astype(int)
        ys = np.rint(center_y + radius * np.sin(angles)).astype(int)

        valid_mask = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask]
        if len(valid_xs) == 0:
            return 0.0

        supported_count = sum(
            1
            for x, y in zip(valid_xs, valid_ys)
            if self._has_nearby_edge(int(x), int(y), edges)
        )
        return supported_count / float(len(valid_xs))

    def _has_nearby_edge(self, x: int, y: int, edges: np.ndarray) -> bool:
        """샘플 지점 주변 작은 영역에 에지 픽셀이 있는지 검사한다."""
        image_height, image_width = edges.shape[:2]
        probe_radius = self.config.circle_edge_probe_radius
        x_start = max(0, x - probe_radius)
        x_end = min(image_width, x + probe_radius + 1)
        y_start = max(0, y - probe_radius)
        y_end = min(image_height, y + probe_radius + 1)
        return bool(np.any(edges[y_start:y_end, x_start:x_end] > 0))

    def _ranking_score(self, x: int, y: int, support_score: float, frame_shape: tuple[int, int, int]) -> float:
        """원형 에지 지지도와 선택적인 우측 상단 선호도를 합산한다."""
        if not self.config.circle_top_right_priority_enabled:
            return support_score
        weight = self.config.circle_top_right_weight
        return (1.0 - weight) * support_score + weight * upper_right_score(x, y, frame_shape)


class ShapeDetectorFactory:
    """선택된 shape 모드에 맞는 검출기 객체를 생성하는 팩토리 클래스이다."""

    @staticmethod
    def create(config: RuntimeConfig) -> BaseShapeDetector:
        """straight이면 직선 검출기, circle이면 원 검출기를 반환한다."""
        if config.mode == "straight":
            return HoughLineDetector(config)
        if config.mode == "circle":
            return HoughCircleDetector(config)
        raise ValueError(f"Unsupported mode: {config.mode}")


class ShapeDetectionApplication:
    """프레임 읽기, Y 추출, Canny, Hough 검출, UI 출력을 조율하는 응용 클래스이다."""

    def __init__(self, config: RuntimeConfig, environment: EnvironmentInfo) -> None:
        """실행에 필요한 소스, 전처리기, 검출기 객체를 생성한다."""
        self.config = config
        self.environment = environment
        self.source = FrameSource(config.input_source)
        self.y_extractor = YCbCrYExtractor()
        self.edge_detector = CannyEdgeDetector(config)
        self.shape_detector = ShapeDetectorFactory.create(config)

    def run(self) -> None:
        """입력 영상이 끝나거나 사용자가 종료할 때까지 전체 처리 루프를 실행한다."""
        self.source.open()
        display_manager = ResultDisplayManager(self.config, self.environment, self.source.fps())
        effective_max_frames = self._effective_max_frames()

        self._print_start_message()
        frame_index = 0
        try:
            while True:
                success, frame = self.source.read()
                if not success or frame is None:
                    break

                y_channel = self.y_extractor.extract(frame)
                edges = self.edge_detector.detect(y_channel)
                overlay_frame = self.shape_detector.detect_and_draw(frame, y_channel, edges)

                should_continue = display_manager.handle_frame(edges, overlay_frame, frame_index)
                frame_index += 1
                if not should_continue:
                    break
                if effective_max_frames > 0 and frame_index >= effective_max_frames:
                    print(f"INFO: reached max frame limit: {effective_max_frames}")
                    break
        finally:
            display_manager.close()
            self.source.release()

        self._print_finish_message(display_manager, frame_index)

    def _effective_max_frames(self) -> int:
        """GUI가 없는 웹캠 실행에서 무한 루프를 피하기 위한 실제 프레임 제한을 계산한다."""
        if self.config.max_frames > 0:
            return self.config.max_frames
        if self.source.is_camera and not self.environment.can_show_opencv_window():
            return 120
        return 0

    def _print_start_message(self) -> None:
        """프로그램 시작 시 주요 설정을 로그로 출력한다."""
        print(f"INFO: mode = {self.config.mode}")
        print(f"INFO: input source = {self.config.input_source}")
        print(f"INFO: environment = {self.environment.label()}")
        print(f"INFO: lane filter = {self.config.lane_filter_enabled}")
        print(f"INFO: circle upper-right priority = {self.config.circle_top_right_priority_enabled}")

    @staticmethod
    def _print_finish_message(display_manager: ResultDisplayManager, processed_frames: int) -> None:
        """처리 완료 후 결과 파일 경로와 처리 프레임 수를 출력한다."""
        print(f"DONE: processed frames = {processed_frames}")
        if display_manager.last_image_path is not None:
            print(f"DONE: last result image = {display_manager.last_image_path}")
        if display_manager.video_path is not None:
            print(f"DONE: result video = {display_manager.video_path}")


def build_config(argv: Optional[list[str]] = None) -> RuntimeConfig:
    """form.py의 UserInputForm을 사용하여 최종 RuntimeConfig를 생성한다."""
    return UserInputForm(make_default_config(), argv=argv).collect_config()


def main(argv: Optional[list[str]] = None) -> None:
    """설정을 읽고 실행환경을 감지한 뒤 ShapeDetectionApplication을 시작한다."""
    try:
        config = build_config(argv)
    except ValueError as error:
        print(error)
        sys.exit(1)

    environment = EnvironmentInfo.detect()
    application = ShapeDetectionApplication(config, environment)

    try:
        application.run()
    except RuntimeError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
