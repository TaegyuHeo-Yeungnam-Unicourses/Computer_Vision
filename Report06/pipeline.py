"""
pipeline.py
프레임 하나를 Y 성분, Canny, Hough 검출 순서로 처리한다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

import cv2

from detectors import ShapeDetectorFactory


class EdgeDetectionResult:
    """Y 성분 영상과 Canny 결과를 함께 저장한다."""

    def __init__(self, y_channel, edges):
        """처리 중간 결과를 저장한다."""
        self.y_channel = y_channel
        self.edges = edges


class YChannelEdgeDetector:
    """BGR 프레임에서 Y 밝기 성분을 꺼내 Canny를 적용한다."""

    def __init__(self, settings):
        """에지 검출 설정을 저장한다."""
        self.settings = settings

    def detect(self, bgr_frame):
        """Y 성분과 Canny 에지 영상을 반환한다."""
        y_channel = self.extract_y_channel(bgr_frame)
        edges = self.detect_edges(y_channel)
        return EdgeDetectionResult(y_channel, edges)

    def extract_y_channel(self, bgr_frame):
        """BGR 영상을 YCrCb로 바꾸고 Y 밝기 성분만 꺼낸다."""
        ycrcb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        return ycrcb_frame[:, :, 0]

    def detect_edges(self, y_channel):
        """Y 성분 영상에 Blur와 Canny Edge Detection을 적용한다."""
        blurred = cv2.GaussianBlur(y_channel, self.settings.gaussian_kernel_size, 0)
        edges = cv2.Canny(
            blurred,
            self.settings.canny_low_threshold,
            self.settings.canny_high_threshold,
        )
        return edges


class FrameProcessor:
    """프레임 처리에 필요한 객체들을 조립하고 실행한다."""

    def __init__(self, config, settings):
        """에지 검출기와 모드별 도형 검출기를 만든다."""
        self.edge_detector = YChannelEdgeDetector(settings.edge)
        self.shape_detector = ShapeDetectorFactory(config, settings).create()

    def process(self, frame):
        """프레임 하나를 처리해 Canny 결과와 검출 결과를 반환한다."""
        edge_result = self.edge_detector.detect(frame)
        overlay = self.shape_detector.create_overlay(
            frame,
            edge_result.y_channel,
            edge_result.edges,
        )
        return edge_result.edges, overlay
