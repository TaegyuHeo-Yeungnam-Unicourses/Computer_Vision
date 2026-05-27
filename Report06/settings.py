"""
settings.py
프로그램 기본값과 영상 처리 파라미터를 객체로 묶어 관리한다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

import math
from pathlib import Path


class ConfigDefaults:
    """실행 설정을 만들 때 사용할 기본값을 저장한다."""

    def __init__(
        self,
        mode,
        input_type,
        input_source,
        webcam_source,
        output_dir,
        lane_filter_enabled,
        circle_top_right_priority_enabled,
        max_frames,
    ):
        """기본 실행값을 객체 안에 저장한다."""
        self.mode = mode
        self.input_type = input_type
        self.input_source = input_source
        self.webcam_source = webcam_source
        self.output_dir = output_dir
        self.lane_filter_enabled = lane_filter_enabled
        self.circle_top_right_priority_enabled = circle_top_right_priority_enabled
        self.max_frames = max_frames


class InputSettings:
    """파일과 웹캠 입력을 열 때 필요한 값을 저장한다."""

    def __init__(self):
        """입력 파일 기준 폴더와 이미지 확장자를 준비한다."""
        self.base_dir = Path(__file__).resolve().parent
        self.fallback_fps = 30.0
        self.image_extensions = [
            ".bmp",
            ".dib",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tif",
            ".tiff",
        ]


class EdgeSettings:
    """Y 성분 추출 뒤 Canny 에지 검출에 사용할 값을 저장한다."""

    def __init__(self):
        """Blur와 Canny 임계값을 준비한다."""
        self.gaussian_kernel_size = (5, 5)
        self.canny_low_threshold = 80
        self.canny_high_threshold = 160


class LineSettings:
    """Hough 직선 검출과 차선 후보 점수화 값을 저장한다."""

    def __init__(self):
        """직선 검출 파라미터와 표시 개수를 준비한다."""
        self.max_results = 10
        self.lower_half_score_weight = 1.5
        self.rho = 1.0
        self.theta = math.pi / 180.0
        self.threshold = 60
        self.min_length = 60
        self.max_gap = 20
        self.lane_min_abs_angle = 20.0
        self.lane_max_abs_angle = 75.0


class CircleSettings:
    """Hough 원 검출과 원 후보 점수화 값을 저장한다."""

    def __init__(self):
        """원 검출 파라미터와 표시 개수를 준비한다."""
        self.max_results = 3
        self.dp = 1.2
        self.min_dist_ratio = 0.08
        self.param1 = 120
        self.param2 = 30
        self.min_radius = 8
        self.max_radius = 0
        self.median_blur_kernel_size = 5
        self.sample_count = 180
        self.edge_probe_radius = 2
        self.top_right_weight = 0.35


class DisplaySettings:
    """결과 표시와 저장 주기에 사용할 값을 저장한다."""

    def __init__(self):
        """창 표시 지연 시간과 저장 주기를 준비한다."""
        self.wait_delay_ms = 1
        self.save_every_n_frames = 30
        self.headless_webcam_frame_limit = 120


class ProgramSettings:
    """Report06 전체에서 공유할 설정 묶음이다."""

    def __init__(self):
        """역할별 설정 객체를 만든다."""
        self.defaults = ConfigDefaults(
            "straight",
            "file",
            "test_video.mp4",
            "0",
            "output",
            True,
            False,
            0,
        )
        self.input = InputSettings()
        self.edge = EdgeSettings()
        self.line = LineSettings()
        self.circle = CircleSettings()
        self.display = DisplaySettings()
