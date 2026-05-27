"""
source.py
이미지, 동영상 파일, 웹캠에서 프레임을 읽는다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

import math
from pathlib import Path

import cv2


class FrameSource:
    """이미지, 동영상 파일, 웹캠 입력을 같은 방식으로 읽게 해 주는 클래스이다."""

    def __init__(self, input_type, source, settings):
        """입력 종류, 입력 소스, 입력 설정을 저장한다."""
        self.input_type = input_type
        self.source = source
        self.settings = settings
        self.capture = None
        self.image = None
        self.image_was_read = False

    def is_webcam(self):
        """입력이 웹캠이면 True를 반환한다."""
        return self.input_type == "webcam"

    def open(self):
        """입력 종류에 맞게 소스를 연다."""
        if self.is_webcam():
            self.open_webcam()
        else:
            self.open_file()

    def read(self):
        """다음 프레임을 읽는다."""
        if self.image is not None:
            return self.read_image_once()

        if self.capture is None:
            return False, None

        ok, frame = self.capture.read()
        if ok:
            return True, frame
        return False, None

    def fps(self):
        """영상 FPS를 반환한다. 알 수 없으면 기본 FPS를 반환한다."""
        if self.capture is None:
            return self.settings.fallback_fps

        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if fps > 1.0 and not math.isnan(fps):
            return fps
        return self.settings.fallback_fps

    def release(self):
        """입력 자원을 정리한다."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read_image_once(self):
        """이미지는 한 프레임처럼 한 번만 반환한다."""
        if self.image_was_read:
            return False, None
        self.image_was_read = True
        return True, self.image.copy()

    def open_webcam(self):
        """웹캠 번호를 이용해 카메라를 연다."""
        if not self.source.isdigit():
            raise RuntimeError("ERROR: webcam source must be a camera number, for example 0.")

        self.capture = cv2.VideoCapture(int(self.source))
        if not self.capture.isOpened():
            raise RuntimeError("ERROR: webcam could not be opened: " + self.source)

    def open_file(self):
        """이미지 또는 동영상 파일을 연다."""
        path = self.resolve_file_path(self.source)
        if not path.exists():
            raise RuntimeError("ERROR: input file does not exist: " + str(path))

        if self.is_image_file(path):
            self.open_image_file(path)
            return

        self.open_video_file(path)

    def resolve_file_path(self, source):
        """상대 경로를 Report06 폴더 기준 절대 경로로 바꾼다."""
        path = Path(source)
        if path.is_absolute():
            return path
        return (self.settings.base_dir / path).resolve()

    def is_image_file(self, path):
        """파일 확장자가 이미지이면 True를 반환한다."""
        return path.suffix.lower() in self.settings.image_extensions

    def open_image_file(self, path):
        """이미지 파일을 읽는다."""
        self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError("ERROR: image file could not be read: " + str(path))

    def open_video_file(self, path):
        """동영상 파일을 연다."""
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError("ERROR: video file could not be opened: " + str(path))
