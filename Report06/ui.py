"""
ui.py
운영체제와 GUI 가능 여부를 감지하고, 결과 영상을 창 또는 파일로 출력하는 모듈이다.
Windows는 기본적으로 창 표시를 사용하고, Linux/WSL은 안전할 때만 창 표시를 사용한다.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from form import RuntimeConfig


def draw_label(image: np.ndarray, text: str, position: tuple[int, int] = (20, 35)) -> None:
    """영상 위에 상태 문구를 그린다."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


def make_combined_frame(edges: np.ndarray, overlay_frame: np.ndarray) -> np.ndarray:
    """Canny 에지 영상과 Hough 결과 영상을 좌우로 결합한다."""
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    draw_label(edges_bgr, "Canny Edge Detection")
    return np.hstack((edges_bgr, overlay_frame))


@dataclass(frozen=True)
class EnvironmentInfo:
    """현재 실행환경과 OpenCV 창 표시 가능 여부를 저장한다."""

    system_name: str
    is_windows: bool
    is_linux: bool
    is_wsl: bool
    has_gui: bool

    @classmethod
    def detect(cls) -> "EnvironmentInfo":
        """Windows, Linux, WSL 여부와 GUI 사용 가능 여부를 감지한다."""
        system_name = platform.system()
        is_windows = system_name.lower() == "windows"
        is_linux = system_name.lower() == "linux"
        is_wsl = cls._detect_wsl() if is_linux else False
        has_linux_gui = cls._linux_opencv_gui_is_safe() if is_linux else False
        has_gui = is_windows or has_linux_gui
        return cls(system_name, is_windows, is_linux, is_wsl, has_gui)

    @staticmethod
    def _detect_wsl() -> bool:
        """/proc/version 내용을 확인해 WSL 여부를 판별한다."""
        try:
            version_text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return "microsoft" in version_text.lower() or "wsl" in version_text.lower()

    @staticmethod
    def _linux_opencv_gui_is_safe() -> bool:
        """Linux/WSL에서 OPENCV_GUI=1이고 X 서버가 응답할 때만 GUI를 허용한다."""
        gui_requested = os.environ.get("OPENCV_GUI", "").strip().lower() in {"1", "true", "yes", "on"}
        if not gui_requested or not os.environ.get("DISPLAY"):
            return False

        try:
            check_result = subprocess.run(
                ["xdpyinfo"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return check_result.returncode == 0

    def can_show_opencv_window(self) -> bool:
        """현재 환경에서 cv2.imshow를 사용해도 안전한지 반환한다."""
        return self.has_gui

    def label(self) -> str:
        """로그에 출력할 실행환경 이름을 만든다."""
        if self.is_windows:
            return "Windows GUI"
        if self.is_wsl and self.has_gui:
            return "WSL with GUI"
        if self.is_wsl:
            return "WSL headless"
        if self.is_linux and self.has_gui:
            return "Linux GUI"
        if self.is_linux:
            return "Linux headless"
        return self.system_name


class ResultDisplayManager:
    """결과 프레임을 GUI 창으로 보여 주거나 파일로 저장하는 클래스이다."""

    def __init__(self, config: RuntimeConfig, environment: EnvironmentInfo, fps: float) -> None:
        """출력 폴더, 동영상 writer, 창 이름 등 UI 상태를 초기화한다."""
        self.config = config
        self.environment = environment
        self.fps = fps
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.window_name = f"Computer Vision Report - {config.mode}"
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_path: Optional[Path] = None
        self.last_image_path: Optional[Path] = None

    def handle_frame(self, edges: np.ndarray, overlay_frame: np.ndarray, frame_index: int) -> bool:
        """한 프레임의 결과를 저장하고, GUI 가능 환경이면 창에 표시한다."""
        combined_frame = make_combined_frame(edges, overlay_frame)
        self._ensure_video_writer(combined_frame)
        self._write_video_frame(combined_frame)

        if frame_index == 0 or frame_index % self.config.save_every_n_frames == 0:
            self.last_image_path = self._save_image(combined_frame, frame_index)
            if not self.environment.can_show_opencv_window():
                print(f"INFO: result image saved: {self.last_image_path}")

        if not self.environment.can_show_opencv_window():
            return True

        cv2.imshow(self.window_name, combined_frame)
        key = cv2.waitKey(self.config.wait_delay_ms) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        """동영상 writer와 OpenCV 창을 정리한다."""
        if self.video_writer is not None:
            self.video_writer.release()
        if self.environment.can_show_opencv_window():
            cv2.destroyAllWindows()

    def _ensure_video_writer(self, frame: np.ndarray) -> None:
        """출력 프레임 크기를 알게 된 시점에 동영상 writer를 생성한다."""
        if self.video_writer is not None:
            return

        height, width = frame.shape[:2]
        self.video_path = self.output_dir / f"result_{self.config.mode}_{self.timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))

        if not self.video_writer.isOpened():
            self.video_path = self.output_dir / f"result_{self.config.mode}_{self.timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))

        if not self.video_writer.isOpened():
            self.video_writer = None
            self.video_path = None
            print("WARNING: result video writer could not be opened; images will still be saved.")

    def _write_video_frame(self, frame: np.ndarray) -> None:
        """사용 가능한 동영상 writer가 있을 때 프레임을 기록한다."""
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def _save_image(self, frame: np.ndarray, frame_index: int) -> Path:
        """현재 결합 결과 프레임을 PNG 파일로 저장한다."""
        image_path = self.output_dir / f"result_{self.config.mode}_{self.timestamp}_{frame_index:06d}.png"
        cv2.imwrite(str(image_path), frame)
        return image_path
