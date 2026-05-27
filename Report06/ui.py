"""
ui.py
결과 화면을 만들고, 화면에 보여 주거나 파일로 저장한다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

import os
import platform
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def draw_label(image, text, position=(20, 35)):
    """영상 왼쪽 위에 글자를 그린다."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


def make_combined_frame(edges, overlay):
    """Canny 결과와 검출 결과를 좌우로 붙인다."""
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    draw_label(edges_bgr, "Canny Edge Detection")
    return np.hstack((edges_bgr, overlay))


class EnvironmentInfo:
    """현재 컴퓨터에서 OpenCV 창을 열 수 있는지 저장한다."""

    def __init__(self):
        """운영체제와 GUI 가능 여부를 확인해 저장한다."""
        self.system_name = platform.system()
        self.is_wsl = self.check_wsl()
        self.has_gui = self.check_gui()

    def check_wsl(self):
        """Linux가 WSL인지 확인한다."""
        if self.system_name.lower() != "linux":
            return False

        try:
            version_text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False

        version_text = version_text.lower()
        if "microsoft" in version_text or "wsl" in version_text:
            return True
        return False

    def check_gui(self):
        """OpenCV 창을 열 수 있는 환경인지 확인한다."""
        system_name = self.system_name.lower()
        if system_name == "windows" or system_name == "darwin":
            return True

        gui_requested = os.environ.get("OPENCV_GUI", "").lower()
        has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")

        if gui_requested in ["1", "true", "yes", "on"] and has_display:
            return True
        return False

    def can_show_opencv_window(self):
        """OpenCV 창을 사용할 수 있으면 True를 반환한다."""
        return self.has_gui

    def label(self):
        """현재 실행 환경 이름을 문자열로 반환한다."""
        if self.is_wsl and self.has_gui:
            return "WSL GUI"
        if self.is_wsl:
            return "WSL headless"
        if self.has_gui:
            return self.system_name + " GUI"
        return self.system_name + " headless"


class ResultDisplayManager:
    """결과 프레임을 저장하고, 가능하면 OpenCV 창에도 보여 준다."""

    def __init__(self, config, environment, fps):
        """저장 폴더, 창 이름, 동영상 저장 상태를 준비한다."""
        self.config = config
        self.environment = environment
        self.fps = fps
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.window_name = "Hough Detection - " + config.mode
        self.video_writer = None
        self.video_writer_checked = False
        self.video_path = None
        self.last_image_path = None

    def handle_frame(self, edges, overlay, frame_index):
        """프레임 하나를 저장하고 화면 표시가 가능하면 보여 준다."""
        combined = make_combined_frame(edges, overlay)
        self.open_video_writer_if_needed(combined)
        self.write_video(combined)

        if frame_index == 0 or frame_index % self.config.save_every_n_frames == 0:
            self.last_image_path = self.save_image(combined, frame_index)
            if not self.environment.can_show_opencv_window():
                print("INFO: result image saved: " + str(self.last_image_path))

        if not self.environment.can_show_opencv_window():
            return True

        cv2.imshow(self.window_name, combined)
        key = cv2.waitKey(self.config.wait_delay_ms) & 0xFF
        if key == ord("q") or key == 27:
            return False
        return True

    def close(self):
        """동영상 저장기와 OpenCV 창을 정리한다."""
        if self.video_writer is not None:
            self.video_writer.release()
        if self.environment.can_show_opencv_window():
            cv2.destroyAllWindows()

    def open_video_writer_if_needed(self, frame):
        """첫 프레임 크기에 맞춰 동영상 저장기를 만든다."""
        if self.video_writer_checked:
            return
        self.video_writer_checked = True

        height, width = frame.shape[:2]
        self.video_path = self.output_dir / ("result_" + self.config.mode + "_" + self.timestamp + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))

        if writer.isOpened():
            self.video_writer = writer
        else:
            self.video_path = None
            writer.release()
            print("WARNING: result video writer could not be opened; PNG images will still be saved.")

    def write_video(self, frame):
        """동영상 저장기가 있으면 프레임을 기록한다."""
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def save_image(self, frame, frame_index):
        """현재 프레임을 PNG 이미지로 저장한다."""
        file_name = "result_" + self.config.mode + "_" + self.timestamp + "_" + str(frame_index).zfill(6) + ".png"
        image_path = self.output_dir / file_name
        cv2.imwrite(str(image_path), frame)
        return image_path
