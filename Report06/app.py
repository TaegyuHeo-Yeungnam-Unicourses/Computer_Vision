"""
app.py
입력, 처리, 출력을 조립해 전체 실행 흐름을 관리한다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

from pipeline import FrameProcessor
from source import FrameSource
from ui import ResultDisplayManager


class ConsoleReporter:
    """프로그램 시작, 제한 도달, 종료 메시지를 출력한다."""

    def print_start(self, config, environment):
        """프로그램 시작 정보를 출력한다."""
        print("INFO: mode = " + config.mode)
        print("INFO: input type = " + config.input_type)
        print("INFO: input source = " + config.input_source)
        print("INFO: environment = " + environment.label())

    def print_frame_limit(self, max_frames):
        """프레임 제한 도달 메시지를 출력한다."""
        print("INFO: reached max frame limit: " + str(max_frames))

    def print_finish(self, display, processed_frames):
        """프로그램 종료 정보를 출력한다."""
        print("DONE: processed frames = " + str(processed_frames))
        if display is None:
            return
        if display.last_image_path is not None:
            print("DONE: last result image = " + str(display.last_image_path))
        if display.video_path is not None:
            print("DONE: result video = " + str(display.video_path))


class ShapeDetectionApplication:
    """입력, 처리, 출력을 순서대로 실행하는 클래스이다."""

    def __init__(self, config, settings, environment, reporter=None):
        """실행 설정과 의존 객체를 저장한다."""
        self.config = config
        self.settings = settings
        self.environment = environment
        self.reporter = reporter or ConsoleReporter()
        self.source = FrameSource(config.input_type, config.input_source, settings.input)
        self.processor = FrameProcessor(config, settings)

    def run(self):
        """프레임을 반복해서 읽고 처리한다."""
        display = None
        processed_frames = 0

        try:
            self.source.open()
            display = ResultDisplayManager(
                self.config,
                self.environment,
                self.source.fps(),
                self.settings.display,
            )
            max_frames = self.effective_max_frames()
            self.reporter.print_start(self.config, self.environment)
            processed_frames = self.process_frames(display, max_frames)
        finally:
            if display is not None:
                display.close()
            self.source.release()

        self.reporter.print_finish(display, processed_frames)

    def process_frames(self, display, max_frames):
        """입력 프레임을 하나씩 처리한다."""
        frame_index = 0

        while True:
            ok, frame = self.source.read()
            if not ok:
                break
            if frame is None:
                break

            edges, overlay = self.processor.process(frame)
            should_continue = display.handle_frame(edges, overlay, frame_index)

            frame_index = frame_index + 1
            if not should_continue:
                break
            if max_frames > 0 and frame_index >= max_frames:
                self.reporter.print_frame_limit(max_frames)
                break

        return frame_index

    def effective_max_frames(self):
        """실제로 사용할 최대 프레임 수를 정한다."""
        if self.config.max_frames > 0:
            return self.config.max_frames
        if self.source.is_webcam() and not self.environment.can_show_opencv_window():
            return self.settings.display.headless_webcam_frame_limit
        return 0
