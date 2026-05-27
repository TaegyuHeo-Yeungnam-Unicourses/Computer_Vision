"""
form.py
실행할 때 필요한 설정값을 만든다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

import argparse
from pathlib import Path


class RuntimeConfig:
    """프로그램 실행 중에 사용할 설정값을 저장하는 클래스이다."""

    def __init__(
        self,
        mode,
        input_type,
        input_source,
        output_dir,
        lane_filter_enabled,
        circle_top_right_priority_enabled,
        max_frames,
        wait_delay_ms,
        save_every_n_frames,
    ):
        """설정값을 객체 안에 저장한다."""
        self.mode = mode
        self.input_type = input_type
        self.input_source = input_source
        self.output_dir = output_dir
        self.lane_filter_enabled = lane_filter_enabled
        self.circle_top_right_priority_enabled = circle_top_right_priority_enabled
        self.max_frames = max_frames
        self.wait_delay_ms = wait_delay_ms
        self.save_every_n_frames = save_every_n_frames


class ConfigDefaults:
    """main.py 위쪽에 있는 기본 설정값을 저장하는 클래스이다."""

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
        wait_delay_ms,
        save_every_n_frames,
    ):
        """기본 설정값을 객체 안에 저장한다."""
        self.mode = mode
        self.input_type = input_type
        self.input_source = input_source
        self.webcam_source = webcam_source
        self.output_dir = output_dir
        self.lane_filter_enabled = lane_filter_enabled
        self.circle_top_right_priority_enabled = circle_top_right_priority_enabled
        self.max_frames = max_frames
        self.wait_delay_ms = wait_delay_ms
        self.save_every_n_frames = save_every_n_frames


def normalize_mode(value):
    """검출 모드 문자열을 검사하고 소문자로 바꾼다."""
    mode = value.strip().lower()
    if mode != "straight" and mode != "circle":
        raise ValueError("ERROR: mode must be 'straight' or 'circle'.")
    return mode


def normalize_input_type(value):
    """입력 종류 문자열을 검사하고 소문자로 바꾼다."""
    input_type = value.strip().lower()
    if input_type != "file" and input_type != "webcam":
        raise ValueError("ERROR: input type must be 'file' or 'webcam'.")
    return input_type


def parse_bool_text(value, default_value):
    """on/off, yes/no 같은 문자열을 True 또는 False로 바꾼다."""
    if value is None:
        return default_value

    text = value.strip().lower()
    if text in ["1", "true", "t", "yes", "y", "on"]:
        return True
    if text in ["0", "false", "f", "no", "n", "off"]:
        return False
    raise ValueError("ERROR: boolean value expected, but got '" + value + "'.")


def parse_non_negative_int(value, default_value):
    """문자열을 0 이상의 정수로 바꾼다."""
    if value is None:
        return default_value
    if value.strip() == "":
        return default_value

    try:
        number = int(value)
    except ValueError:
        raise ValueError("ERROR: non-negative integer expected, but got '" + value + "'.")

    if number < 0:
        raise ValueError("ERROR: non-negative integer expected, but got '" + value + "'.")
    return number


def ask_text(message, default_value):
    """사용자에게 문자열을 입력받고, 빈 입력이면 기본값을 사용한다."""
    answer = input(message + " [" + str(default_value) + "]: ").strip()
    if answer == "":
        return default_value
    return answer


def ask_choice(message, choices, default_value):
    """사용자에게 정해진 선택지 중 하나를 입력받는다."""
    choice_text = "/".join(choices)
    while True:
        answer = ask_text(message + " (" + choice_text + ")", default_value).lower()
        if answer in choices:
            return answer
        print("잘못된 입력이다. " + choice_text + " 중 하나를 입력한다.")


def ask_bool(message, default_value):
    """사용자에게 y 또는 n을 입력받아 True 또는 False로 바꾼다."""
    if default_value:
        default_text = "y"
    else:
        default_text = "n"

    while True:
        answer = ask_text(message + " (y/n)", default_text)
        try:
            return parse_bool_text(answer, default_value)
        except ValueError:
            print("잘못된 입력이다. y 또는 n으로 입력한다.")


def ask_non_negative_int(message, default_value):
    """사용자에게 0 이상의 정수를 입력받는다."""
    while True:
        answer = ask_text(message, str(default_value))
        try:
            return parse_non_negative_int(answer, default_value)
        except ValueError:
            print("잘못된 입력이다. 0 이상의 정수를 입력한다.")


class UserInputForm:
    """기본값, 명령행 인자, 선택 입력을 합쳐 최종 설정을 만든다."""

    def __init__(self, defaults, argv=None):
        """기본 설정과 명령행 인자를 저장한다."""
        self.defaults = defaults
        self.argv = argv

    def collect_config(self):
        """최종 실행 설정을 만들어 반환한다."""
        args = self.parse_arguments()

        mode = normalize_mode(args.mode or self.defaults.mode)
        input_type = normalize_input_type(args.input_type or self.defaults.input_type)
        input_source = self.default_source(input_type, args.source)
        lane_filter_enabled = parse_bool_text(args.lane_filter, self.defaults.lane_filter_enabled)
        circle_priority_enabled = parse_bool_text(
            args.circle_priority,
            self.defaults.circle_top_right_priority_enabled,
        )
        max_frames = parse_non_negative_int(args.max_frames, self.defaults.max_frames)

        if args.form:
            values = self.ask_user(
                mode,
                input_type,
                input_source,
                lane_filter_enabled,
                circle_priority_enabled,
                max_frames,
            )
            mode = values[0]
            input_type = values[1]
            input_source = values[2]
            lane_filter_enabled = values[3]
            circle_priority_enabled = values[4]
            max_frames = values[5]

        return RuntimeConfig(
            mode,
            input_type,
            input_source,
            Path(self.defaults.output_dir),
            lane_filter_enabled,
            circle_priority_enabled,
            max_frames,
            self.defaults.wait_delay_ms,
            self.defaults.save_every_n_frames,
        )

    def parse_arguments(self):
        """명령행 인자를 읽는다."""
        parser = argparse.ArgumentParser(description="YCbCr Y + Canny + Hough 도형 검출")
        parser.add_argument("--mode", "--shape", dest="mode", help="검출 모드: straight 또는 circle")
        parser.add_argument("--input-type", choices=["file", "webcam"], help="입력 종류")
        parser.add_argument("--source", help="파일 경로 또는 웹캠 번호")
        parser.add_argument("--lane-filter", help="차선 필터: on/off")
        parser.add_argument("--circle-priority", help="원 검출 시 우측 상단 우선: on/off")
        parser.add_argument("--max-frames", help="최대 처리 프레임 수. 0이면 제한 없음")
        parser.add_argument("--form", action="store_true", help="터미널 입력 폼을 사용한다")
        return parser.parse_args(self.argv)

    def default_source(self, input_type, argument_source):
        """입력 종류에 맞는 기본 입력 소스를 고른다."""
        if argument_source is not None and argument_source.strip() != "":
            return argument_source.strip()
        if input_type == "webcam":
            return self.defaults.webcam_source
        return self.defaults.input_source

    def ask_user(
        self,
        mode,
        input_type,
        input_source,
        lane_filter_enabled,
        circle_priority_enabled,
        max_frames,
    ):
        """터미널에서 사용자에게 설정값을 다시 입력받는다."""
        print("\n[입력 설정]")
        print("Enter만 누르면 현재 기본값을 사용한다.")

        mode = ask_choice("검출 모드", ["straight", "circle"], mode)
        input_type = ask_choice("입력 종류", ["file", "webcam"], input_type)

        if input_type == "webcam":
            default_source = self.defaults.webcam_source
        else:
            default_source = input_source
        input_source = ask_text("파일 경로 또는 웹캠 번호", default_source)

        if mode == "straight":
            lane_filter_enabled = ask_bool("차선 필터를 사용할지 선택한다", lane_filter_enabled)
        else:
            circle_priority_enabled = ask_bool("원 검출 시 우측 상단 우선을 사용할지 선택한다", circle_priority_enabled)

        max_frames = ask_non_negative_int("최대 처리 프레임 수. 0은 제한 없음", max_frames)
        print()
        return mode, input_type, input_source, lane_filter_enabled, circle_priority_enabled, max_frames


if __name__ == "__main__":
    defaults = ConfigDefaults(
        "straight",
        "file",
        "./test_video.mp4",
        "0",
        "output",
        True,
        False,
        0,
        1,
        30,
    )
    print(UserInputForm(defaults).collect_config())
