"""
form.py
사용자 입력, 명령행 인자, 환경변수를 RuntimeConfig로 정리하는 모듈이다.
main.py는 이 모듈에서 만든 설정 객체만 받아 분석을 수행한다.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class RuntimeConfig:
    """프로그램 실행에 필요한 모든 설정값을 보관하는 불변 데이터 클래스이다."""

    mode: str
    input_source: str
    output_dir: Path
    lane_filter_enabled: bool
    circle_top_right_priority_enabled: bool
    max_frames: int = 0

    # Canny Edge Detection 관련 설정이다.
    canny_low_threshold: int = 80
    canny_high_threshold: int = 160

    # Hough Line Detection 관련 설정이다.
    line_rho: float = 1.0
    line_theta: float = 3.141592653589793 / 180.0
    line_threshold: int = 60
    line_min_length: int = 60
    line_max_gap: int = 20

    # 차선형 직선 필터 관련 설정이다.
    lane_min_abs_angle: float = 20.0
    lane_max_abs_angle: float = 75.0
    lane_min_y_ratio: float = 0.45

    # Hough Circle Detection 관련 설정이다.
    circle_dp: float = 1.2
    circle_min_dist_ratio: float = 0.08
    circle_param1: int = 120
    circle_param2: int = 30
    circle_min_radius: int = 8
    circle_max_radius: int = 0

    # 원 후보 점수 계산 관련 설정이다.
    circle_sample_count: int = 180
    circle_edge_probe_radius: int = 2
    circle_top_right_weight: float = 0.35

    # GUI 표시와 파일 저장 관련 설정이다.
    wait_delay_ms: int = 1
    save_every_n_frames: int = 30


@dataclass(frozen=True)
class ConfigDefaults:
    """main.py에 위치한 기본 설정값을 form.py로 전달하기 위한 데이터 클래스이다."""

    shape: str
    input_source: str
    output_dir: str
    lane_filter_enabled: bool
    circle_top_right_priority_enabled: bool
    max_frames: int


def normalize_shape(raw_value: str) -> str:
    """shape 문자열을 소문자로 정규화하고 straight 또는 circle인지 검증한다."""
    normalized_value = raw_value.strip().lower()
    if normalized_value not in {"straight", "circle"}:
        raise ValueError(f"ERROR: shape must be one of [straight, circle], but got '{raw_value}'.")
    return normalized_value


def parse_bool_text(raw_value: Optional[str], default_value: bool) -> bool:
    """문자열 형태의 on/off, true/false, 1/0 값을 bool 값으로 변환한다."""
    if raw_value is None:
        return default_value

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise ValueError(f"ERROR: boolean value expected, but got '{raw_value}'.")


def parse_non_negative_int(raw_value: Optional[str], default_value: int) -> int:
    """문자열을 0 이상의 정수로 변환하고, 값이 없으면 기본값을 반환한다."""
    if raw_value is None or str(raw_value).strip() == "":
        return default_value

    try:
        parsed_value = int(str(raw_value).strip())
    except ValueError as error:
        raise ValueError(f"ERROR: non-negative integer expected, but got '{raw_value}'.") from error

    if parsed_value < 0:
        raise ValueError(f"ERROR: non-negative integer expected, but got '{raw_value}'.")
    return parsed_value


def choose_first_text(*values: Optional[str], default_value: str) -> str:
    """비어 있지 않은 문자열 중 첫 번째 값을 선택하고, 없으면 기본값을 반환한다."""
    for value in values:
        if value is not None and value.strip() != "":
            return value.strip()
    return default_value


def ask_text(prompt_message: str, default_value: str) -> str:
    """터미널에서 문자열 입력을 받고, 빈 입력이면 기본값을 반환한다."""
    answer = input(f"{prompt_message} [{default_value}]: ").strip()
    return answer if answer else default_value


def ask_bool(prompt_message: str, default_value: bool) -> bool:
    """터미널에서 y/n 형태의 bool 입력을 받고, 빈 입력이면 기본값을 반환한다."""
    default_text = "y" if default_value else "n"
    while True:
        answer = input(f"{prompt_message} (y/n) [{default_text}]: ").strip()
        if answer == "":
            return default_value
        try:
            return parse_bool_text(answer, default_value)
        except ValueError:
            print("잘못된 입력이다. y 또는 n으로 입력한다.")


def ask_non_negative_int(prompt_message: str, default_value: int) -> int:
    """터미널에서 0 이상의 정수를 입력받고, 빈 입력이면 기본값을 반환한다."""
    while True:
        answer = input(f"{prompt_message} [{default_value}]: ").strip()
        if answer == "":
            return default_value
        try:
            return parse_non_negative_int(answer, default_value)
        except ValueError:
            print("잘못된 입력이다. 0 이상의 정수를 입력한다.")


def ask_shape(prompt_message: str, default_value: str) -> str:
    """터미널에서 straight 또는 circle 값을 입력받고 검증된 값을 반환한다."""
    while True:
        answer = input(f"{prompt_message} [straight/circle, 기본: {default_value}]: ").strip()
        raw_value = answer if answer else default_value
        try:
            return normalize_shape(raw_value)
        except ValueError:
            print("잘못된 입력이다. straight 또는 circle 중 하나를 입력한다.")


class UserInputForm:
    """사용자 입력, 명령행 인자, 환경변수를 하나의 RuntimeConfig로 조립하는 클래스이다."""

    def __init__(self, defaults: ConfigDefaults, argv: Optional[Sequence[str]] = None) -> None:
        """기본 설정과 선택적인 argv 값을 저장한다."""
        self.defaults = defaults
        self.argv = list(argv) if argv is not None else None

    def collect_config(self) -> RuntimeConfig:
        """명령행 인자, 환경변수, 터미널 입력을 순서대로 반영하여 RuntimeConfig를 만든다."""
        args = self._parse_arguments()

        mode = normalize_shape(
            choose_first_text(
                args.shape,
                os.environ.get("shape"),
                os.environ.get("SHAPE"),
                default_value=self.defaults.shape,
            )
        )
        input_source = choose_first_text(
            args.source,
            os.environ.get("VIDEO_SOURCE"),
            default_value=self.defaults.input_source,
        )
        lane_filter_enabled = self._resolve_bool(
            args.lane_filter,
            "LANE_FILTER_ENABLED",
            self.defaults.lane_filter_enabled,
        )
        circle_priority_enabled = self._resolve_bool(
            args.circle_priority,
            "CIRCLE_TOP_RIGHT_PRIORITY_ENABLED",
            self.defaults.circle_top_right_priority_enabled,
        )
        max_frames = parse_non_negative_int(
            choose_first_text(args.max_frames, os.environ.get("MAX_FRAMES"), default_value=str(self.defaults.max_frames)),
            self.defaults.max_frames,
        )

        if self._should_show_interactive_form(args):
            mode, input_source, lane_filter_enabled, circle_priority_enabled, max_frames = self._collect_interactive_values(
                mode,
                input_source,
                lane_filter_enabled,
                circle_priority_enabled,
                max_frames,
            )

        return RuntimeConfig(
            mode=mode,
            input_source=input_source,
            output_dir=Path(self.defaults.output_dir),
            lane_filter_enabled=lane_filter_enabled,
            circle_top_right_priority_enabled=circle_priority_enabled,
            max_frames=max_frames,
        )

    def _parse_arguments(self) -> argparse.Namespace:
        """main.py가 받은 명령행 인자를 파싱한다."""
        parser = argparse.ArgumentParser(
            description="YCbCr Y 성분 기반 Canny + Hough 직선/원 검출 프로그램",
        )
        parser.add_argument("--shape", help="검출 모드: straight 또는 circle")
        parser.add_argument("--source", help="입력 소스: 카메라 번호, 동영상 경로, 이미지 경로")
        parser.add_argument("--lane-filter", help="차선 필터 사용 여부: on/off, true/false, 1/0")
        parser.add_argument("--circle-priority", help="원 검출 시 우측 상단 우선 여부: on/off, true/false, 1/0")
        parser.add_argument("--max-frames", help="최대 처리 프레임 수. 0이면 제한 없음")
        parser.add_argument("--no-form", action="store_true", help="터미널 대화형 입력을 생략한다")
        parser.add_argument("--form", action="store_true", help="터미널 대화형 입력을 명시적으로 사용한다")
        return parser.parse_args(self.argv)

    def _resolve_bool(self, argument_value: Optional[str], environment_name: str, default_value: bool) -> bool:
        """명령행 인자와 환경변수 중 우선순위가 높은 bool 설정을 선택한다."""
        if argument_value is not None:
            return parse_bool_text(argument_value, default_value)
        return parse_bool_text(os.environ.get(environment_name), default_value)

    def _should_show_interactive_form(self, args: argparse.Namespace) -> bool:
        """현재 실행이 터미널 입력을 받을 수 있는지 판단한다."""
        if args.no_form:
            return False
        if args.form:
            return True
        return sys.stdin.isatty()

    def _collect_interactive_values(
        self,
        mode: str,
        input_source: str,
        lane_filter_enabled: bool,
        circle_priority_enabled: bool,
        max_frames: int,
    ) -> tuple[str, str, bool, bool, int]:
        """터미널 대화형 입력으로 설정값을 최종 보정한다."""
        print("\n[입력 설정 Form]")
        print("Enter만 누르면 대괄호 안의 기본값을 사용한다.")
        mode = ask_shape("검출 모드를 선택한다", mode)
        input_source = ask_text("입력 소스를 입력한다. 예: 0, ./road.mp4, ./image.jpg", input_source)

        if mode == "straight":
            lane_filter_enabled = ask_bool("직선 모드의 차선 필터를 사용할지 선택한다", lane_filter_enabled)
        else:
            circle_priority_enabled = ask_bool("원 모드의 우측 상단 우선 옵션을 사용할지 선택한다", circle_priority_enabled)

        max_frames = ask_non_negative_int("최대 처리 프레임 수를 입력한다. 0은 제한 없음", max_frames)
        print()
        return mode, input_source, lane_filter_enabled, circle_priority_enabled, max_frames


if __name__ == "__main__":
    """form.py를 단독 실행할 때 현재 입력 설정 결과를 확인한다."""
    defaults = ConfigDefaults(
        shape="straight",
        input_source="0",
        output_dir="output",
        lane_filter_enabled=True,
        circle_top_right_priority_enabled=False,
        max_frames=0,
    )
    print(UserInputForm(defaults).collect_config())
