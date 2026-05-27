"""
detectors.py
Canny 에지 결과에서 Hough 직선 또는 원 후보를 찾고 화면에 그린다.

어려운 문법을 줄이기 위해 타입 힌트, dataclass, 데코레이터를 사용하지 않는다.
"""

import math

import cv2
import numpy as np

from ui import draw_label


class LineCandidate:
    """검출된 직선 후보를 저장하고 기하 계산을 담당한다."""

    def __init__(self, x1, y1, x2, y2, score=0.0):
        """직선 양 끝점과 점수를 저장한다."""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score

    def start(self):
        """직선 시작 좌표를 반환한다."""
        return self.x1, self.y1

    def end(self):
        """직선 끝 좌표를 반환한다."""
        return self.x2, self.y2

    def length(self):
        """직선 길이를 반환한다."""
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def middle_y(self):
        """직선 중점의 y 좌표를 반환한다."""
        return (self.y1 + self.y2) / 2.0

    def abs_angle(self):
        """직선 각도를 0도에서 90도 사이 값으로 반환한다."""
        angle = abs(math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1)))
        if angle > 90.0:
            angle = 180.0 - angle
        return angle

    def is_lane_angle(self, settings):
        """직선 각도가 차선처럼 보이는 대각선 범위인지 확인한다."""
        angle = self.abs_angle()
        return angle >= settings.lane_min_abs_angle and angle <= settings.lane_max_abs_angle


class CircleCandidate:
    """검출된 원 후보를 저장한다."""

    def __init__(self, x, y, radius, score):
        """원의 중심, 반지름, 점수를 저장한다."""
        self.x = x
        self.y = y
        self.radius = radius
        self.score = score

    def center(self):
        """원의 중심 좌표를 반환한다."""
        return self.x, self.y


class LineCandidateFactory:
    """HoughLinesP 원시 결과를 점수가 있는 직선 후보로 바꾼다."""

    def __init__(self, config, settings):
        """실행 설정과 직선 검출 설정을 저장한다."""
        self.config = config
        self.settings = settings

    def make(self, raw_line, image_height):
        """원시 직선 하나를 후보 객체로 바꾼다."""
        candidate = LineCandidate(
            int(raw_line[0]),
            int(raw_line[1]),
            int(raw_line[2]),
            int(raw_line[3]),
        )

        if self.config.lane_filter_enabled and not candidate.is_lane_angle(self.settings):
            return None

        candidate.score = self.score(candidate, image_height)
        return candidate

    def score(self, candidate, image_height):
        """직선 길이와 위치를 이용해 후보 점수를 계산한다."""
        score = candidate.length()
        if self.config.lane_filter_enabled and candidate.middle_y() >= image_height / 2.0:
            score = score * self.settings.lower_half_score_weight
        return score


class CircleCandidateFactory:
    """HoughCircles 원시 결과를 점수가 있는 원 후보로 바꾼다."""

    def __init__(self, config, settings):
        """실행 설정과 원 검출 설정을 저장한다."""
        self.config = config
        self.settings = settings

    def make(self, x, y, radius, edges, frame_shape):
        """원 중심과 반지름을 후보 객체로 바꾼다."""
        support = self.circle_support_score(edges, x, y, radius)
        score = support

        if self.config.circle_top_right_priority_enabled:
            position = self.upper_right_score(x, y, frame_shape)
            score = (1.0 - self.settings.top_right_weight) * support
            score = score + self.settings.top_right_weight * position

        return CircleCandidate(x, y, radius, score)

    def upper_right_score(self, x, y, frame_shape):
        """우측 상단에 가까운 원일수록 높은 위치 점수를 준다."""
        height = frame_shape[0]
        width = frame_shape[1]
        distance = math.hypot(width - x, y)
        max_distance = math.hypot(width, height)
        score = 1.0 - distance / max_distance
        if score < 0.0:
            score = 0.0
        return score

    def circle_support_score(self, edges, center_x, center_y, radius):
        """원 둘레 샘플 지점에 에지가 얼마나 많이 있는지 계산한다."""
        height = edges.shape[0]
        width = edges.shape[1]
        hit_count = 0
        sample_count = 0

        angles = np.linspace(0.0, 2.0 * math.pi, self.settings.sample_count, endpoint=False)
        for angle in angles:
            x = int(round(center_x + radius * math.cos(angle)))
            y = int(round(center_y + radius * math.sin(angle)))

            if x >= 0 and x < width and y >= 0 and y < height:
                sample_count = sample_count + 1
                if self.has_nearby_edge(edges, x, y):
                    hit_count = hit_count + 1

        if sample_count == 0:
            return 0.0
        return hit_count / sample_count

    def has_nearby_edge(self, edges, x, y):
        """한 좌표 주변에 Canny 에지가 있는지 확인한다."""
        height = edges.shape[0]
        width = edges.shape[1]
        radius = self.settings.edge_probe_radius

        x1 = max(0, x - radius)
        x2 = min(width, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(height, y + radius + 1)

        if np.any(edges[y1:y2, x1:x2] > 0):
            return True
        return False


class HoughLineDetector:
    """Canny 에지 영상에서 Hough 직선을 찾고 그린다."""

    def __init__(self, config, settings):
        """실행 설정과 직선 검출 설정을 저장한다."""
        self.config = config
        self.settings = settings
        self.candidate_factory = LineCandidateFactory(config, settings)

    def create_overlay(self, frame, y_channel, edges):
        """직선을 검출하고 프레임 위에 그린다."""
        lines = self.detect(edges, frame.shape)
        return self.draw(frame, lines)

    def detect(self, edges, frame_shape):
        """점수가 높은 직선 후보들을 찾는다."""
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=self.settings.rho,
            theta=self.settings.theta,
            threshold=self.settings.threshold,
            minLineLength=self.settings.min_length,
            maxLineGap=self.settings.max_gap,
        )

        if raw_lines is None:
            return []

        image_height = frame_shape[0]
        candidates = []
        for raw_line in raw_lines:
            candidate = self.candidate_factory.make(raw_line[0], image_height)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=candidate_score, reverse=True)
        return candidates[:self.settings.max_results]

    def draw(self, frame, lines):
        """선택된 직선을 프레임 위에 그린다."""
        output = frame.copy()

        rank = 1
        for line in lines:
            cv2.line(output, line.start(), line.end(), (0, 0, 255), 3)
            cv2.putText(
                output,
                "#" + str(rank) + " " + str(round(line.score)),
                line.start(),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            rank = rank + 1

        if self.config.lane_filter_enabled:
            lane_text = "ON"
        else:
            lane_text = "OFF"

        label = "Top " + str(len(lines)) + "/" + str(self.settings.max_results)
        label = label + " Lines | lane filter: " + lane_text
        draw_label(output, label)
        return output


class HoughCircleDetector:
    """밝기 영상과 Canny 에지 영상에서 Hough 원을 찾고 그린다."""

    def __init__(self, config, settings):
        """실행 설정과 원 검출 설정을 저장한다."""
        self.config = config
        self.settings = settings
        self.candidate_factory = CircleCandidateFactory(config, settings)

    def create_overlay(self, frame, y_channel, edges):
        """원을 검출하고 프레임 위에 그린다."""
        circles = self.detect(y_channel, edges, frame.shape)
        return self.draw(frame, circles)

    def detect(self, y_channel, edges, frame_shape):
        """점수가 높은 원 후보들을 찾는다."""
        height = frame_shape[0]
        width = frame_shape[1]
        min_distance = max(10, int(min(height, width) * self.settings.min_dist_ratio))
        blurred = cv2.medianBlur(y_channel, self.settings.median_blur_kernel_size)

        raw_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.settings.dp,
            minDist=min_distance,
            param1=self.settings.param1,
            param2=self.settings.param2,
            minRadius=self.settings.min_radius,
            maxRadius=self.settings.max_radius,
        )

        if raw_circles is None:
            return []

        rounded = np.round(raw_circles[0]).astype(int)
        candidates = []
        for circle in rounded:
            x = int(circle[0])
            y = int(circle[1])
            radius = int(circle[2])
            if radius > 0:
                candidate = self.candidate_factory.make(x, y, radius, edges, frame_shape)
                candidates.append(candidate)

        candidates.sort(key=candidate_score, reverse=True)
        return candidates[:self.settings.max_results]

    def draw(self, frame, circles):
        """선택된 원을 프레임 위에 그린다."""
        output = frame.copy()

        rank = 1
        for circle in circles:
            cv2.circle(output, circle.center(), circle.radius, (0, 255, 0), 3)
            cv2.circle(output, circle.center(), 3, (0, 0, 255), -1)
            cv2.putText(
                output,
                "#" + str(rank) + " " + str(round(circle.score, 2)),
                (circle.x + 10, circle.y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            rank = rank + 1

        if self.config.circle_top_right_priority_enabled:
            score_text = "upper-right"
        else:
            score_text = "edge support"

        label = "Top " + str(len(circles)) + "/" + str(self.settings.max_results)
        label = label + " Circles | score: " + score_text
        draw_label(output, label)
        return output


class ShapeDetectorFactory:
    """실행 모드에 맞는 검출 객체를 만든다."""

    def __init__(self, config, settings):
        """실행 설정과 전체 설정을 저장한다."""
        self.config = config
        self.settings = settings

    def create(self):
        """직선 또는 원 검출 객체를 반환한다."""
        if self.config.mode == "straight":
            return HoughLineDetector(self.config, self.settings.line)
        return HoughCircleDetector(self.config, self.settings.circle)


def candidate_score(candidate):
    """후보 정렬에 사용할 점수를 반환한다."""
    return candidate.score
