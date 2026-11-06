from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .comfort_agent import AgentState
from .config import Settings
from .models import (
    FramePoseResult,
    PoseDetection,
    PostureMetrics,
    SeatState,
    VehicleState,
)

from .comfort_evaluator import (
    ComfortAssessment,
)


SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass
class AgentOverlay:
    agent_state: str
    agent_message: str
    posture: PostureMetrics
    comfort: ComfortAssessment
    vehicle: VehicleState
    seat: SeatState


@dataclass
class InterfaceButton:
    action: str
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    primary: bool = False
    warning: bool = False
    disabled: bool = False

    def contains(
        self,
        x: int,
        y: int,
    ) -> bool:
        return (
            not self.disabled
            and self.x1 <= x <= self.x2
            and self.y1 <= y <= self.y2
        )


class PoseRenderer:
    PANEL_WIDTH = 430
    MINIMUM_INTERFACE_HEIGHT = 660

    COLOR_BACKGROUND = (27, 29, 35)
    COLOR_CAMERA_BACKGROUND = (18, 20, 25)
    COLOR_CARD = (43, 46, 55)
    COLOR_CARD_LIGHT = (53, 57, 67)
    COLOR_PRIMARY = (220, 128, 42)
    COLOR_SUCCESS = (72, 184, 112)
    COLOR_WARNING = (48, 154, 235)
    COLOR_DANGER = (80, 88, 225)
    COLOR_TEXT = (245, 246, 248)
    COLOR_MUTED = (173, 179, 190)
    COLOR_BORDER = (76, 81, 94)
    COLOR_DISABLED = (55, 58, 66)

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings
        self.buttons: list[InterfaceButton] = []

    def get_action_at(
        self,
        x: int,
        y: int,
    ) -> str | None:
        for button in self.buttons:
            if button.contains(x, y):
                return button.action

        return None

    def render(
        self,
        frame: Any,
        result: FramePoseResult,
        overlay: AgentOverlay | None = None,
    ) -> Any:
        interface_height = max(
            frame.shape[0],
            self.MINIMUM_INTERFACE_HEIGHT,
        )

        camera_canvas = np.full(
            (
                interface_height,
                frame.shape[1],
                3,
            ),
            self.COLOR_CAMERA_BACKGROUND,
            dtype=np.uint8,
        )

        camera_canvas[
            0:frame.shape[0],
            0:frame.shape[1],
        ] = frame

        for pose in result.poses:
            self._draw_pose(
                camera_canvas,
                pose,
                is_driver=pose is result.driver,
            )

        self._draw_camera_status(
            camera_canvas,
            result,
            overlay,
            original_frame_height=frame.shape[0],
        )

        panel = np.full(
            (
                interface_height,
                self.PANEL_WIDTH,
                3,
            ),
            self.COLOR_BACKGROUND,
            dtype=np.uint8,
        )

        self.buttons = []

        if overlay is not None:
            self._draw_assistant_panel(
                panel,
                overlay,
            )

        return np.hstack(
            (
                camera_canvas,
                panel,
            )
        )

    def _draw_camera_status(
        self,
        image: Any,
        result: FramePoseResult,
        overlay: AgentOverlay | None,
        original_frame_height: int,
    ) -> None:
        driver_detected = result.driver is not None

        status_text = (
            "DRIVER DETECTED"
            if driver_detected
            else "DRIVER NOT DETECTED"
        )

        status_color = (
            self.COLOR_SUCCESS
            if driver_detected
            else self.COLOR_DANGER
        )

        cv2.rectangle(
            image,
            (14, 14),
            (255, 52),
            (20, 22, 27),
            -1,
        )

        cv2.circle(
            image,
            (34, 33),
            7,
            status_color,
            -1,
            cv2.LINE_AA,
        )

        self._draw_text(
            image,
            status_text,
            51,
            40,
            0.52,
            self.COLOR_TEXT,
        )

        if overlay is not None:
            posture_text = self._posture_label(
                overlay.posture
            )

            cv2.rectangle(
                image,
                (14, 62),
                (315, 101),
                (20, 22, 27),
                -1,
            )

            self._draw_text(
                image,
                f"Posture: {posture_text}",
                28,
                87,
                0.50,
                self.COLOR_TEXT,
            )

        if image.shape[0] > original_frame_height:
            information_y = original_frame_height + 36

            self._draw_text(
                image,
                "Camera and pose perception",
                22,
                information_y,
                0.58,
                self.COLOR_TEXT,
                thickness=2,
            )

            self._draw_text(
                image,
                (
                    "For full comfort analysis, keep shoulders, "
                    "hips, knees and ankles visible."
                ),
                22,
                information_y + 31,
                0.44,
                self.COLOR_MUTED,
            )

            self._draw_text(
                image,
                (
                    f"People: {result.detected_people}   "
                    f"Inference: {result.processing_time_ms:.0f} ms"
                ),
                22,
                information_y + 61,
                0.44,
                self.COLOR_MUTED,
            )

    def _draw_assistant_panel(
        self,
        panel: Any,
        overlay: AgentOverlay,
    ) -> None:
        self._draw_text(
            panel,
            "SEAT COMFORT",
            24,
            35,
            0.74,
            self.COLOR_TEXT,
            thickness=2,
        )

        self._draw_text(
            panel,
            "Guided driver comfort assistant",
            24,
            59,
            0.45,
            self.COLOR_MUTED,
        )

        self._draw_progress_indicator(
            panel,
            overlay.agent_state,
        )

        self._draw_vehicle_status(
            panel,
            overlay.vehicle,
        )

        self._draw_message_card(
            panel,
            overlay,
        )

        self._draw_context_card(
            panel,
            overlay,
        )

        self._draw_buttons(
            panel,
            overlay,
        )

        self._draw_text(
            panel,
            "Prototype simulation. No real seat actuation.",
            24,
            panel.shape[0] - 13,
            0.39,
            self.COLOR_MUTED,
        )

    def _draw_progress_indicator(
        self,
        panel: Any,
        agent_state: str,
    ) -> None:
        stages = (
            "Comfort",
            "Concern",
            "Review",
            "Result",
        )

        state_to_stage = {
            AgentState.MONITORING.value: 0,
            AgentState.WAITING_FOR_COMFORT.value: 0,
            AgentState.WAITING_FOR_COMPLAINT.value: 1,
            AgentState.WAITING_FOR_CONFIRMATION.value: 2,
            AgentState.QUEUED_UNTIL_STOPPED.value: 2,
            AgentState.ADJUSTMENT_APPLIED.value: 3,
        }

        active_stage = state_to_stage.get(
            agent_state,
            0,
        )

        start_x = 29
        y = 84
        spacing = 96

        for index, stage in enumerate(stages):
            x = start_x + index * spacing

            color = (
                self.COLOR_PRIMARY
                if index <= active_stage
                else self.COLOR_BORDER
            )

            if index < len(stages) - 1:
                cv2.line(
                    panel,
                    (x + 14, y),
                    (x + spacing - 14, y),
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.circle(
                panel,
                (x, y),
                8,
                color,
                -1,
                cv2.LINE_AA,
            )

            self._draw_text(
                panel,
                stage,
                x - 23,
                y + 25,
                0.35,
                (
                    self.COLOR_TEXT
                    if index <= active_stage
                    else self.COLOR_MUTED
                ),
            )

    def _draw_vehicle_status(
        self,
        panel: Any,
        vehicle: VehicleState,
    ) -> None:
        top = 122
        bottom = 164

        parked = (
            vehicle.speed_kph <= 0.1
            and vehicle.parking_brake
        )

        color = (
            self.COLOR_SUCCESS
            if parked
            else self.COLOR_WARNING
        )

        text = (
            "Parked: adjustment available"
            if parked
            else (
                f"Moving at {vehicle.speed_kph:.0f} km/h: "
                "adjustment locked"
            )
        )

        cv2.rectangle(
            panel,
            (20, top),
            (self.PANEL_WIDTH - 20, bottom),
            self.COLOR_CARD,
            -1,
        )

        cv2.circle(
            panel,
            (40, top + 21),
            7,
            color,
            -1,
            cv2.LINE_AA,
        )

        self._draw_text(
            panel,
            text,
            57,
            top + 27,
            0.44,
            self.COLOR_TEXT,
        )

    def _draw_message_card(
        self,
        panel: Any,
        overlay: AgentOverlay,
    ) -> None:
        top = 177
        bottom = 305

        cv2.rectangle(
            panel,
            (20, top),
            (self.PANEL_WIDTH - 20, bottom),
            self.COLOR_CARD,
            -1,
        )

        self._draw_text(
            panel,
            "ASSISTANT",
            36,
            top + 26,
            0.42,
            self.COLOR_PRIMARY,
            thickness=2,
        )

        lines = self._wrap_text(
            overlay.agent_message,
            maximum_characters=47,
        )

        for index, line in enumerate(
            lines[:5]
        ):
            self._draw_text(
                panel,
                line,
                36,
                top + 55 + index * 21,
                0.45,
                self.COLOR_TEXT,
            )

    def _draw_context_card(
        self,
        panel: Any,
        overlay: AgentOverlay,
    ) -> None:
        top = 317
        bottom = 403

        cv2.rectangle(
            panel,
            (20, top),
            (self.PANEL_WIDTH - 20, bottom),
            self.COLOR_CARD,
            -1,
        )

        comfort = overlay.comfort
        posture = overlay.posture

        overall_text = (
            f"{comfort.overall_score:.0f}/100"
            if comfort.overall_score is not None
            else "N/A"
        )

        upper_text = (
            f"{comfort.upper_body_score:.0f}"
            if comfort.upper_body_score is not None
            else "N/A"
        )

        lower_text = (
            f"{comfort.lower_body_score:.0f}"
            if comfort.lower_body_score is not None
            else "N/A"
        )

        pedal_text = (
            f"{comfort.pedal_reach_score:.0f}"
            if comfort.pedal_reach_score is not None
            else "N/A"
        )

        thigh_text = (
            f"{comfort.thigh_support_score:.0f}"
            if comfort.thigh_support_score is not None
            else "N/A"
        )

        knee_text = (
            f"{posture.average_knee_angle_deg:.0f} deg"
            if posture.average_knee_angle_deg is not None
            else "N/A"
        )

        if posture.full_body_visible:
            visibility_text = "Full body"
        elif posture.lower_body_visible:
            visibility_text = "Upper body + legs"
        elif posture.upper_body_visible:
            visibility_text = "Upper body only"
        else:
            visibility_text = "Limited view"

        score_color = self.COLOR_SUCCESS

        if comfort.overall_score is None:
            score_color = self.COLOR_MUTED
        elif comfort.overall_score < 60.0:
            score_color = self.COLOR_DANGER
        elif comfort.overall_score < 75.0:
            score_color = self.COLOR_WARNING

        self._draw_text(
            panel,
            (
                f"Comfort  {overall_text}  "
                f"{comfort.comfort_level}"
            ),
            34,
            top + 20,
            0.45,
            score_color,
            thickness=2,
        )

        self._draw_text(
            panel,
            (
                f"Upper body  {upper_text}     "
                f"Lower body  {lower_text}"
            ),
            34,
            top + 40,
            0.39,
            self.COLOR_TEXT,
        )

        self._draw_text(
            panel,
            (
                f"Pedal reach {pedal_text}     "
                f"Thigh support {thigh_text}"
            ),
            34,
            top + 59,
            0.38,
            self.COLOR_TEXT,
        )

        self._draw_text(
            panel,
            (
                f"Knee angle {knee_text}     "
                f"Confidence {comfort.confidence_score:.0f}%"
            ),
            34,
            top + 77,
            0.37,
            self.COLOR_MUTED,
        )

        visibility_color = (
            self.COLOR_SUCCESS
            if posture.lower_body_visible
            else self.COLOR_WARNING
        )

        cv2.circle(
            panel,
            (349, top + 19),
            5,
            visibility_color,
            -1,
            cv2.LINE_AA,
        )

        self._draw_text(
            panel,
            visibility_text,
            361,
            top + 23,
            0.31,
            visibility_color,
        )

    def _draw_buttons(
        self,
        panel: Any,
        overlay: AgentOverlay,
    ) -> None:
        state = overlay.agent_state

        if state == AgentState.MONITORING.value:
            self._add_button(
                panel,
                action="check_comfort",
                label="Start comfort check",
                row=0,
                column=0,
                width_columns=2,
                primary=True,
            )

        elif state == AgentState.WAITING_FOR_COMFORT.value:
            self._add_button(
                panel,
                action="comfortable",
                label="Yes, comfortable",
                row=0,
                column=0,
                primary=True,
            )

            self._add_button(
                panel,
                action="uncomfortable",
                label="Not comfortable",
                row=0,
                column=1,
            )

        elif state == AgentState.WAITING_FOR_COMPLAINT.value:

            self._add_button(
                panel,
                action="too_far",
                label="Too far from pedals",
                row=0,
                column=0,
            )

            self._add_button(
                panel,
                action="too_close",
                label="Too close to pedals",
                row=0,
                column=1,
            )

            self._add_button(
                panel,
                action="lower_back",
                label="Lower back",
                row=1,
                column=0,
            )

            self._add_button(
                panel,
                action="thigh_support",
                label="Thigh support",
                row=1,
                column=1,
            )

            self._add_button(
                panel,
                action="too_high",
                label="Seat too high",
                row=2,
                column=0,
            )

            self._add_button(
                panel,
                action="too_low",
                label="Seat too low",
                row=2,
                column=1,
            )

        elif state in {
            AgentState.WAITING_FOR_CONFIRMATION.value,
            AgentState.QUEUED_UNTIL_STOPPED.value,
        }:
            vehicle_parked = (
                overlay.vehicle.speed_kph <= 0.1
                and overlay.vehicle.parking_brake
            )

            self._add_button(
                panel,
                action="apply",
                label=(
                    "Apply simulated adjustment"
                    if vehicle_parked
                    else "Park vehicle to apply"
                ),
                row=0,
                column=0,
                width_columns=2,
                primary=vehicle_parked,
                disabled=not vehicle_parked,
            )

            self._add_button(
                panel,
                action="cancel",
                label="Cancel recommendation",
                row=1,
                column=0,
                width_columns=2,
                warning=True,
            )

        elif state == AgentState.ADJUSTMENT_APPLIED.value:
            self._add_button(
                panel,
                action="feels_better",
                label="Yes, this feels better",
                row=0,
                column=0,
                primary=True,
            )

            self._add_button(
                panel,
                action="adjust_again",
                label="Adjust again",
                row=0,
                column=1,
            )

        self._add_button(
            panel,
            action="moving",
            label="Simulate moving",
            row=3,
            column=0,
        )

        self._add_button(
            panel,
            action="parked",
            label="Simulate parked",
            row=3,
            column=1,
        )

    def _add_button(
        self,
        panel: Any,
        action: str,
        label: str,
        row: int,
        column: int,
        width_columns: int = 1,
        primary: bool = False,
        warning: bool = False,
        disabled: bool = False,
    ) -> None:
        margin = 20
        gap = 12
        button_height = 44
        panel_top = 418

        single_width = (
            self.PANEL_WIDTH
            - 2 * margin
            - gap
        ) // 2

        x1 = margin + column * (
            single_width + gap
        )

        if width_columns == 2:
            x2 = self.PANEL_WIDTH - margin
        else:
            x2 = x1 + single_width

        y1 = panel_top + row * (
            button_height + 9
        )
        y2 = y1 + button_height

        if disabled:
            fill_color = self.COLOR_DISABLED
            text_color = (120, 125, 135)
        elif primary:
            fill_color = self.COLOR_PRIMARY
            text_color = self.COLOR_TEXT
        elif warning:
            fill_color = (64, 65, 73)
            text_color = (190, 198, 255)
        else:
            fill_color = self.COLOR_CARD_LIGHT
            text_color = self.COLOR_TEXT

        cv2.rectangle(
            panel,
            (x1, y1),
            (x2, y2),
            fill_color,
            -1,
        )

        cv2.rectangle(
            panel,
            (x1, y1),
            (x2, y2),
            self.COLOR_BORDER,
            1,
        )

        font_scale = (
            0.40
            if len(label) > 22
            else 0.43
        )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )[0]

        text_x = x1 + max(
            8,
            (x2 - x1 - text_size[0]) // 2,
        )

        text_y = y1 + (
            button_height + text_size[1]
        ) // 2

        cv2.putText(
            panel,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

        self.buttons.append(
            InterfaceButton(
                action=action,
                label=label,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                primary=primary,
                warning=warning,
                disabled=disabled,
            )
        )

    def _draw_pose(
        self,
        image: Any,
        pose: PoseDetection,
        is_driver: bool,
    ) -> None:
        box_color = (
            self.COLOR_SUCCESS
            if is_driver
            else self.COLOR_WARNING
        )

        point_color = (
            (70, 105, 255)
            if is_driver
            else (255, 105, 220)
        )

        skeleton_color = (
            (255, 185, 65)
            if is_driver
            else (255, 220, 90)
        )

        if self.settings.draw_bounding_box:
            box = pose.bounding_box

            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            cv2.rectangle(
                image,
                (x1, max(0, y1 - 28)),
                (x1 + 145, y1),
                box_color,
                -1,
            )

            label = (
                "Driver"
                if is_driver
                else "Person"
            )

            cv2.putText(
                image,
                f"{label}  {box.confidence:.2f}",
                (x1 + 7, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        visible_points = {
            point.index: point
            for point in pose.keypoints
            if (
                point.confidence
                >= self.settings.keypoint_threshold
            )
        }

        if self.settings.draw_skeleton:
            for start, end in SKELETON_EDGES:
                start_point = visible_points.get(
                    start
                )
                end_point = visible_points.get(
                    end
                )

                if (
                    start_point is None
                    or end_point is None
                ):
                    continue

                cv2.line(
                    image,
                    (
                        int(start_point.x),
                        int(start_point.y),
                    ),
                    (
                        int(end_point.x),
                        int(end_point.y),
                    ),
                    skeleton_color,
                    2,
                    cv2.LINE_AA,
                )

        for point in visible_points.values():
            cv2.circle(
                image,
                (
                    int(point.x),
                    int(point.y),
                ),
                4,
                point_color,
                -1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _posture_label(
        posture: PostureMetrics,
    ) -> str:
        if not posture.valid:
            return "Limited view"

        labels: list[str] = []

        if posture.leaning_forward:
            labels.append("Forward")

        if posture.leaning_left:
            labels.append("Left")

        if posture.leaning_right:
            labels.append("Right")

        if posture.shoulders_unbalanced:
            labels.append("Shoulder tilt")

        if not labels:
            return "Neutral"

        return ", ".join(labels)

    @staticmethod
    def _draw_text(
        image: Any,
        text: str,
        x: int,
        y: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _wrap_text(
        text: str,
        maximum_characters: int,
    ) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current_line = ""

        for word in words:
            candidate = (
                word
                if not current_line
                else f"{current_line} {word}"
            )

            if len(candidate) <= maximum_characters:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)

                current_line = word

        if current_line:
            lines.append(current_line)

        return lines