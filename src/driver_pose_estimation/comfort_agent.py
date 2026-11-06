from __future__ import annotations

from enum import Enum

from .models import (
    AgentDecision,
    PostureMetrics,
    SeatAdjustment,
)
from .safety_gate import SeatSafetyGate
from .seat_controller import SimulatedSeatController
from .seat_recommender import SeatRecommender
from .vehicle_state import VehicleStateProvider


class AgentState(str, Enum):
    MONITORING = "MONITORING"
    WAITING_FOR_COMFORT = "WAITING_FOR_COMFORT"
    WAITING_FOR_COMPLAINT = "WAITING_FOR_COMPLAINT"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    QUEUED_UNTIL_STOPPED = "QUEUED_UNTIL_STOPPED"
    ADJUSTMENT_APPLIED = "ADJUSTMENT_APPLIED"


class ComfortAgent:
    def __init__(
        self,
        recommender: SeatRecommender,
        safety_gate: SeatSafetyGate,
        seat_controller: SimulatedSeatController,
        vehicle_state: VehicleStateProvider,
    ) -> None:
        self.recommender = recommender
        self.safety_gate = safety_gate
        self.seat_controller = seat_controller
        self.vehicle_state = vehicle_state

        self.state = AgentState.MONITORING

        self.latest_posture = PostureMetrics(
            valid=False
        )

        self.pending_adjustments: list[
            SeatAdjustment
        ] = []

        self.message = (
            "I can help review your seating comfort. "
            "Select Start comfort check when you are ready."
        )

    def update_posture(
        self,
        posture: PostureMetrics,
    ) -> None:
        self.latest_posture = posture

    def ask_about_comfort(
        self,
    ) -> AgentDecision:
        self.pending_adjustments = []
        self.state = AgentState.WAITING_FOR_COMFORT

        self.message = (
            "Are you comfortable in your current "
            "seating position?"
        )

        return AgentDecision(
            message=self.message
        )

    def answer_comfortable(
        self,
    ) -> AgentDecision:
        if self.state != AgentState.WAITING_FOR_COMFORT:
            self.message = (
                "Please start a comfort check first."
            )

            return AgentDecision(
                message=self.message
            )

        self.pending_adjustments = []
        self.state = AgentState.MONITORING

        self.message = (
            "Great. No adjustment is needed. "
            "I will continue observing your posture."
        )

        return AgentDecision(
            message=self.message
        )

    def answer_uncomfortable(
        self,
    ) -> AgentDecision:
        if self.state != AgentState.WAITING_FOR_COMFORT:
            self.message = (
                "Please start a comfort check first."
            )

            return AgentDecision(
                message=self.message
            )

        self.state = AgentState.WAITING_FOR_COMPLAINT

        self.message = (
            "What feels uncomfortable? "
            "Choose the option that best matches your experience."
        )

        return AgentDecision(
            message=self.message
        )

    def submit_complaint(
        self,
        complaint: str,
    ) -> AgentDecision:
        if self.state != AgentState.WAITING_FOR_COMPLAINT:
            self.message = (
                "Please first indicate that you are "
                "not comfortable."
            )

            return AgentDecision(
                message=self.message
            )

        recommendations = self.recommender.recommend(
            posture=self.latest_posture,
            complaint=complaint,
        )

        if not recommendations:
            self.pending_adjustments = []
            self.state = AgentState.MONITORING

            self.message = (
                "I could not identify a suitable adjustment "
                "for that concern. No movement was requested."
            )

            return AgentDecision(
                message=self.message
            )

        self.pending_adjustments = recommendations
        vehicle = self.vehicle_state.get_state()

        recommendation_text = self._format_recommendations(
            recommendations
        )

        perception_note = (
            " I also used the available upper-body pose information."
            if self.latest_posture.valid
            else (
                " The recommendation is based on your reported "
                "discomfort because pose visibility is limited."
            )
        )

        if (
            vehicle.speed_kph > 0.1
            or not vehicle.parking_brake
        ):
            self.state = AgentState.QUEUED_UNTIL_STOPPED

            self.message = (
                f"I recommend {recommendation_text}."
                f"{perception_note} "
                "The adjustment is locked while moving. "
                "Park the vehicle before applying it."
            )

            return AgentDecision(
                message=self.message,
                adjustments=recommendations,
                requires_confirmation=True,
                queued_until_stopped=True,
            )

        self.state = AgentState.WAITING_FOR_CONFIRMATION

        self.message = (
            f"I recommend {recommendation_text}."
            f"{perception_note} "
            "Would you like to apply the simulated adjustment?"
        )

        return AgentDecision(
            message=self.message,
            adjustments=recommendations,
            requires_confirmation=True,
        )

    def confirm_adjustment(
        self,
    ) -> AgentDecision:
        safety = self.safety_gate.evaluate(
            vehicle=self.vehicle_state.get_state(),
            adjustments=self.pending_adjustments,
            driver_confirmed=True,
        )

        if not safety.allowed:
            if self.pending_adjustments:
                self.state = AgentState.QUEUED_UNTIL_STOPPED

            self.message = (
                f"{safety.reason} "
                "The recommendation remains unapplied."
            )

            return AgentDecision(
                message=self.message,
                adjustments=list(
                    self.pending_adjustments
                ),
                queued_until_stopped=True,
            )

        self.seat_controller.apply(
            self.pending_adjustments
        )

        self.pending_adjustments = []
        self.state = AgentState.ADJUSTMENT_APPLIED

        self.message = (
            "The simulated adjustment has been applied. "
            "Does this position feel better?"
        )

        return AgentDecision(
            message=self.message
        )

    def confirm_improvement(
        self,
    ) -> AgentDecision:
        self.pending_adjustments = []
        self.state = AgentState.MONITORING

        self.message = (
            "Excellent. I will keep this simulated "
            "seat position."
        )

        return AgentDecision(
            message=self.message
        )

    def request_another_adjustment(
        self,
    ) -> AgentDecision:
        self.pending_adjustments = []
        self.state = AgentState.WAITING_FOR_COMPLAINT

        self.message = (
            "Understood. What still feels uncomfortable?"
        )

        return AgentDecision(
            message=self.message
        )

    def cancel_adjustment(
        self,
    ) -> AgentDecision:
        self.pending_adjustments = []
        self.state = AgentState.MONITORING

        self.message = (
            "The recommendation was cancelled. "
            "The simulated seat was not changed."
        )

        return AgentDecision(
            message=self.message
        )

    @staticmethod
    def _format_recommendations(
        recommendations: list[SeatAdjustment],
    ) -> str:
        readable: list[str] = []

        for recommendation in recommendations:
            axis = recommendation.axis.value

            if axis == "fore_aft":
                direction = (
                    "forward"
                    if recommendation.delta > 0
                    else "backward"
                )

                readable.append(
                    f"a small {direction} seat movement"
                )

            elif axis == "height":
                direction = (
                    "upward"
                    if recommendation.delta > 0
                    else "downward"
                )

                readable.append(
                    f"a small {direction} height adjustment"
                )

            elif axis == "backrest":
                readable.append(
                    "a small backrest adjustment"
                )

            elif axis == "lumbar":
                readable.append(
                    "a small lumbar-support adjustment"
                )

            elif axis == "cushion_tilt":
                readable.append(
                    "a small seat-cushion tilt adjustment"
                )

        if not readable:
            return "no adjustment"

        if len(readable) == 1:
            return readable[0]

        return " and ".join(readable)