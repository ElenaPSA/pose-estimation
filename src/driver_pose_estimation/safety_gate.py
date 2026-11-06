from __future__ import annotations

from dataclasses import dataclass

from .models import (
    SeatAdjustment,
    VehicleState,
)


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    reason: str


class SeatSafetyGate:
    MAXIMUM_INCREMENT = 1.0

    def evaluate(
        self,
        vehicle: VehicleState,
        adjustments: list[SeatAdjustment],
        driver_confirmed: bool,
    ) -> SafetyResult:
        if not adjustments:
            return SafetyResult(
                allowed=False,
                reason=(
                    "No seat adjustment is pending."
                ),
            )

        if not driver_confirmed:
            return SafetyResult(
                allowed=False,
                reason=(
                    "Explicit driver confirmation "
                    "is required."
                ),
            )

        if vehicle.speed_kph > 0.1:
            return SafetyResult(
                allowed=False,
                reason=(
                    "Adjustment blocked because "
                    "the vehicle is moving."
                ),
            )

        if not vehicle.parking_brake:
            return SafetyResult(
                allowed=False,
                reason=(
                    "Adjustment blocked because "
                    "the parking brake is not engaged."
                ),
            )

        if any(
            abs(adjustment.delta)
            > self.MAXIMUM_INCREMENT
            for adjustment in adjustments
        ):
            return SafetyResult(
                allowed=False,
                reason=(
                    "Adjustment blocked because an "
                    "increment exceeds the demo limit."
                ),
            )

        return SafetyResult(
            allowed=True,
            reason=(
                "Safety conditions are satisfied."
            ),
        )