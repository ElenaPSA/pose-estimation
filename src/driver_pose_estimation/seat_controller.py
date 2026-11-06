from __future__ import annotations

import logging
from dataclasses import replace

from .models import (
    SeatAdjustment,
    SeatAxis,
    SeatState,
)


logger = logging.getLogger(__name__)


class SimulatedSeatController:
    def __init__(self) -> None:
        self._state = SeatState()

    @property
    def state(self) -> SeatState:
        return replace(self._state)

    def apply(
        self,
        adjustments: list[SeatAdjustment],
    ) -> SeatState:
        for adjustment in adjustments:
            self._apply_one(adjustment)

            logger.info(
                "SIMULATED SEAT COMMAND | "
                "axis=%s delta=%.2f reason=%s",
                adjustment.axis.value,
                adjustment.delta,
                adjustment.reason,
            )

        return self.state

    def _apply_one(
        self,
        adjustment: SeatAdjustment,
    ) -> None:
        if adjustment.axis == SeatAxis.FORE_AFT:
            self._state.fore_aft = self._clamp(
                self._state.fore_aft
                + adjustment.delta
            )

        elif adjustment.axis == SeatAxis.HEIGHT:
            self._state.height = self._clamp(
                self._state.height
                + adjustment.delta
            )

        elif adjustment.axis == SeatAxis.BACKREST:
            self._state.backrest = self._clamp(
                self._state.backrest
                + adjustment.delta
            )

        elif adjustment.axis == SeatAxis.LUMBAR:
            self._state.lumbar = self._clamp(
                self._state.lumbar
                + adjustment.delta
            )

        elif adjustment.axis == SeatAxis.CUSHION_TILT:
            self._state.cushion_tilt = self._clamp(
                self._state.cushion_tilt
                + adjustment.delta
            )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            -5.0,
            min(5.0, value),
        )