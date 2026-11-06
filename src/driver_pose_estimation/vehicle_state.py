from __future__ import annotations

from threading import Lock

from .models import VehicleState


class VehicleStateProvider:
    def __init__(
        self,
        initial_speed_kph: float = 0.0,
        initial_parking_brake: bool = True,
    ) -> None:
        self._state = VehicleState(
            speed_kph=max(
                0.0,
                initial_speed_kph,
            ),
            parking_brake=initial_parking_brake,
            ignition_on=True,
        )

        self._lock = Lock()

    def get_state(self) -> VehicleState:
        with self._lock:
            return VehicleState(
                speed_kph=self._state.speed_kph,
                parking_brake=(
                    self._state.parking_brake
                ),
                ignition_on=self._state.ignition_on,
            )

    def set_moving(
        self,
        speed_kph: float = 30.0,
    ) -> None:
        with self._lock:
            self._state.speed_kph = max(
                1.0,
                speed_kph,
            )
            self._state.parking_brake = False

    def set_parked(self) -> None:
        with self._lock:
            self._state.speed_kph = 0.0
            self._state.parking_brake = True