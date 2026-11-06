from driver_pose_estimation.models import (
    SeatAdjustment,
    SeatAxis,
    VehicleState,
)
from driver_pose_estimation.safety_gate import (
    SeatSafetyGate,
)


def test_adjustment_is_blocked_while_moving() -> None:
    gate = SeatSafetyGate()

    result = gate.evaluate(
        vehicle=VehicleState(
            speed_kph=30.0,
            parking_brake=False,
        ),
        adjustments=[
            SeatAdjustment(
                axis=SeatAxis.LUMBAR,
                delta=1.0,
                reason="Test",
            )
        ],
        driver_confirmed=True,
    )

    assert result.allowed is False


def test_adjustment_is_allowed_when_parked() -> None:
    gate = SeatSafetyGate()

    result = gate.evaluate(
        vehicle=VehicleState(
            speed_kph=0.0,
            parking_brake=True,
        ),
        adjustments=[
            SeatAdjustment(
                axis=SeatAxis.LUMBAR,
                delta=1.0,
                reason="Test",
            )
        ],
        driver_confirmed=True,
    )

    assert result.allowed is True