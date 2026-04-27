"""
CLIENT USAGE GUIDE FOR MOTION

Motion values are raw, normalized, and device-agnostic. They describe movement,
not intent.

Coordinate system:
- x: left (-) → right (+)
- y: up (-) → down (+)
- range: approximately [-1, 1] depending on frame-to-frame movement

Typical interpretation patterns:

1. Continuous control (recommended)
   - use vy for vertical scrolling
   - use vx for horizontal movement or navigation

2. Speed scaling
   - abs(vx), abs(vy) represent intensity
   - can be mapped to acceleration or repeat rate

3. Direction detection
   - vy > 0 → moving down
   - vy < 0 → moving up

Important:
- Motion is NOT gesture-level intent (e.g. swipe)
- Motion is NOT stabilized for UI decisions
- Clients are responsible for smoothing, thresholds, and mapping to actions
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class Motion:
    """
    Represents the estimated motion of a hand over time.
    Attributes:
        vx (float): Horizontal velocity component (normalized, per frame window).
                    Positive values indicate movement to the right, negative to the left.
        vy (float): Vertical velocity component (normalized, per frame window).
                    Positive values indicate movement downward, negative upward.
        speed (float): Magnitude of the velocity vector (sqrt(vx^2 + vy^2)),
                       representing overall movement intensity.
    """

    vx: float
    vy: float
    speed: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "vx": self.vx,
            "vy": self.vy,
            "speed": self.speed,
        }


class MotionEstimator:
    """
    Estimates hand motion (velocity) over time using a short history of pointer positions.

    This class maintains a per-hand buffer of recent pointer positions and computes
    velocity as the average of frame-to-frame deltas. This provides a balance between
    responsiveness and noise reduction.

    Motion values are normalized in the same coordinate space as the input pointer
    (typically [0,1]), making them device-agnostic. Interpretation of these values
    (e.g., scrolling speed, cursor movement) is left to downstream consumers.

    Args:
        buffer_size (int): Number of recent pointer samples to keep per hand.
                           Larger values smooth motion but may reduce responsiveness.
    """

    def __init__(self, buffer_size=3):
        self.history = {
            "Left": deque(maxlen=buffer_size),
            "Right": deque(maxlen=buffer_size),
        }

    def update(self, hand_label, pointer) -> Motion | None:
        """
        Update motion estimation for a given hand using the latest pointer position.

        The method appends the current pointer position to the hand's history and computes
        velocity as the average of consecutive deltas across the buffer.

        If insufficient data is available (fewer than 2 samples), or if the pointer is not
        provided (e.g., hand configuration does not support pointing), no motion is returned.

        Args:
            hand_label (str): Identifier for the hand (e.g., "Left", "Right").
            pointer (Pointer | None): Current pointer position. Must expose `x` and `y`
                                     attributes in normalized coordinates [0,1].

        Returns:
            Motion | None: Estimated motion for the hand, or None if unavailable.
        """
        if pointer is None:
            return None

        h = self.history[hand_label]
        h.append((pointer.x, pointer.y))

        if len(h) < 2:
            return None

        dxs = []
        dys = []

        for i in range(1, len(h)):
            dxs.append(h[i][0] - h[i - 1][0])
            dys.append(h[i][1] - h[i - 1][1])

        vx = sum(dxs) / len(dxs)
        vy = sum(dys) / len(dys)
        speed = math.sqrt(vx * vx + vy * vy)

        return Motion(vx=vx, vy=vy, speed=speed)
