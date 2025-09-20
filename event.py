from dataclasses import dataclass

import numpy as np

from tracker import HandLabel


@dataclass(frozen=True)
class HandState:
    label: HandLabel
    stable_fingers: tuple[int, int, int, int, int]
    gesture: str | None
    landmarks: object | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "label": self.label.lower(),
            "fingers": self.stable_fingers,
            "gesture": self.gesture,
        }


@dataclass
class HandEvent:
    """
    Unified event: contains finger states and/or gestures
    detected in the same frame.
    """

    hands: list[HandState]
    frame: np.ndarray | None = None  # OpenCV frame

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary containing and event"""
        return {
            "hands": [hand.to_dict() for hand in self.hands],
        }
