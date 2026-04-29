from dataclasses import dataclass

import numpy as np

from detector import Pointer
from motion import Motion
from tracker import HandLabel


@dataclass(frozen=True)
class HandState:
    label: HandLabel
    stable_fingers: tuple[int, int, int, int, int]
    pointer: Pointer | None
    gesture: str | None
    motion: Motion | None
    landmarks: object | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "label": self.label.lower(),
            "fingers": self.stable_fingers,
            "pointer": self.pointer.to_dict() if self.pointer else None,
            "gesture": self.gesture,
            "motion": self.motion.to_dict() if self.motion else None,
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
