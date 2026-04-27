from collections import deque
from dataclasses import dataclass
from typing import Literal

from tracker import HandLabel


@dataclass
class Pointer:
    """
    Represents a normalized 2D pointer derived from hand landmarks.

    Attributes:
        x (float): Horizontal position in normalized coordinates [0.0, 1.0],
                   where 0 is left and 1 is right of the frame.
        y (float): Vertical position in normalized coordinates [0.0, 1.0],
                   where 0 is top and 1 is bottom of the frame.

    Notes:
        - Pointer is gesture-independent and represents the current control position.
        - Consumers can use it for continuous interactions like cursor movement,
          dragging, or selection.
        - May be None when no valid pointing configuration is detected (e.g. fist).
    """

    x: float
    y: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return {
            "x": self.x,
            "y": self.y,
        }


class FingersDetector:
    """
    Analyzes hand landmarks to determine finger states (extended or folded).

    This class consumes hand landmark data (e.g., from HandTracker) and provides
    detection of which fingers are up for a given hand
    """

    THUMB_PIP = 3
    THUMB_TIP = 4

    PALM_CENTER = 9

    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    def __init__(
        self,
        buffer_size=5,
    ):
        self.buffer_size = buffer_size
        self.history: dict[HandLabel, deque] = {
            "Left": deque(maxlen=buffer_size),
            "Right": deque(maxlen=buffer_size),
        }

    def process_hand(self, landmarks, hand_label) -> tuple[int, int, int, int, int]:
        """
        Process a single hand and return stable fingers positions.
        """

        fingers = self.fingers_up(landmarks, hand_label)
        self.history[hand_label].append(fingers)

        # Compute stable fingers (most frequent in history)
        stable_fingers = max(
            set(self.history[hand_label]),
            key=self.history[hand_label].count,
        )
        return stable_fingers

    def fingers_up(self, landmarks, hand_label) -> tuple[int, int, int, int, int]:
        """
        Determine which fingers are up.
        Returns list: [thumb, index, middle, ring, pinky]
        """

        fingers = []
        landmark = landmarks.landmark

        # Thumb (x-axis check)
        if hand_label == "Right":
            fingers.append(
                1 if landmark[self.THUMB_TIP].x < landmark[self.THUMB_PIP].x else 0
            )
        else:  # Left hand
            fingers.append(
                1 if landmark[self.THUMB_TIP].x > landmark[self.THUMB_PIP].x else 0
            )

        # Other 4 fingers (y-axis check)
        tip_ids = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        for tip_id in tip_ids:
            tip = landmark[tip_id]
            pip = landmark[tip_id - 2]
            mcp = landmark[tip_id - 3]
            wrist = landmark[0]

            # Check both position of tip vs joint and relative distance to the writst.
            if tip.y < pip.y and pip.y < mcp.y and abs(tip.y - wrist.y) > 0.1:
                fingers.append(1)
            else:
                fingers.append(0)

        return tuple(fingers)

    def resolve_pointer(self, landmarks, fingers) -> Pointer | None:
        """
        Returns (x, y, source) or None
        """

        # One finger → index tip
        if fingers == (0, 1, 0, 0, 0):
            tip = landmarks.landmark[self.INDEX_TIP]
            return Pointer(tip.x, tip.y)

        # Two fingers → midpoint (index + middl tipse)
        if fingers == (0, 1, 1, 0, 0):
            i = landmarks.landmark[self.INDEX_TIP]
            m = landmarks.landmark[self.MIDDLE_TIP]
            x = (i.x + m.x) / 2
            y = (i.y + m.y) / 2
            return Pointer(x, y)

        # Open hand → palm cente
        if fingers == (1, 1, 1, 1, 1):
            palm = landmarks.landmark[self.PALM_CENTER]
            return Pointer(palm.x, palm.y)

        return None


class GestureDetector:
    def process_hand(self, landmarks, hand_label) -> str | None:
        raise NotImplementedError


class SwipeGestureDetector(GestureDetector):
    def __init__(self, buffer_size=5, threshold=0.1):
        """
        Detect a swipe gesture for movement normalized to [0,1] when past the threshold
        """

        self.buffer_size = buffer_size
        self.threshold = threshold
        self.history: dict[HandLabel, deque] = {
            "Left": deque(maxlen=buffer_size),
            "Right": deque(maxlen=buffer_size),
        }

    def process_hand(self, landmarks, hand_label) -> str | None:
        """
        Process a single hand gesture by comparing wrist offsets
        """

        wrist = landmarks.landmark[0]
        self.history[hand_label].append((wrist.x, wrist.y))

        if len(self.history[hand_label]) < self.buffer_size:
            return None

        x0, y0 = self.history[hand_label][0]
        x1, y1 = self.history[hand_label][-1]

        dx = x1 - x0
        dy = y1 - y0

        # Horizontal dominates
        if abs(dx) > abs(dy):
            if abs(dx) > self.threshold:
                return "swipe_right" if dx > 0 else "swipe_left"
        else:
            # Vertical dominates
            if abs(dy) > self.threshold:
                return "swipe_down" if dy > 0 else "swipe_up"

        return None
