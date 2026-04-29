from collections import deque
from dataclasses import dataclass

from tracker import HandLabel


@dataclass(frozen=True)
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

    WRIST = 0

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
        Resolve pointer dynamically based on extended fingers.

        Rules:
        - 0 fingers → wrist
        - 1 finger → that fingertip
        - 2+ fingers → centroid of extended fingers
        - open hand → ignore thumb for stability
        """

        tips_map = [
            (0, self.THUMB_TIP),
            (1, self.INDEX_TIP),
            (2, self.MIDDLE_TIP),
            (3, self.RING_TIP),
            (4, self.PINKY_TIP),
        ]

        extended = [landmarks.landmark[idx] for i, idx in tips_map if fingers[i] == 1]

        # No fingers → wrist
        if not extended:
            wrist = landmarks.landmark[self.WRIST]
            return Pointer(wrist.x, wrist.y)

        # One finger → direct pointer
        if len(extended) == 1:
            tip = extended[0]
            return Pointer(tip.x, tip.y)

        # Open hand → ignore thumb if all fingers extended
        if sum(fingers) >= 4 and fingers[0] == 1:
            extended = [
                landmarks.landmark[idx]
                for i, idx in tips_map
                if i != 0 and fingers[i] == 1  # exclude thumb
            ]

        # Centroid of extended fingers
        x = sum(p.x for p in extended) / len(extended)
        y = sum(p.y for p in extended) / len(extended)

        return Pointer(x, y)
