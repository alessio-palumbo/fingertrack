from collections import deque
from typing import Literal

from tracker import HandLabel


class FingersDetector:
    """
    Analyzes hand landmarks to determine finger states (extended or folded).

    This class consumes hand landmark data (e.g., from HandTracker) and provides
    detection of which fingers are up for a given hand
    """

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
            fingers.append(1 if landmark[4].x < landmark[3].x else 0)
        else:  # Left hand
            fingers.append(1 if landmark[4].x > landmark[3].x else 0)

        # Other 4 fingers (y-axis check)
        tip_ids = [8, 12, 16, 20]
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
