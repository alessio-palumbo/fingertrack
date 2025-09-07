from dataclasses import dataclass

import numpy as np

from detector import FingersDetector, SwipeGestureDetector
from tracker import HandLabel, HandTracker


@dataclass(frozen=True)
class Hand:
    stable_fingers: tuple[int, int, int, int, int]
    gesture: str | None


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


class HandEngine:
    """
    Orchestrates hand tracking, finger detection, and gesture detection.

    Responsibilities:
      - Use HandTracker to parse frames into landmarks and handedness
      - Pass landmarks to FingersDetector and GestureDetector
      - Combine results into unified event objects
      - Dispatch events to registered consumers
    """

    def __init__(self, consumers=None):
        self.hand_tracker = HandTracker()
        self.fingers_detector = FingersDetector()
        self.gesture_detector = SwipeGestureDetector()
        self.consumers = consumers or []
        self.last_hand: dict[HandLabel, Hand | None] = {
            "Left": None,
            "Right": None,
        }

    def process_frame(self, frame):
        """
        Process a video frame: detect hands, analyze fingers & gestures,
        and notify consumers with the results.
        """
        hands_data = self.hand_tracker.detect(frame)
        if not hands_data:
            return

        any_change = False
        hand_event = HandEvent(
            hands=[],
            frame=frame,
        )

        for landmarks, hand_label in hands_data:
            stable_fingers = self.fingers_detector.process_hand(landmarks, hand_label)
            gesture = self.gesture_detector.process_hand(landmarks, hand_label)

            hand_event.hands.append(
                HandState(
                    label=hand_label,
                    stable_fingers=stable_fingers,
                    gesture=gesture,
                    landmarks=landmarks,
                )
            )

            hand = Hand(stable_fingers=stable_fingers, gesture=gesture)
            if self.last_hand.get(hand_label) != hand:
                any_change = True
            self.last_hand[hand_label] = hand

        for consumer in self.consumers:
            if consumer.always_consume or any_change:
                consumer.consume(hand_event)
