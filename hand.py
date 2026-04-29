from dataclasses import dataclass

from interpreter import GestureEngine

from consumer import BaseConsumer
from detector import FingersDetector
from event import HandEvent, HandState
from motion import MotionEstimator
from tracker import HandLabel, HandTracker


@dataclass(frozen=True)
class Hand:
    stable_fingers: tuple[int, int, int, int, int]
    gesture: str | None


class HandEngine:
    """
    Orchestrates hand tracking, finger detection, and gesture detection.

    Responsibilities:
      - Use HandTracker to parse frames into landmarks and handedness
      - Pass landmarks to FingersDetector and GestureDetector
      - Combine results into unified event objects
      - Dispatch events to registered consumers
    """

    _MOTION_BUFFER_SIZE = 3

    def __init__(
        self,
        buffer_size: int = 5,
        frame_skip: int = 1,
        consumers: list[BaseConsumer] | None = None,
        gesture_engine: GestureEngine | None = None,
    ):
        self.buffer_size = buffer_size
        self.frame_skip = frame_skip
        self.hand_tracker = HandTracker()
        self.fingers_detector = FingersDetector(buffer_size=buffer_size)
        self.gesture_engine = gesture_engine or GestureEngine()
        self.motion_estimator = MotionEstimator(buffer_size=self._MOTION_BUFFER_SIZE)
        self.consumers = consumers or []
        self.frame_mod = 0
        self.last_hand: dict[HandLabel, Hand | None] = {
            "Left": None,
            "Right": None,
        }

    def process_frame(self, frame):
        """
        Process a video frame: detect hands, analyze fingers & gestures,
        and notify consumers with the results.
        """

        self.frame_mod = (self.frame_mod + 1) % self.frame_skip
        if self.frame_mod != 0:
            return

        hands_data = self.hand_tracker.detect(frame)

        any_change = False
        any_gesture = False

        hand_event = HandEvent(
            hands=[],
            frame=frame,
        )

        for landmarks, hand_label in hands_data:
            stable_fingers = self.fingers_detector.process_hand(landmarks, hand_label)
            pointer = self.fingers_detector.resolve_pointer(landmarks, stable_fingers)
            motion = self.motion_estimator.update(hand_label, pointer)
            gesture = self.gesture_engine.process(motion, hand_label)

            hand_event.hands.append(
                HandState(
                    label=hand_label,
                    stable_fingers=stable_fingers,
                    pointer=pointer,
                    gesture=gesture,
                    motion=motion,
                    landmarks=landmarks,
                )
            )

            hand = Hand(stable_fingers=stable_fingers, gesture=gesture)
            if self.last_hand.get(hand_label) != hand:
                any_change = True
            self.last_hand[hand_label] = hand

            if gesture:
                any_gesture = True

        should_emit = any_change or any_gesture

        for consumer in self.consumers:
            if consumer.always_consume or should_emit:
                consumer.consume(hand_event)
