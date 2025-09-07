import argparse
import json
from typing import List

import cv2
import requests
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

from hand import HandEvent


class BaseConsumer:
    """Defines consumers common properties"""

    @property
    def always_consume(self) -> bool:
        """Defines whether consuming frames with repeated data"""
        return False

    def consume(self, _: HandEvent):
        raise NotImplementedError


class StdoutConsumer(BaseConsumer):
    """Default consumer that prints finger states as JSON to stdout."""

    def consume(self, event: HandEvent) -> None:
        print(json.dumps(event.to_dict()), flush=True)


class HttpConsumer(BaseConsumer):
    """HTTP consumer that POST finger states as JSON to the set URL."""

    def __init__(self, url: str):
        self.url = url

    def consume(self, event: HandEvent) -> None:
        try:
            requests.post(self.url, json=event.to_dict(), timeout=0.5)
        except requests.RequestException as e:
            print(f"Failed to send event: {e}")


class OpenCVWindowConsumer(BaseConsumer):
    """Consumer that displays hand landmarks and finger states in a window."""

    def __init__(self, window_name="Gesture Detector"):
        self.window_name = window_name

    @property
    def always_consume(self) -> bool:
        return True

    def consume(self, event: HandEvent):
        frame = event.frame
        if frame is None:
            return

        for i, hand in enumerate(event.hands):
            if hand.landmarks:
                mp_draw.draw_landmarks(frame, hand.landmarks, HAND_CONNECTIONS)

            stable_gesture = self.classify_gesture(hand.stable_fingers)

            y_pos = 50 + i * 40
            cv2.putText(
                frame,
                f"{hand.label}: {stable_gesture} - Gesture: {hand.gesture}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0) if hand.label == "Left" else (0, 255, 0),
                2,
            )

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def classify_gesture(self, fingers):
        """
        Map finger states to gesture name.
        Extend this dictionary for more gestures.
        """
        gesture_map = {
            (0, 1, 0, 0, 0): "Pointing (Index)",
            (0, 1, 1, 0, 0): "Victory Sign",
            (0, 1, 1, 1, 0): "Three-Finger Salute",
            (0, 1, 0, 0, 1): "Horns",
            (1, 1, 0, 0, 1): "I love you",
            (0, 0, 1, 0, 0): "Rude!!!",
            (0, 0, 0, 0, 0): "Fist",
            (1, 0, 0, 0, 0): "Thumbs Up",
            (1, 1, 1, 1, 1): "Open Palm",
            (1, 0, 0, 0, 1): "Shaka Sign",
        }
        return gesture_map.get(tuple(fingers), f"{sum(fingers)} fingers")


def get_consumers_from_args() -> List[BaseConsumer]:
    """Returns the consumers as defined in the args or the default"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--consumer",
        choices=["stdout", "http"],
        default="stdout",
        help="Select output consumer (default: stdout)",
    )
    parser.add_argument("--url", help="URL for http consumer")
    parser.add_argument(
        "--show-window", action="store_true", help="Display OpenCV feed"
    )
    args = parser.parse_args()

    consumers = []
    if args.show_window:
        consumers.append(OpenCVWindowConsumer())
    if args.consumer == "http":
        consumers.append(HttpConsumer(args.url))
    else:
        consumers.append(StdoutConsumer())
    return consumers
