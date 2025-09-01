import argparse
import json
from typing import Any, List, Protocol

import cv2
import requests
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

from fingers_detector import FingersDetector


class FingerStateEvent:
    def __init__(self, stable_fingers, frame=None, landmarks=None, hand_label=None):
        self.stable_fingers = stable_fingers
        self.frame = frame
        self.landmarks = landmarks
        self.hand_label = hand_label


class FingersConsumer(Protocol):
    """Defines an interface for consuming finger state outputs."""

    def consume(self, data: FingerStateEvent) -> None: ...


class StdoutConsumer:
    """Default consumer that prints finger states as JSON to stdout."""

    def consume(self, data: FingerStateEvent) -> None:
        print(json.dumps(data.stable_fingers), flush=True)


class HttpConsumer:
    """HTTP consumer that POST finger states as JSON to the set URL."""

    def __init__(self, url: str):
        self.url = url

    def consume(self, data: FingerStateEvent) -> None:
        requests.post(self.url, json=data.stable_fingers)


class OpenCVWindowConsumer:
    """Consumer that displays hand landmarks and finger states in a window."""

    def __init__(self, classify_fn=None, window_name="Gesture Detector"):
        self.classify_fn = classify_fn
        self.window_name = window_name

    def consume(self, data: FingerStateEvent):
        """
        data: tuple (frame, landmarks, hand_label, stable_fingers)
        """

        if data.landmarks:
            mp_draw.draw_landmarks(data.frame, data.landmarks, HAND_CONNECTIONS)

        # Use classify_fn if provided or fallback.
        stable_gesture = (
            self.classify_fn(data.stable_fingers)
            if self.classify_fn
            else str(data.stable_fingers)
        )

        cv2.putText(
            data.frame,
            f"{data.hand_label} - {stable_gesture}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow(self.window_name, data.frame)
        cv2.waitKey(1)

    def close(self):
        """Close any opened window"""
        cv2.destroyAllWindows()


def get_consumers_from_args(detector: FingersDetector) -> List[FingersConsumer]:
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
        consumers.append(OpenCVWindowConsumer(classify_fn=detector.classify_gesture))
    if args.consumer == "http":
        consumers.append(HttpConsumer(args.url))
    else:
        consumers.append(StdoutConsumer())
    return consumers
