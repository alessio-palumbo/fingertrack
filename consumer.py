import argparse
import json
from typing import List

import cv2
import requests
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

from fingers_detector import FingersDetector, FingersStateEvent


class BaseConsumer:
    """Defines consumers common properties"""

    @property
    def always_consume(self) -> bool:
        """Defines whether consuming frames with repeated data"""
        return False

    def consume(self, event):
        raise NotImplementedError

    def close(self):
        pass


class StdoutConsumer(BaseConsumer):
    """Default consumer that prints finger states as JSON to stdout."""

    def consume(self, event: FingersStateEvent) -> None:
        print(json.dumps(event.stable_fingers), flush=True)


class HttpConsumer(BaseConsumer):
    """HTTP consumer that POST finger states as JSON to the set URL."""

    def __init__(self, url: str):
        self.url = url

    def consume(self, event: FingersStateEvent) -> None:
        requests.post(self.url, json=event.stable_fingers)


class OpenCVWindowConsumer(BaseConsumer):
    """Consumer that displays hand landmarks and finger states in a window."""

    def __init__(self, window_name="Gesture Detector"):
        self.window_name = window_name

    @property
    def always_consume(self) -> bool:
        return True

    def consume(self, event: FingersStateEvent):
        """
        data: tuple (frame, landmarks, hand_label, stable_fingers)
        """

        if event.landmarks:
            mp_draw.draw_landmarks(event.frame, event.landmarks, HAND_CONNECTIONS)

        stable_gesture = FingersDetector.classify_gesture(event.stable_fingers)

        cv2.putText(
            event.frame,
            f"{event.hand_label} - {stable_gesture}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow(self.window_name, event.frame)
        cv2.waitKey(1)

    def close(self):
        """Close any opened window"""
        cv2.destroyAllWindows()


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
