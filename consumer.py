import json
from enum import Enum

import cv2
import numpy as np
import requests
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

from event import HandEvent


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


class PreviewMode(Enum):
    """
    Controls how the OpenCV preview renders hand tracking output.
    FULL:
        Shows the original camera frame with MediaPipe landmarks and overlays.
    LANDMARKS:
        Renders only the hand skeleton on a black background (no camera feed).
    """

    FULL = "full"
    LANDMARKS = "landmarks"


class OpenCVWindowConsumer(BaseConsumer):
    """Consumer that displays hand landmarks and finger states in a window."""

    def __init__(self, window_name="Gesture Detector", mode=PreviewMode.FULL):
        self.window_name = window_name
        self.mode = mode

    @property
    def always_consume(self) -> bool:
        return True

    def consume(self, event: HandEvent):
        frame = event.frame
        if frame is None:
            return

        if self.mode == PreviewMode.LANDMARKS:
            height, width, _ = event.frame.shape
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        if not event.hands:
            self._draw_text(frame, "No hands detected", center=True)
        else:
            for i, hand in enumerate(event.hands):
                if hand.landmarks:
                    mp_draw.draw_landmarks(frame, hand.landmarks, HAND_CONNECTIONS)

                stable_gesture = self.classify_gesture(hand.stable_fingers)
                text = f"{hand.label}: {stable_gesture} - Gesture: {hand.gesture}"

                self._draw_text(frame, text, (10, 50 + i * 40))

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

    def _draw_text(
        self,
        frame,
        text: str,
        position: tuple[int, int] | None = None,
        center: bool = False,
        color: tuple[int, int, int] = (255, 255, 255),
        scale: float = 1.0,
        thickness: int = 2,
    ) -> None:
        """
        Draw text on the given frame with optional positioning or centering.

        Args:
            frame (np.ndarray): Image to draw on.
            text (str): Text to render.
            position (tuple[int, int] | None): Top-left position (x, y). Ignored if center=True.
            center (bool): If True, text is centered on the frame.
            color (tuple[int, int, int]): Text color in BGR.
            scale (float): Font scale.
            thickness (int): Text thickness.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w, _ = frame.shape

        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size

        if center:
            x = (w - text_w) // 2
            y = (h + text_h) // 2
        else:
            x, y = position or (10, 30)

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
