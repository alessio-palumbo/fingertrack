from typing import Literal

import cv2
import mediapipe as mp

HandLabel = Literal["Left", "Right"]


class HandTracker:
    """
    Wrapper around MediaPipe Hands for detecting and tracking hand landmarks.

    Args:
        max_hands (int): Maximum number of hands to detect (default: 2).
        detection_conf (float): Minimum confidence for hand detection (default: 0.7).
        track_conf (float): Minimum confidence for hand landmark tracking (default: 0.7).

    Attributes:
        hands (mp.solutions.hands.Hands): MediaPipe hands detector instance.
        mp_draw (module): MediaPipe drawing utilities for rendering landmarks.
    """

    def __init__(
        self,
        max_hands=2,
        detection_conf=0.7,
        track_conf=0.7,
    ):
        self.max_hands = max_hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame) -> list[tuple[object, HandLabel]]:
        """
        Run hand detection on a single video frame.

        Args:
            frame (numpy.ndarray):
                A BGR image (as returned by OpenCV `cv2.VideoCapture.read`).

        Returns:
            list[tuple[mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, str]]:
                A list of detected hands. Each entry is a tuple:
                    - landmarks: a NormalizedLandmarkList with 21 landmarks
                      (x, y in [0,1] relative to image width/height, z relative to depth).
                    - hand_label: "Left" or "Right" depending on handedness classification.

                Returns an empty list if no hands are detected.

        Notes:
            - The function internally converts the BGR frame to RGB for MediaPipe.
            - Coordinates are normalized and must be scaled to pixel values if needed.
            - Multiple hands may be returned in a single frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                hands_data.append((landmarks, label))
        return hands_data
