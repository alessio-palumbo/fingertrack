from collections import deque

import cv2
import mediapipe as mp


class FingersStateEvent:
    def __init__(self, stable_fingers, frame=None, landmarks=None, hand_label=None):
        self.stable_fingers = stable_fingers
        self.frame = frame
        self.landmarks = landmarks
        self.hand_label = hand_label


class FingersDetector:
    """
    Detects finger states (extended or folded) from MediaPipe hand landmarks.

    This class provides methods to:
      - Detect which fingers are up for a given hand
      - Classify a simple gesture from the finger states (optional, for visualization)
      - Draw hand landmarks on frames for display

    Attributes:
        mp_hands: MediaPipe Hands solution reference
        mp_draw: MediaPipe Drawing utilities reference
        hands: The MediaPipe Hands instance used for detection
    """

    def __init__(
        self,
        max_hands=1,
        detection_conf=0.7,
        track_conf=0.7,
        history_size=5,
        consumers=None,
    ):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.history_size = history_size
        self.fingers_history = deque(maxlen=history_size)
        self.last_fingers = None
        self.consumers = consumers if consumers else []

    def detect(self, frame):
        """
        Run detection on a frame. Returns landmarks + handedness.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                hands_data.append((landmarks, label))
        return hands_data

    def process_hand(self, frame, landmarks, hand_label):
        """
        Process a single hand: smooth fingers and notify consumers.
        """
        fingers = self.fingers_up(landmarks, hand_label)
        # Convert to tuple for immutability in deque/set
        self.fingers_history.append(tuple(fingers))
        # Compute stable fingers (most frequent in history)
        stable_fingers = max(set(self.fingers_history), key=self.fingers_history.count)

        event = FingersStateEvent(stable_fingers, frame, landmarks, hand_label)
        for consumer in self.consumers:
            if consumer.always_consume or stable_fingers != self.last_fingers:
                consumer.consume(event)

        self.last_fingers = stable_fingers

    def fingers_up(self, landmarks, hand_label):
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

        return fingers

    @staticmethod
    def classify_gesture(fingers):
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
