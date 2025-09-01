import json

import cv2

from consumer import FingerStateEvent, get_consumers_from_args
from fingers_detector import FingersDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = FingersDetector()
    consumers = get_consumers_from_args(detector)

    # Fingers change tracking
    last_fingers = None
    # Track fingers history for smoothing
    fingers_history = []
    history_size = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for more natural video output.
        mirrored_frame = cv2.flip(frame, 1)

        hands_data = detector.detect(mirrored_frame)
        for landmarks, hand_label in hands_data:
            fingers = detector.fingers_up(landmarks, hand_label)

            # Smoothing / Debouncing
            fingers_history.append(tuple(fingers))
            if len(fingers_history) > history_size:
                fingers_history.pop(0)

            stable_fingers = max(set(fingers_history), key=fingers_history.count)
            if stable_fingers != last_fingers:
                event = FingerStateEvent(stable_fingers, frame, landmarks, hand_label)
                for consumer in consumers:
                    consumer.consume(event)
                last_fingers = stable_fingers

    cap.release()
    for consumer in consumers:
        if hasattr(consumer, "close"):
            consumer.close()


if __name__ == "__main__":
    main()
